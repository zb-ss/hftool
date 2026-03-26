"""Vision-language task handler.

Supports Qwen VL models for analyzing images, with multi-GPU support.
Primarily used for video keyframe analysis in the voiceover pipeline.
"""

import gc
from typing import Any, Dict, Optional

import torch
from hftool.tasks.base import BaseTask


def _patch_qwen_vl_for_rocm(model) -> None:
    """Patch torch.repeat_interleave for ROCm compatibility.

    torch.repeat_interleave with CUDA tensors crashes on ROCm HIP
    (hipErrorIllegalState). This globally patches it to move tensors
    to CPU, run repeat_interleave there, then move back to the
    original device. Only activated when ROCm is detected.
    """
    _original_repeat_interleave = torch.repeat_interleave

    def _safe_repeat_interleave(input, repeats, *args, **kwargs):
        # Only patch when both args are CUDA tensors (the problematic case)
        if isinstance(repeats, torch.Tensor) and repeats.is_cuda:
            device = input.device
            result = _original_repeat_interleave(
                input.cpu(), repeats.cpu(), *args, **kwargs
            )
            return result.to(device)
        return _original_repeat_interleave(input, repeats, *args, **kwargs)

    torch.repeat_interleave = _safe_repeat_interleave


class VisionLanguageTask(BaseTask):
    """Handler for vision-language model inference.

    Accepts an image and a text prompt, returns a text response.
    Designed for keyframe analysis but usable for any VLM task.

    Supported models:
    - Qwen/Qwen2.5-VL-7B-Instruct (default)
    - Any AutoModelForCausalLM-compatible VLM
    """

    def __init__(self, device: str = "auto", dtype: Optional[str] = None):
        super().__init__(device=device, dtype=dtype)
        self._model = None
        self._processor = None

    def load_pipeline(self, model: str, **kwargs) -> Dict[str, Any]:
        """Load a VLM pipeline.

        Args:
            model: HuggingFace model repo or local path
            **kwargs: Forwarded to from_pretrained()

        Returns:
            Dict with "type", "model", "processor", and "device" keys
        """
        from hftool.utils.deps import check_dependencies
        check_dependencies(["transformers", "torch", "PIL"], extra="with_vlm")

        from transformers import AutoProcessor
        # AutoModelForImageTextToText resolves the correct VLM class:
        # - Qwen3-VL → Qwen2_5_VLForConditionalGeneration (full attention, works on all GPUs)
        # - Qwen3.5 → Qwen3_5ForConditionalGeneration (GatedDeltaNet, Instinct GPUs only)
        # AutoModelForCausalLM would strip the vision encoder (text-only).
        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:
            # Older transformers fallback
            from transformers import AutoModelForCausalLM as AutoVLM

        from hftool.core.device import (
            configure_rocm_env,
            detect_device,
            get_multi_gpu_kwargs,
            get_optimal_dtype,
        )

        configure_rocm_env()

        device = self.device if self.device != "auto" else detect_device()

        if self.dtype:
            import torch
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(self.dtype, torch.float32)
        else:
            dtype = get_optimal_dtype(device)

        gpu_config = get_multi_gpu_kwargs()
        load_kwargs: Dict[str, Any] = {"torch_dtype": dtype}
        if gpu_config["use_multi_gpu"]:
            load_kwargs["device_map"] = gpu_config["device_map"]
            load_kwargs["max_memory"] = gpu_config["max_memory"]
        else:
            load_kwargs["device_map"] = device

        load_kwargs.update(kwargs)

        # Apply ROCm workaround BEFORE loading — torch.repeat_interleave
        # crashes with CUDA tensors on ROCm HIP (hipErrorIllegalState)
        if device in ("cuda", "rocm"):
            is_rocm = getattr(torch.version, "hip", None) is not None
            if is_rocm:
                _patch_qwen_vl_for_rocm(None)

        model_obj = AutoVLM.from_pretrained(model, **load_kwargs)
        processor = AutoProcessor.from_pretrained(model)

        self._model = model_obj
        self._processor = processor

        pipeline = {
            "type": "qwen3.5",
            "model": model_obj,
            "processor": processor,
            "device": device,
        }
        self._pipeline = pipeline
        return pipeline

    def run_inference(self, pipeline: Dict[str, Any], input_data: Dict[str, Any], **kwargs) -> Dict[str, str]:
        """Run VLM inference on an image + prompt.

        Args:
            pipeline: Loaded pipeline dict from load_pipeline()
            input_data: Dict with "image_path" (str) and "prompt" (str)
            **kwargs: Additional generate() arguments

        Returns:
            Dict with "text" key containing the model response
        """
        import torch
        from PIL import Image

        model_obj = pipeline["model"]
        processor = pipeline["processor"]

        prompt = input_data["prompt"]
        image_path = input_data.get("image_path")

        # System prompt to disable thinking mode and get direct responses
        system_msg = {
            "role": "system",
            "content": "You are a concise assistant. Answer directly without thinking blocks, analysis headers, or bullet points. No <think> tags.",
        }

        # Text-only mode (no image) — used for script assembly prompts
        if not image_path:
            messages = [
                system_msg,
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], return_tensors="pt", padding=True)
            inputs = inputs.to(model_obj.device)
        else:
            image = Image.open(image_path).convert("RGB")

            messages = [
                system_msg,
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # Let the processor handle resizing with constrained token budget.
            # max_pixels controls vision token count and VRAM during attention.
            # 262144 (~512x512) keeps attention under 24GB VRAM. Higher values
            # cause OOM in the vision encoder's scaled_dot_product_attention.
            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True,
                min_pixels=28 * 28,
                max_pixels=262144,
            )
            inputs = inputs.to(model_obj.device)

        max_new_tokens = kwargs.pop("max_new_tokens", 1024)

        with torch.no_grad():
            # Multimodal models pass extra keys (pixel_values, image_grid_thw,
            # mm_token_type_ids) that are consumed by prepare_inputs_for_generation
            # but not declared in forward(). Transformers 5.x validates strictly,
            # so we temporarily disable the check for VLM inference.
            original_validate = getattr(model_obj, '_validate_model_kwargs', None)
            model_obj._validate_model_kwargs = lambda model_kwargs: None
            try:
                output_ids = model_obj.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)
            finally:
                if original_validate is not None:
                    model_obj._validate_model_kwargs = original_validate

        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Strip thinking artifacts from Qwen 3.5 responses
        import re
        # Remove <think>...</think> blocks
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        # Remove common analysis preambles the model prepends
        response = re.sub(
            r"^(The user wants.*?\n\n|.*?\*\*Analyze.*?\*\*\n)",
            "", response, flags=re.DOTALL,
        )
        response = response.strip()

        return {"text": response.strip()}

    def analyze_frame(
        self,
        image_path: str,
        prompt: str,
        previous_context: str = "",
    ) -> str:
        """Convenience wrapper for single-frame analysis.

        Substitutes {previous_description} in the prompt with previous_context,
        then runs inference and returns the text response.

        Args:
            image_path: Path to the image file
            prompt: Prompt template; may contain {previous_description}
            previous_context: Text from the previously analyzed frame

        Returns:
            Model response as a plain string
        """
        resolved_prompt = prompt.replace("{previous_description}", previous_context)
        result = self.run_inference(
            self._pipeline,
            {"image_path": image_path, "prompt": resolved_prompt},
        )
        return result["text"]

    def save_output(self, result: Dict[str, str], output_path: str, **kwargs) -> str:
        """Write the text response to a file.

        Args:
            result: Dict with "text" key from run_inference()
            output_path: Destination file path
            **kwargs: Unused

        Returns:
            output_path
        """
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(result["text"])
        return output_path

    def cleanup(self) -> None:
        """Release VRAM by deleting the model and processor.

        Should be called before loading the TTS pipeline to free GPU memory.
        """
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._pipeline = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def create_task(device: str = "auto", dtype: Optional[str] = None) -> VisionLanguageTask:
    """Factory function to create a VisionLanguageTask.

    Args:
        device: Device to run on ("auto", "cuda", "mps", "cpu")
        dtype: Data type ("bfloat16", "float16", "float32", or None for auto)

    Returns:
        Configured VisionLanguageTask instance
    """
    return VisionLanguageTask(device=device, dtype=dtype)
