"""Text-to-video task handler.

Supports HunyuanVideo-1.5, CogVideoX, Wan2.2, LTX-2, and other diffusers-based video models.
"""

from typing import Any, Dict, List, Optional
from hftool.tasks.base import BaseTask, TextInputMixin
from hftool.io.output_handler import save_video
from hftool.utils.progress import console


class TextToVideoTask(TextInputMixin, BaseTask):
    """Handler for text-to-video generation using diffusers.

    Supported models:
    - HunyuanVideo-1.5 (hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-*)
    - CogVideoX (THUDM/CogVideoX-5b, THUDM/CogVideoX-5b-I2V)
    - Wan2.1/2.2 (Wan-AI/Wan2.1-T2V-*)
    - LTX-2 (Lightricks/LTX-2) - 19B audio-video foundation model
    - And other diffusers-compatible video models
    """
    
    # Model-specific default configurations
    MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
        "HunyuanVideo-1.5": {
            "num_frames": 61,  # ~2.5 seconds at 24fps
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
            "height": 480,
            "width": 848,  # 16:9 aspect ratio
        },
        "HunyuanVideo": {
            "num_frames": 61,
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
        },
        "CogVideoX": {
            "num_frames": 49,
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
        },
        "Wan2": {
            "num_frames": 81,
            "num_inference_steps": 50,
            "guidance_scale": 5.0,
        },
        "LTX-2": {
            "num_frames": 97,  # ~4 seconds at 24fps
            "num_inference_steps": 50,
            "guidance_scale": 3.0,
            "height": 512,
            "width": 768,  # Must be divisible by 32
        },
    }
    
    def __init__(
        self,
        device: str = "auto",
        dtype: Optional[str] = None,
        mode: str = "t2v"  # "t2v" for text-to-video, "i2v" for image-to-video
    ):
        super().__init__(device, dtype)
        self._model_name: Optional[str] = None
        self.mode = mode
    
    def _get_model_config(self, model: str) -> Dict[str, Any]:
        """Get default config for a specific model."""
        model_lower = model.lower()
        
        for key, config in self.MODEL_CONFIGS.items():
            if key.lower() in model_lower:
                return config.copy()
        
        # Default config for unknown models
        return {
            "num_frames": 49,
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
        }
    
    def load_pipeline(self, model: str, **kwargs) -> Any:
        """Load a diffusers pipeline for text-to-video.
        
        Args:
            model: HuggingFace model name or local path
            **kwargs: Additional arguments for from_pretrained
        
        Returns:
            Loaded diffusers pipeline
        
        Environment Variables:
            HFTOOL_MULTI_GPU: Control multi-GPU behavior
                - "1", "true", "yes", "balanced": Use device_map="balanced" to spread across GPUs
                - "0", "false", "no": Disable multi-GPU, use single GPU
                - unset: Auto-detect and use multi-GPU if available
            HFTOOL_CPU_OFFLOAD: Control CPU offload (only when multi-GPU disabled)
                - "0": Disabled (full GPU)
                - "1": Model CPU offload (default)
                - "2": Sequential CPU offload (most memory efficient)
        """
        import os
        from hftool.utils.deps import check_dependencies, check_ffmpeg
        check_dependencies(["diffusers", "torch", "accelerate"], extra="with_video")
        check_ffmpeg()

        import torch
        from hftool.core.device import (
            detect_device, get_optimal_dtype, get_device_info,
            configure_rocm_env, get_multi_gpu_kwargs
        )

        # Configure ROCm optimizations early (before any GPU operations)
        configure_rocm_env()

        # Get device info
        device_info = get_device_info()
        device = self.device if self.device != "auto" else detect_device()

        if self.dtype:
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(self.dtype, torch.bfloat16)
        else:
            dtype = get_optimal_dtype(device)

        self._model_name = model
        model_lower = model.lower()

        # Prepare load kwargs
        load_kwargs = {"torch_dtype": dtype, **kwargs}

        # Get multi-GPU configuration (centralized logic)
        gpu_config = get_multi_gpu_kwargs(reserve_per_gpu_gb=6.0)
        if gpu_config["message"]:
            console.info(gpu_config["message"])

        if gpu_config["use_multi_gpu"]:
            load_kwargs["device_map"] = gpu_config["device_map"]
            load_kwargs["max_memory"] = gpu_config["max_memory"]

        # Check CPU offload settings (used when multi-GPU not active)
        cpu_offload_env = os.environ.get("HFTOOL_CPU_OFFLOAD", "").lower()
        disable_cpu_offload = cpu_offload_env in ("0", "false", "no")
        use_sequential_offload = cpu_offload_env == "2"
        
        # Load model-specific pipelines
        if "hunyuanvideo" in model_lower:
            pipe = self._load_hunyuanvideo(model, dtype, **load_kwargs)
        elif "cogvideo" in model_lower:
            pipe = self._load_cogvideox(model, dtype, **load_kwargs)
        elif "wan" in model_lower:
            pipe = self._load_wan(model, dtype, **load_kwargs)
        elif "ltx" in model_lower:
            pipe = self._load_ltx2(model, dtype, **load_kwargs)
        else:
            # Generic video pipeline
            from diffusers import DiffusionPipeline
            pipe = DiffusionPipeline.from_pretrained(model, **load_kwargs)
        
        # Enable VAE optimizations for memory efficiency
        # Tiling processes the image/video in smaller chunks
        if hasattr(pipe, "vae"):
            if hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
                console.log("VAE tiling enabled")
            # Slicing processes batch elements one at a time
            if hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()
                console.log("VAE slicing enabled")
        # Pipeline-level VAE slicing (for video frame-by-frame processing)
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()

        # Check if pipeline has been placed on devices via device_map
        has_device_map = hasattr(pipe, "hf_device_map") and pipe.hf_device_map

        if has_device_map:
            # Check if any component ended up on CPU - this breaks inference
            cpu_components = [k for k, v in pipe.hf_device_map.items() if v == "cpu"]
            if cpu_components:
                console.warn(f"Components on CPU will cause errors: {cpu_components}")
                console.log("Falling back to sequential CPU offload...")
                # Reset the device map and use sequential offload instead
                pipe = pipe.to("cpu")  # Reset to CPU first
                if hasattr(pipe, "enable_sequential_cpu_offload"):
                    pipe.enable_sequential_cpu_offload()
                else:
                    console.log("Using model CPU offload as fallback...")
                    pipe.enable_model_cpu_offload()
            else:
                console.log(f"Model distributed across devices: {pipe.hf_device_map}")
        elif disable_cpu_offload:
            # User explicitly disabled CPU offload
            console.log(f"Loading model fully on {device}...")
            pipe.to(device)
        elif use_sequential_offload:
            # Sequential offload - most memory efficient, slower
            if hasattr(pipe, "enable_sequential_cpu_offload"):
                console.log("Enabling sequential CPU offload (most memory-efficient)...")
                pipe.enable_sequential_cpu_offload()
            elif hasattr(pipe, "enable_model_cpu_offload"):
                console.log("Enabling model CPU offload...")
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(device)
        else:
            # Default for video: use sequential offload for large models, model offload otherwise
            # Sequential is slower but handles models larger than VRAM
            if hasattr(pipe, "enable_sequential_cpu_offload"):
                console.log("Enabling sequential CPU offload (required for large video models)...")
                pipe.enable_sequential_cpu_offload()
            elif hasattr(pipe, "enable_model_cpu_offload"):
                console.log("Enabling model CPU offload...")
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(device)
        
        return pipe
    
    def _load_hunyuanvideo(self, model: str, dtype, **kwargs) -> Any:
        """Load HunyuanVideo-1.5 pipeline.
        
        For multi-GPU: Loads transformer with device_map across GPUs,
        then uses CPU offload for VAE to prevent OOM during decode.
        """
        import os
        import torch
        from hftool.core.device import get_device_info

        device_info = get_device_info()
        num_gpus = device_info.device_count
        
        # Check if multi-GPU is requested
        multi_gpu_env = os.environ.get("HFTOOL_MULTI_GPU", "").lower()
        use_multi_gpu = num_gpus > 1 and multi_gpu_env not in ("0", "false", "no")
        
        # Remove device_map from kwargs if present - we handle it specially
        kwargs.pop("device_map", None)
        kwargs.pop("max_memory", None)
        
        if "torch_dtype" not in kwargs:
            kwargs["torch_dtype"] = dtype
        
        try:
            if use_multi_gpu:
                # Multi-GPU: Load transformer distributed across GPUs, VAE with CPU offload
                console.log(f"Loading HunyuanVideo with multi-GPU transformer ({num_gpus} GPUs)...")

                from diffusers import HunyuanVideo15Pipeline, AutoModel

                # Calculate max memory per GPU for transformer (leave room for other ops)
                max_memory = {}
                for i in range(num_gpus):
                    try:
                        mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                        max_memory[i] = f"{int(mem_gb - 4)}GB"  # Reserve 4GB per GPU
                    except Exception:
                        pass

                # Load transformer distributed across GPUs
                transformer = AutoModel.from_pretrained(
                    model,
                    subfolder="transformer",
                    torch_dtype=dtype,
                    device_map="auto",
                    max_memory=max_memory if max_memory else None,
                )
                console.log(f"Transformer distributed: {getattr(transformer, 'hf_device_map', 'single device')}")
                
                # Load pipeline without transformer (we'll add it)
                pipe = HunyuanVideo15Pipeline.from_pretrained(
                    model,
                    transformer=transformer,
                    **kwargs
                )
            else:
                from diffusers import HunyuanVideo15Pipeline
                pipe = HunyuanVideo15Pipeline.from_pretrained(model, **kwargs)
                
        except ImportError:
            # Fallback if HunyuanVideo15Pipeline not available
            from diffusers import DiffusionPipeline
            pipe = DiffusionPipeline.from_pretrained(model, **kwargs)
        
        return pipe
    
    def _load_cogvideox(self, model: str, dtype, **kwargs) -> Any:
        """Load CogVideoX pipeline."""
        if "torch_dtype" not in kwargs:
            kwargs["torch_dtype"] = dtype
        if self.mode == "i2v" or "i2v" in model.lower():
            from diffusers import CogVideoXImageToVideoPipeline
            pipe = CogVideoXImageToVideoPipeline.from_pretrained(model, **kwargs)
        else:
            from diffusers import CogVideoXPipeline
            pipe = CogVideoXPipeline.from_pretrained(model, **kwargs)
        return pipe
    
    def _load_wan(self, model: str, dtype, **kwargs) -> Any:
        """Load Wan2.x pipeline."""
        if "torch_dtype" not in kwargs:
            kwargs["torch_dtype"] = dtype

        if self.mode == "i2v" or "i2v" in model.lower():
            from diffusers import WanImageToVideoPipeline
            pipe = WanImageToVideoPipeline.from_pretrained(model, **kwargs)
        else:
            from diffusers import WanPipeline
            pipe = WanPipeline.from_pretrained(model, **kwargs)
        return pipe

    def _load_ltx2(self, model: str, dtype, **kwargs) -> Any:
        """Load LTX-2 pipeline.

        LTX-2 is a video generation model from Lightricks.
        Requires diffusers main branch (0.37.0+) with LTX2Pipeline support.

        See: https://huggingface.co/Lightricks/LTX-2
        """
        from hftool.utils.errors import HFToolError

        if "torch_dtype" not in kwargs:
            kwargs["torch_dtype"] = dtype

        try:
            if self.mode == "i2v" or "i2v" in model.lower():
                from diffusers import LTX2ImageToVideoPipeline as LTXPipeline
                pipeline_name = "LTX2ImageToVideoPipeline"
            else:
                from diffusers import LTX2Pipeline as LTXPipeline
                pipeline_name = "LTX2Pipeline"
        except (ImportError, AttributeError):
            # LTX2Pipeline is only available in diffusers 0.37.0+ (currently unreleased)
            import os
            in_docker = os.environ.get("HFTOOL_IN_DOCKER") == "1"

            if in_docker:
                # Can't pip install in Docker with --user flag (permission denied)
                raise HFToolError(
                    "LTX-2 requires diffusers main branch (0.37.0+) which is not in the Docker image.",
                    suggestion=(
                        "Rebuild the Docker image with updated diffusers:\n"
                        "  hftool docker build --no-cache\n\n"
                        "Or run natively (outside Docker) to auto-install dependencies."
                    )
                )

            # Native mode: try to auto-install
            console.info("LTX2Pipeline not available. Installing diffusers from main branch...")
            from hftool.core.download import install_pip_dependencies
            success = install_pip_dependencies(
                ["git+https://github.com/huggingface/diffusers"],
                force=True
            )
            if success:
                console.info("Diffusers updated. Please restart hftool to use LTX-2.")
            raise HFToolError(
                "LTX-2 requires diffusers main branch (0.37.0+) which was just installed.",
                suggestion="Please restart hftool to use the updated diffusers library."
            )

        console.log(f"Loading {pipeline_name}...")
        pipe = LTXPipeline.from_pretrained(model, **kwargs)
        return pipe
    
    def get_default_kwargs(self) -> Dict[str, Any]:
        """Get default inference kwargs based on loaded model."""
        if self._model_name:
            return self._get_model_config(self._model_name)
        return {}
    
    def run_inference(self, pipeline: Any, prompt: str, **kwargs) -> List[Any]:
        """Run text-to-video inference.
        
        Args:
            pipeline: Loaded diffusers pipeline
            prompt: Text prompt for video generation
            **kwargs: Additional inference arguments
                - num_frames: Number of frames to generate
                - num_inference_steps: Number of denoising steps
                - guidance_scale: CFG scale (handled via guider for HunyuanVideo1.5)
                - height: Video height
                - width: Video width
                - negative_prompt: Negative prompt
                - seed: Random seed
                - image: Input image for I2V mode
        
        Returns:
            List of PIL.Image frames
        """
        import torch
        
        # Handle seed -> generator conversion
        seed = kwargs.pop("seed", None)
        if seed is not None and "generator" not in kwargs:
            kwargs["generator"] = torch.Generator("cpu").manual_seed(seed)
        
        # Get model-specific defaults and merge with kwargs
        defaults = self.get_default_kwargs()
        inference_kwargs = {**defaults, **kwargs}
        
        # Handle guidance_scale for pipelines that use guider system (HunyuanVideo1.5)
        # These pipelines don't accept guidance_scale at runtime - it must be set on the guider
        guidance_scale = inference_kwargs.pop("guidance_scale", None)
        if guidance_scale is not None and hasattr(pipeline, "guider"):
            # Update the guider's guidance_scale
            try:
                pipeline.guider = pipeline.guider.new(guidance_scale=guidance_scale)
            except Exception:
                pass  # If guider doesn't support this, ignore
        
        # Run inference
        result = pipeline(prompt=prompt, **inference_kwargs)
        
        # Extract frames (different pipelines return different formats)
        if hasattr(result, "frames"):
            frames = result.frames
        elif hasattr(result, "images"):
            frames = result.images
        else:
            frames = result

        # Handle torch tensors - convert to numpy
        if hasattr(frames, "cpu"):
            frames = frames.cpu().numpy()

        # Handle numpy arrays with batch dimension
        import numpy as np
        if isinstance(frames, np.ndarray):
            # Diffusers returns shape (batch, num_frames, H, W, C) or (batch, num_frames, C, H, W)
            if frames.ndim == 5:
                frames = frames[0]  # Take first batch -> (num_frames, H, W, C)
            if frames.ndim == 4:
                # Convert to list of individual frames
                frames = [frames[i] for i in range(frames.shape[0])]
        # Handle nested list: [[frame1, frame2, ...]]
        elif isinstance(frames, list) and len(frames) > 0 and isinstance(frames[0], list):
            frames = frames[0]

        return frames
    
    def save_output(self, result: List[Any], output_path: str, **kwargs) -> str:
        """Save generated video to file.
        
        Args:
            result: List of PIL.Image frames
            output_path: Path to save video
            **kwargs: Additional save arguments
                - fps: Frames per second (default: 24)
        
        Returns:
            Path to saved file
        """
        fps = kwargs.pop("fps", 24)
        return save_video(result, output_path, fps=fps, **kwargs)


class ImageToVideoTask(TextToVideoTask):
    """Handler for image-to-video generation.
    
    Inherits from TextToVideoTask but handles image input.
    """
    
    def __init__(self, device: str = "auto", dtype: Optional[str] = None):
        super().__init__(device, dtype, mode="i2v")
    
    def validate_input(self, input_data: Any) -> Any:
        """Validate image input for I2V.

        Args:
            input_data: Tuple of (image_path, prompt), dict with 'image' and 'prompt',
                       or JSON string containing the dict

        Returns:
            Dict with 'image' and 'prompt' keys
        """
        import json
        from hftool.io.input_loader import load_image

        # Handle JSON string input (from interactive mode)
        if isinstance(input_data, str):
            try:
                input_data = json.loads(input_data)
            except json.JSONDecodeError:
                raise ValueError(
                    "I2V input must be a dict with 'image' and 'prompt' keys, "
                    "a tuple of (image_path, prompt), or a JSON string. "
                    f"Got string: {input_data[:50]}..."
                )

        if isinstance(input_data, dict):
            image = input_data.get("image")
            prompt = input_data.get("prompt", "")
        elif isinstance(input_data, (list, tuple)) and len(input_data) == 2:
            image, prompt = input_data
        else:
            raise ValueError(
                "I2V input must be a dict with 'image' and 'prompt' keys, "
                "or a tuple of (image_path, prompt)"
            )
        
        # Load image if it's a path
        if isinstance(image, str):
            image = load_image(image)
        
        return {"image": image, "prompt": prompt}
    
    def run_inference(self, pipeline: Any, input_data: Any, **kwargs) -> List[Any]:
        """Run image-to-video inference."""
        validated = self.validate_input(input_data)
        kwargs["image"] = validated["image"]
        return super().run_inference(pipeline, validated["prompt"], **kwargs)


def create_task(
    device: str = "auto",
    dtype: Optional[str] = None,
    mode: str = "t2v"
) -> TextToVideoTask:
    """Factory function to create a video generation task.
    
    Args:
        device: Device to run on ("auto", "cuda", "mps", "cpu")
        dtype: Data type ("bfloat16", "float16", "float32")
        mode: "t2v" for text-to-video, "i2v" for image-to-video
    
    Returns:
        Configured task instance
    """
    if mode == "i2v":
        return ImageToVideoTask(device=device, dtype=dtype)
    return TextToVideoTask(device=device, dtype=dtype, mode=mode)
