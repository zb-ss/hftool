"""Text-to-video task handler.

Supports HunyuanVideo-1.5, CogVideoX, Wan2.2, and other diffusers-based video models.
"""

from typing import Any, Dict, List, Optional
from hftool.tasks.base import BaseTask, TextInputMixin
from hftool.io.output_handler import save_video


class TextToVideoTask(TextInputMixin, BaseTask):
    """Handler for text-to-video generation using diffusers.
    
    Supported models:
    - HunyuanVideo-1.5 (hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-*)
    - CogVideoX (THUDM/CogVideoX-5b, THUDM/CogVideoX-5b-I2V)
    - Wan2.1/2.2 (Wan-AI/Wan2.1-T2V-*)
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
        import click
        from hftool.utils.deps import check_dependencies, check_ffmpeg
        check_dependencies(["diffusers", "torch", "accelerate"], extra="with_video")
        check_ffmpeg()
        
        import torch
        from hftool.core.device import detect_device, get_optimal_dtype, get_device_info, configure_rocm_env
        
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
        
        # Determine loading strategy
        num_gpus = device_info.device_count if device == "cuda" else 0
        
        # Check multi-GPU settings
        multi_gpu_env = os.environ.get("HFTOOL_MULTI_GPU", "").lower()
        force_multi_gpu = multi_gpu_env in ("1", "true", "yes", "balanced")
        disable_multi_gpu = multi_gpu_env in ("0", "false", "no")
        use_multi_gpu = (num_gpus > 1 and not disable_multi_gpu) or force_multi_gpu
        
        # Check CPU offload settings
        cpu_offload_env = os.environ.get("HFTOOL_CPU_OFFLOAD", "").lower()
        disable_cpu_offload = cpu_offload_env in ("0", "false", "no")
        
        # Prepare load kwargs
        load_kwargs = {"torch_dtype": dtype, **kwargs}
        
        # Configure device_map for multi-GPU
        if use_multi_gpu and num_gpus > 1:
            click.echo(f"Multi-GPU mode: Distributing model across {num_gpus} GPUs...")
            load_kwargs["device_map"] = "balanced"
            max_memory = {}
            for i in range(num_gpus):
                try:
                    mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    max_memory[i] = f"{int(mem_gb - 2)}GB"
                except Exception:
                    pass
            if max_memory:
                load_kwargs["max_memory"] = max_memory
                click.echo(f"GPU memory allocation: {max_memory}")
        elif num_gpus > 1:
            click.echo(f"Multi-GPU detected ({num_gpus} GPUs) but disabled, using single GPU")
        
        # Load model-specific pipelines
        if "hunyuanvideo" in model_lower:
            pipe = self._load_hunyuanvideo(model, dtype, **load_kwargs)
        elif "cogvideo" in model_lower:
            pipe = self._load_cogvideox(model, dtype, **load_kwargs)
        elif "wan" in model_lower:
            pipe = self._load_wan(model, dtype, **load_kwargs)
        else:
            # Generic video pipeline
            from diffusers import DiffusionPipeline
            pipe = DiffusionPipeline.from_pretrained(model, **load_kwargs)
        
        # Enable VAE optimizations
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        
        # Check if pipeline has been placed on devices via device_map
        has_device_map = hasattr(pipe, "hf_device_map") and pipe.hf_device_map
        
        if has_device_map:
            click.echo(f"Model distributed across devices: {pipe.hf_device_map}")
        elif not disable_cpu_offload:
            # Enable CPU offload for memory efficiency (video models are large)
            if hasattr(pipe, "enable_model_cpu_offload"):
                click.echo("Enabling model CPU offload...")
                pipe.enable_model_cpu_offload()
        else:
            click.echo(f"Loading model on {device}...")
            pipe.to(device)
        
        return pipe
    
    def _load_hunyuanvideo(self, model: str, dtype, **kwargs) -> Any:
        """Load HunyuanVideo-1.5 pipeline."""
        # dtype is passed for backwards compatibility but torch_dtype should be in kwargs
        if "torch_dtype" not in kwargs:
            kwargs["torch_dtype"] = dtype
        try:
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
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(model, **kwargs)
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
                - guidance_scale: CFG scale
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
        
        # Run inference
        result = pipeline(prompt=prompt, **inference_kwargs)
        
        # Extract frames (different pipelines return different formats)
        if hasattr(result, "frames"):
            frames = result.frames
            # Some pipelines return nested list: [[frame1, frame2, ...]]
            if frames and isinstance(frames[0], list):
                frames = frames[0]
        elif hasattr(result, "images"):
            frames = result.images
        else:
            frames = result
        
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
            input_data: Tuple of (image_path, prompt) or dict with 'image' and 'prompt'
        
        Returns:
            Dict with 'image' and 'prompt' keys
        """
        from hftool.io.input_loader import load_image
        
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
