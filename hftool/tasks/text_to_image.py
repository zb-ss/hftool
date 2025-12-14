"""Text-to-image task handler.

Supports Z-Image, Stable Diffusion XL, FLUX, and other diffusers-based models.
"""

from typing import Any, Dict, Optional
from hftool.tasks.base import BaseTask, TextInputMixin
from hftool.io.output_handler import save_image


class TextToImageTask(TextInputMixin, BaseTask):
    """Handler for text-to-image generation using diffusers.
    
    Supported models:
    - Z-Image-Turbo (Tongyi-MAI/Z-Image-Turbo)
    - Stable Diffusion XL (stabilityai/stable-diffusion-xl-base-1.0)
    - FLUX (black-forest-labs/FLUX.1-schnell)
    - And other diffusers-compatible models
    """
    
    # Model-specific default configurations
    MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
        "Z-Image-Turbo": {
            "num_inference_steps": 9,
            "guidance_scale": 0.0,
            "height": 1024,
            "width": 1024,
        },
        "Z-Image": {
            "num_inference_steps": 9,
            "guidance_scale": 0.0,
            "height": 1024,
            "width": 1024,
        },
        "FLUX": {
            "num_inference_steps": 4,
            "guidance_scale": 0.0,
            "height": 1024,
            "width": 1024,
        },
        "stable-diffusion-xl": {
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "height": 1024,
            "width": 1024,
        },
        "stable-diffusion": {
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "height": 512,
            "width": 512,
        },
    }
    
    def __init__(self, device: str = "auto", dtype: Optional[str] = None):
        super().__init__(device, dtype)
        self._model_name: Optional[str] = None
    
    def _get_model_config(self, model: str) -> Dict[str, Any]:
        """Get default config for a specific model."""
        model_lower = model.lower()
        
        for key, config in self.MODEL_CONFIGS.items():
            if key.lower() in model_lower:
                return config.copy()
        
        # Default config for unknown models
        return {
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "height": 512,
            "width": 512,
        }
    
    def load_pipeline(self, model: str, **kwargs) -> Any:
        """Load a diffusers pipeline for text-to-image.
        
        Args:
            model: HuggingFace model name or local path
            **kwargs: Additional arguments for from_pretrained
        
        Returns:
            Loaded diffusers pipeline
        """
        from hftool.utils.deps import check_dependencies
        check_dependencies(["diffusers", "torch", "accelerate"], extra="with_video")
        
        from hftool.core.device import detect_device, get_optimal_dtype
        
        # Determine device and dtype
        device = self.device if self.device != "auto" else detect_device()
        
        if self.dtype:
            import torch
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(self.dtype, torch.bfloat16)
        else:
            dtype = get_optimal_dtype(device)
        
        self._model_name = model
        
        # Try to load with appropriate pipeline
        try:
            # First try ZImagePipeline for Z-Image models
            if "z-image" in model.lower() or "zimage" in model.lower():
                from diffusers import ZImagePipeline
                pipe = ZImagePipeline.from_pretrained(
                    model,
                    torch_dtype=dtype,
                    **kwargs
                )
            else:
                # Generic DiffusionPipeline for auto-detection
                from diffusers import DiffusionPipeline
                pipe = DiffusionPipeline.from_pretrained(
                    model,
                    torch_dtype=dtype,
                    **kwargs
                )
        except Exception as e:
            # Fallback to AutoPipelineForText2Image
            from diffusers import AutoPipelineForText2Image
            pipe = AutoPipelineForText2Image.from_pretrained(
                model,
                torch_dtype=dtype,
                **kwargs
            )
        
        # Move to device
        pipe.to(device)
        
        # Enable memory optimizations for ROCm
        try:
            from hftool.core.device import is_rocm
            if is_rocm():
                # Enable attention slicing for better memory efficiency
                if hasattr(pipe, "enable_attention_slicing"):
                    pipe.enable_attention_slicing()
        except Exception:
            pass
        
        return pipe
    
    def get_default_kwargs(self) -> Dict[str, Any]:
        """Get default inference kwargs based on loaded model."""
        if self._model_name:
            return self._get_model_config(self._model_name)
        return {}
    
    def run_inference(self, pipeline: Any, prompt: str, **kwargs) -> Any:
        """Run text-to-image inference.
        
        Args:
            pipeline: Loaded diffusers pipeline
            prompt: Text prompt for image generation
            **kwargs: Additional inference arguments
                - num_inference_steps: Number of denoising steps
                - guidance_scale: CFG scale
                - height: Image height
                - width: Image width
                - negative_prompt: Negative prompt
                - seed: Random seed
                - generator: torch.Generator (created from seed if not provided)
        
        Returns:
            PIL.Image object
        """
        import torch
        
        # Handle seed -> generator conversion
        seed = kwargs.pop("seed", None)
        if seed is not None and "generator" not in kwargs:
            device = next(pipeline.unet.parameters()).device if hasattr(pipeline, "unet") else "cpu"
            kwargs["generator"] = torch.Generator(device=str(device)).manual_seed(seed)
        
        # Get model-specific defaults and merge with kwargs
        defaults = self.get_default_kwargs()
        inference_kwargs = {**defaults, **kwargs}
        
        # Run inference
        result = pipeline(prompt=prompt, **inference_kwargs)
        
        # Return the first image
        return result.images[0]
    
    def save_output(self, result: Any, output_path: str, **kwargs) -> str:
        """Save generated image to file.
        
        Args:
            result: PIL.Image from inference
            output_path: Path to save image
            **kwargs: Additional save arguments (quality, etc.)
        
        Returns:
            Path to saved file
        """
        return save_image(result, output_path, **kwargs)


def create_task(device: str = "auto", dtype: Optional[str] = None) -> TextToImageTask:
    """Factory function to create a TextToImageTask.
    
    Args:
        device: Device to run on ("auto", "cuda", "mps", "cpu")
        dtype: Data type ("bfloat16", "float16", "float32")
    
    Returns:
        Configured TextToImageTask instance
    """
    return TextToImageTask(device=device, dtype=dtype)
