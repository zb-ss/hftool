"""Image-to-image task handler.

Supports SDXL img2img for style transfer, image editing, and refinement.
"""

from typing import Any, Dict, Optional
from hftool.tasks.base import BaseTask
from hftool.io.output_handler import save_image


class ImageToImageTask(BaseTask):
    """Handler for image-to-image generation using diffusers.
    
    Supported models:
    - SDXL Refiner (stabilityai/stable-diffusion-xl-refiner-1.0)
    - SDXL Base img2img (stabilityai/stable-diffusion-xl-base-1.0)
    
    Input format:
    - JSON: {"image": "path/to/image.png", "prompt": "style description"}
    - Or just an image path (will use default prompt)
    """
    
    # Model-specific default configurations
    MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
        "refiner": {
            "num_inference_steps": 30,
            "strength": 0.3,  # Lower = more like original
            "guidance_scale": 7.5,
        },
        "stable-diffusion-xl": {
            "num_inference_steps": 30,
            "strength": 0.7,  # Higher = more transformation
            "guidance_scale": 7.5,
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
        
        # Default config
        return {
            "num_inference_steps": 30,
            "strength": 0.5,
            "guidance_scale": 7.5,
        }
    
    def _parse_input(self, input_data: Any) -> tuple:
        """Parse input data to extract image path and prompt.
        
        Args:
            input_data: Either a path string, or dict with "image" and "prompt"
        
        Returns:
            Tuple of (image_path, prompt)
        """
        import json
        
        if isinstance(input_data, dict):
            image_path = input_data.get("image")
            prompt = input_data.get("prompt", "")
            return image_path, prompt
        
        if isinstance(input_data, str):
            # Try to parse as JSON
            if input_data.strip().startswith("{"):
                try:
                    data = json.loads(input_data)
                    return data.get("image"), data.get("prompt", "")
                except json.JSONDecodeError:
                    pass
            
            # Assume it's just an image path
            return input_data, ""
        
        raise ValueError(f"Invalid input format: {type(input_data)}")
    
    def load_pipeline(self, model: str, **kwargs) -> Any:
        """Load a diffusers pipeline for image-to-image.
        
        Args:
            model: HuggingFace model name or local path
            **kwargs: Additional arguments for from_pretrained
        
        Returns:
            Loaded diffusers pipeline
        """
        import os
        import click
        from hftool.utils.deps import check_dependencies
        check_dependencies(["diffusers", "torch", "accelerate"], extra="with_t2i")
        
        import torch
        from hftool.core.device import detect_device, get_optimal_dtype, configure_rocm_env
        
        # Configure ROCm optimizations
        configure_rocm_env()
        
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
        
        # Check CPU offload settings
        cpu_offload_env = os.environ.get("HFTOOL_CPU_OFFLOAD", "").lower()
        use_cpu_offload = cpu_offload_env in ("1", "2", "true", "yes")
        
        load_kwargs = {"torch_dtype": dtype, **kwargs}
        
        # Load the appropriate pipeline based on model type
        click.echo("Loading img2img pipeline...")
        
        # Check if this is SDXL refiner
        if "refiner" in model_lower:
            from diffusers import StableDiffusionXLImg2ImgPipeline
            click.echo("Using SDXL Refiner pipeline...")
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(model, **load_kwargs)
        else:
            # Try AutoPipeline for other models
            from diffusers import AutoPipelineForImage2Image
            try:
                pipe = AutoPipelineForImage2Image.from_pretrained(
                    model,
                    **load_kwargs
                )
            except Exception as e:
                click.echo(f"AutoPipeline failed: {e}, trying SDXL pipeline...")
                from diffusers import StableDiffusionXLImg2ImgPipeline
                pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(model, **load_kwargs)
        
        # Enable memory optimizations
        try:
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()
        except Exception:
            pass
        
        # Move to device or enable offload
        if use_cpu_offload:
            if hasattr(pipe, "enable_model_cpu_offload"):
                click.echo("Enabling CPU offload...")
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(device)
        else:
            click.echo(f"Loading model on {device}...")
            pipe.to(device)
        
        return pipe
    
    def get_default_kwargs(self) -> Dict[str, Any]:
        """Get default inference kwargs based on loaded model."""
        if self._model_name:
            return self._get_model_config(self._model_name)
        return {}
    
    def run_inference(self, pipeline: Any, input_data: Any, **kwargs) -> Any:
        """Run image-to-image inference.
        
        Args:
            pipeline: Loaded diffusers pipeline
            input_data: Image path or dict with {"image": path, "prompt": text}
            **kwargs: Additional inference arguments
                - strength: How much to transform (0.0-1.0)
                - num_inference_steps: Denoising steps
                - guidance_scale: CFG scale
                - negative_prompt: What to avoid
                - seed: Random seed
        
        Returns:
            PIL.Image object
        """
        import torch
        import click
        from PIL import Image
        
        # Parse input
        image_path, prompt = self._parse_input(input_data)
        
        if not image_path:
            raise ValueError("No image provided. Use: -i '{\"image\": \"path.png\", \"prompt\": \"text\"}'")
        
        # Load the input image
        click.echo(f"Loading input image: {image_path}")
        init_image = Image.open(image_path).convert("RGB")
        
        # Handle seed
        seed = kwargs.pop("seed", None)
        if seed is not None and "generator" not in kwargs:
            device = next(pipeline.unet.parameters()).device if hasattr(pipeline, "unet") else "cpu"
            kwargs["generator"] = torch.Generator(device=str(device)).manual_seed(seed)
        
        # Get model defaults and merge
        defaults = self.get_default_kwargs()
        inference_kwargs = {**defaults, **kwargs}
        
        # Remove any None values
        inference_kwargs = {k: v for k, v in inference_kwargs.items() if v is not None}
        
        click.echo(f"Prompt: {prompt or '(none - refining only)'}")
        click.echo(f"Strength: {inference_kwargs.get('strength', 0.5)}")
        
        # Run inference with OOM handling
        try:
            result = pipeline(
                prompt=prompt,
                image=init_image,
                **inference_kwargs
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                click.echo("", err=True)
                click.echo(click.style("Out of GPU memory!", fg="red"), err=True)
                click.echo("Try: HFTOOL_CPU_OFFLOAD=1 hftool -t i2i ...", err=True)
                raise
            raise
        
        return result.images[0]
    
    def save_output(self, result: Any, output_path: str, **kwargs) -> str:
        """Save generated image to file.
        
        Args:
            result: PIL.Image from inference
            output_path: Path to save image
            **kwargs: Additional save arguments
        
        Returns:
            Path to saved file
        """
        return save_image(result, output_path, **kwargs)


def create_task(device: str = "auto", dtype: Optional[str] = None) -> ImageToImageTask:
    """Factory function to create an ImageToImageTask.
    
    Args:
        device: Device to run on ("auto", "cuda", "mps", "cpu")
        dtype: Data type ("bfloat16", "float16", "float32")
    
    Returns:
        Configured ImageToImageTask instance
    """
    return ImageToImageTask(device=device, dtype=dtype)
