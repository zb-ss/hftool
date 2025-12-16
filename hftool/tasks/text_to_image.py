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
        
        Environment Variables:
            HFTOOL_MULTI_GPU: Control multi-GPU behavior
                - "1", "true", "yes", "balanced": Use device_map="balanced" to spread across GPUs
                - "0", "false", "no": Disable multi-GPU, use single GPU
                - unset: Auto-detect and use multi-GPU if available
            HFTOOL_CPU_OFFLOAD: Control CPU offload (only when multi-GPU disabled)
                - "0": Disabled (full GPU)
                - "1": Model CPU offload
                - "2": Sequential CPU offload (most memory efficient)
        """
        import os
        import click
        from hftool.utils.deps import check_dependencies
        check_dependencies(["diffusers", "torch", "accelerate"], extra="with_video")
        
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
        
        # Determine loading strategy
        num_gpus = device_info.device_count if device == "cuda" else 0
        
        # Check multi-GPU settings
        # HFTOOL_MULTI_GPU: "1"/"true"/"balanced" = use device_map, "0"/"false" = single GPU
        multi_gpu_env = os.environ.get("HFTOOL_MULTI_GPU", "").lower()
        force_multi_gpu = multi_gpu_env in ("1", "true", "yes", "balanced")
        disable_multi_gpu = multi_gpu_env in ("0", "false", "no")
        
        # Auto-enable multi-GPU if multiple GPUs available and not explicitly disabled
        use_multi_gpu = (num_gpus > 1 and not disable_multi_gpu) or force_multi_gpu
        
        # Check CPU offload settings (used when multi-GPU is not active)
        # HFTOOL_CPU_OFFLOAD: 0=disabled (full GPU), 1=model offload, 2=sequential offload
        cpu_offload_env = os.environ.get("HFTOOL_CPU_OFFLOAD", "").lower()
        force_cpu_offload = cpu_offload_env in ("1", "2", "true", "yes")
        disable_cpu_offload = cpu_offload_env in ("0", "false", "no")
        use_sequential = cpu_offload_env == "2"
        
        # Try to load with appropriate pipeline
        model_lower = model.lower()
        is_zimage = "z-image" in model_lower or "zimage" in model_lower
        is_flux = "flux" in model_lower
        
        pipe = None
        load_kwargs = {"torch_dtype": dtype, **kwargs}
        
        # Configure device_map for multi-GPU
        if use_multi_gpu and num_gpus > 1:
            click.echo(f"Multi-GPU mode: Distributing model across {num_gpus} GPUs...")
            # Use "balanced" to evenly distribute model components across GPUs
            load_kwargs["device_map"] = "balanced"
            # Set max_memory per GPU to help with distribution
            # Get memory for each GPU
            max_memory = {}
            for i in range(num_gpus):
                try:
                    mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    # Reserve ~2GB for overhead
                    max_memory[i] = f"{int(mem_gb - 2)}GB"
                except Exception:
                    pass
            if max_memory:
                load_kwargs["max_memory"] = max_memory
                click.echo(f"GPU memory allocation: {max_memory}")
        elif num_gpus > 1:
            click.echo(f"Multi-GPU detected ({num_gpus} GPUs) but disabled, using single GPU")
        
        # Try ZImagePipeline for Z-Image models
        if is_zimage:
            try:
                from diffusers import ZImagePipeline
                pipe = ZImagePipeline.from_pretrained(model, **load_kwargs)
            except ImportError:
                click.echo(
                    "Warning: ZImagePipeline not available. "
                    "Upgrade diffusers: pip install --upgrade diffusers>=0.33.0",
                    err=True
                )
            except Exception as e:
                click.echo(f"Warning: Failed to load ZImagePipeline: {e}", err=True)
        
        # Try FluxPipeline for FLUX models
        if pipe is None and is_flux:
            try:
                from diffusers import FluxPipeline
                pipe = FluxPipeline.from_pretrained(model, **load_kwargs)
            except ImportError:
                pass
            except Exception:
                pass
        
        # Try generic DiffusionPipeline with auto-detection
        if pipe is None:
            try:
                from diffusers import DiffusionPipeline
                pipe = DiffusionPipeline.from_pretrained(
                    model,
                    trust_remote_code=True,
                    **load_kwargs
                )
            except Exception:
                pass
        
        # Final fallback to AutoPipelineForText2Image
        if pipe is None:
            from diffusers import AutoPipelineForText2Image
            pipe = AutoPipelineForText2Image.from_pretrained(model, **load_kwargs)
        
        # Enable memory optimizations
        try:
            # Enable VAE slicing for large images
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            
            # Enable VAE tiling for very large images
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()
        except Exception as e:
            click.echo(f"Note: Could not enable all memory optimizations: {e}", err=True)
        
        # If multi-GPU with device_map was used, the model is already distributed
        # Check if pipeline has been placed on devices via device_map
        has_device_map = hasattr(pipe, "hf_device_map") and pipe.hf_device_map
        
        if has_device_map:
            click.echo(f"Model distributed across devices: {pipe.hf_device_map}")
        else:
            # No device_map, apply single-GPU or CPU offload strategy
            if disable_cpu_offload:
                # User explicitly wants full GPU mode
                click.echo(f"Loading model fully on {device}...")
                pipe.to(device)
            elif use_sequential or (force_cpu_offload and cpu_offload_env == "2"):
                # Sequential offload - most memory efficient, slower
                if hasattr(pipe, "enable_sequential_cpu_offload"):
                    click.echo("Enabling sequential CPU offload (memory-efficient mode)...")
                    pipe.enable_sequential_cpu_offload()
                else:
                    pipe.to(device)
            elif force_cpu_offload:
                # Model offload - faster than sequential but needs more VRAM
                if hasattr(pipe, "enable_model_cpu_offload"):
                    click.echo("Enabling model CPU offload...")
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to(device)
            else:
                # Auto mode - try full GPU first, fall back to CPU offload if needed
                click.echo(f"Loading model on {device}...")
                try:
                    pipe.to(device)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        click.echo("GPU memory insufficient, enabling CPU offload...")
                        if hasattr(pipe, "enable_model_cpu_offload"):
                            pipe.enable_model_cpu_offload()
                        elif hasattr(pipe, "enable_sequential_cpu_offload"):
                            pipe.enable_sequential_cpu_offload()
                        else:
                            raise
                    else:
                        raise
        
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
        import click
        
        # Handle seed -> generator conversion
        seed = kwargs.pop("seed", None)
        if seed is not None and "generator" not in kwargs:
            device = next(pipeline.unet.parameters()).device if hasattr(pipeline, "unet") else "cpu"
            kwargs["generator"] = torch.Generator(device=str(device)).manual_seed(seed)
        
        # Get model-specific defaults and merge with kwargs
        defaults = self.get_default_kwargs()
        inference_kwargs = {**defaults, **kwargs}
        
        # Run inference with OOM handling
        try:
            result = pipeline(prompt=prompt, **inference_kwargs)
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "hip out of memory" in error_msg:
                height = inference_kwargs.get("height", 1024)
                width = inference_kwargs.get("width", 1024)
                click.echo("", err=True)
                click.echo(click.style("Out of GPU memory!", fg="red"), err=True)
                click.echo("", err=True)
                click.echo("Try one of these solutions:", err=True)
                click.echo(f"  1. Use a smaller resolution (current: {width}x{height}):", err=True)
                click.echo(f"     hftool -t t2i -i \"...\" -o out.png -- --height 768 --width 768", err=True)
                click.echo("", err=True)
                click.echo("  2. Enable CPU offload (slower but uses less VRAM):", err=True)
                click.echo("     HFTOOL_CPU_OFFLOAD=1 hftool -t t2i -i \"...\" -o out.png", err=True)
                click.echo("", err=True)
                click.echo("  3. Disable multi-GPU and use single GPU:", err=True)
                click.echo("     HFTOOL_MULTI_GPU=0 hftool -t t2i -i \"...\" -o out.png", err=True)
                click.echo("", err=True)
                raise
            else:
                raise
        
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
