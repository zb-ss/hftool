"""Image-to-image task handler.

Supports SDXL img2img for style transfer, image editing, and refinement.
Also supports Qwen Image Edit for advanced image editing with character consistency.
"""

from typing import Any, Dict, List, Optional, Union
from hftool.tasks.base import BaseTask
from hftool.io.output_handler import save_image


class ImageToImageTask(BaseTask):
    """Handler for image-to-image generation using diffusers.

    Supported models:
    - Qwen Image Edit (Qwen/Qwen-Image-Edit-2511) - Advanced editing with multi-image support
    - FLUX.2 Klein 9B (black-forest-labs/FLUX.2-klein-9B) - Fast multi-reference editing (non-commercial)
    - SDXL Refiner (stabilityai/stable-diffusion-xl-refiner-1.0)
    - SDXL Base img2img (stabilityai/stable-diffusion-xl-base-1.0)

    Input format:
    - JSON: {"image": "path/to/image.png", "prompt": "edit description"}
    - JSON with multiple images: {"image": ["path1.png", "path2.png"], "prompt": "combine them"}
    - Or just an image path (will use default prompt)

    For FLUX.2 Klein, reference images in prompt using "image 1", "image 2", etc.
    """
    
    # Model-specific default configurations
    MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
        "qwen-image-edit": {
            "num_inference_steps": 40,
            "guidance_scale": 1.0,
            "true_cfg_scale": 4.0,
            "negative_prompt": " ",
        },
        "flux2-klein": {
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
            "height": 1024,
            "width": 1024,
        },
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
        """Parse input data to extract image path(s) and prompt.
        
        Args:
            input_data: Either a path string, or dict with "image" and "prompt"
                       "image" can be a single path or list of paths
        
        Returns:
            Tuple of (image_paths, prompt) where image_paths is a string or list
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
        check_dependencies(["diffusers", "torch", "accelerate"], extra="with_t2i")
        
        import torch
        from hftool.core.device import detect_device, get_optimal_dtype, get_device_info, configure_rocm_env
        
        # Configure ROCm optimizations
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
        
        # Auto-enable multi-GPU if multiple GPUs available and not explicitly disabled
        use_multi_gpu = (num_gpus > 1 and not disable_multi_gpu) or force_multi_gpu
        
        # Check CPU offload settings
        cpu_offload_env = os.environ.get("HFTOOL_CPU_OFFLOAD", "").lower()
        use_cpu_offload = cpu_offload_env in ("1", "2", "true", "yes")
        use_sequential = cpu_offload_env == "2"
        
        load_kwargs = {"torch_dtype": dtype, **kwargs}
        
        # Configure device_map for multi-GPU
        if use_multi_gpu and num_gpus > 1:
            click.echo(f"Multi-GPU mode: Distributing model across {num_gpus} GPUs...")
            load_kwargs["device_map"] = "balanced"
            # Set max_memory per GPU
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
        
        # Load the appropriate pipeline based on model type
        click.echo("Loading img2img pipeline...")
        
        # Check if this is FLUX.2 Klein
        is_flux2_klein = "flux2-klein" in model_lower or "flux.2-klein" in model_lower

        # Check if this is Qwen Image Edit
        is_qwen = "qwen-image-edit" in model_lower or "qwen/qwen-image-edit" in model_lower

        if is_flux2_klein:
            click.echo("Using FLUX.2 Klein pipeline...")

            # Check diffusers version
            import diffusers
            from packaging import version
            diffusers_version = version.parse(diffusers.__version__)
            if diffusers_version < version.parse("0.36.0"):
                click.echo(
                    f"Warning: diffusers {diffusers.__version__} detected. "
                    f"FLUX.2 Klein requires diffusers >= 0.36.0",
                    err=True
                )
                click.echo("Upgrade with: pip install --upgrade diffusers>=0.36.0", err=True)

            # FLUX.2 Klein works best with bfloat16
            flux_kwargs = {"torch_dtype": torch.bfloat16}

            # For multi-GPU, use device_map with conservative memory limits
            # Reserve more memory for VAE decode operations
            if use_multi_gpu and num_gpus > 1:
                click.echo(f"Multi-GPU mode: Distributing FLUX.2 Klein across {num_gpus} GPUs...")
                flux_kwargs["device_map"] = "balanced"
                # Reserve 6GB per GPU for VAE decode and intermediate tensors
                gpu_max_memory = {}
                for i in range(num_gpus):
                    try:
                        mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                        gpu_max_memory[i] = f"{int(mem_gb - 6)}GB"  # Reserve 6GB
                    except Exception:
                        pass
                if gpu_max_memory:
                    flux_kwargs["max_memory"] = gpu_max_memory
                    click.echo(f"GPU memory allocation: {gpu_max_memory}")

            try:
                from diffusers import Flux2KleinPipeline
                pipe = Flux2KleinPipeline.from_pretrained(model, **flux_kwargs)

                # Enable memory optimizations
                if hasattr(pipe, "enable_vae_slicing"):
                    pipe.enable_vae_slicing()
                if hasattr(pipe, "enable_vae_tiling"):
                    pipe.enable_vae_tiling()

                # Check if device_map was applied
                has_device_map = hasattr(pipe, "hf_device_map") and pipe.hf_device_map
                if has_device_map:
                    click.echo(f"Model distributed: {pipe.hf_device_map}")
                elif num_gpus > 1:
                    # device_map didn't work, fall back to model CPU offload
                    click.echo("device_map not supported, using model CPU offload...")
                    pipe.enable_model_cpu_offload()
                else:
                    # Single GPU - use CPU offload for safety
                    click.echo("Enabling model CPU offload for FLUX.2 Klein...")
                    pipe.enable_model_cpu_offload()

                # Return early - pipeline is fully configured
                return pipe
            except ImportError:
                click.echo(
                    "Flux2KleinPipeline not available. Installing diffusers from main branch...",
                )
                # Auto-install diffusers from main branch
                from hftool.core.download import install_pip_dependencies
                success = install_pip_dependencies(
                    ["git+https://github.com/huggingface/diffusers"],
                    force=True
                )
                if success:
                    click.echo("Diffusers updated. Please restart hftool to use FLUX.2 Klein.")
                    click.echo("Run the same command again.")
                    import sys
                    sys.exit(0)
                else:
                    click.echo(
                        "Error: Failed to install diffusers. Please run manually:\n"
                        "  pip install git+https://github.com/huggingface/diffusers",
                        err=True
                    )
                    raise
            except Exception as e:
                error_msg = str(e)
                if "gated" in error_msg.lower() or "access" in error_msg.lower():
                    click.echo(
                        "Error: FLUX.2-klein-9B requires accepting the license agreement.\n"
                        "Visit https://huggingface.co/black-forest-labs/FLUX.2-klein-9B to accept the terms.",
                        err=True
                    )
                raise
        elif is_qwen:
            click.echo("Using Qwen Image Edit pipeline...")
            
            # Check diffusers version - Qwen Image Edit needs >= 0.36.0
            import diffusers
            from packaging import version
            diffusers_version = version.parse(diffusers.__version__)
            if diffusers_version < version.parse("0.36.0"):
                click.echo(
                    f"Warning: diffusers {diffusers.__version__} detected. "
                    f"Qwen Image Edit requires diffusers >= 0.36.0",
                    err=True
                )
                click.echo("Upgrade with: pip install --upgrade diffusers>=0.36.0", err=True)
            
            # Qwen works best with bfloat16
            qwen_kwargs = {"torch_dtype": torch.bfloat16}
            
            # Add device_map if multi-GPU
            if use_multi_gpu and num_gpus > 1:
                qwen_kwargs["device_map"] = "balanced"
                if max_memory:
                    qwen_kwargs["max_memory"] = max_memory
            
            # Try the dedicated pipeline
            try:
                from diffusers import QwenImageEditPlusPipeline
                pipe = QwenImageEditPlusPipeline.from_pretrained(model, **qwen_kwargs)
            except ImportError:
                click.echo(
                    "Error: QwenImageEditPlusPipeline not available. "
                    "Please upgrade diffusers: pip install --upgrade diffusers>=0.36.0",
                    err=True
                )
                raise
            except Exception as e:
                error_msg = str(e)
                if "to_dict" in error_msg:
                    click.echo(
                        "Error: Version mismatch detected. This usually means:\n"
                        "  1. diffusers needs to be upgraded: pip install --upgrade diffusers>=0.36.0\n"
                        "  2. transformers needs to be upgraded: pip install --upgrade transformers>=4.45.0",
                        err=True
                    )
                raise
        # Check if this is SDXL refiner
        elif "refiner" in model_lower:
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
        
        # Check if pipeline has been placed on devices via device_map
        has_device_map = hasattr(pipe, "hf_device_map") and pipe.hf_device_map
        
        if has_device_map:
            click.echo(f"Model distributed across devices: {pipe.hf_device_map}")
        elif use_cpu_offload:
            # Apply CPU offload strategy
            if use_sequential and hasattr(pipe, "enable_sequential_cpu_offload"):
                click.echo("Enabling sequential CPU offload (most memory efficient)...")
                pipe.enable_sequential_cpu_offload()
            elif hasattr(pipe, "enable_model_cpu_offload"):
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
    
    def _is_qwen_pipeline(self, pipeline: Any) -> bool:
        """Check if the pipeline is a Qwen Image Edit pipeline."""
        return "QwenImageEdit" in type(pipeline).__name__

    def _is_flux2_klein_pipeline(self, pipeline: Any) -> bool:
        """Check if the pipeline is a FLUX.2 Klein pipeline."""
        return "Flux2Klein" in type(pipeline).__name__
    
    def run_inference(self, pipeline: Any, input_data: Any, **kwargs) -> Any:
        """Run image-to-image inference.
        
        Args:
            pipeline: Loaded diffusers pipeline
            input_data: Image path or dict with {"image": path, "prompt": text}
                       For Qwen, "image" can be a list of paths for multi-image editing
            **kwargs: Additional inference arguments
                - strength: How much to transform (0.0-1.0) - for SDXL models
                - true_cfg_scale: CFG scale for Qwen (default 4.0)
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
        image_paths, prompt = self._parse_input(input_data)
        
        if not image_paths:
            raise ValueError("No image provided. Use: -i '{\"image\": \"path.png\", \"prompt\": \"text\"}'")
        
        # Load the input image(s)
        is_qwen = self._is_qwen_pipeline(pipeline)
        is_flux2_klein = self._is_flux2_klein_pipeline(pipeline)

        if isinstance(image_paths, list):
            # Multiple images (Qwen/FLUX.2 multi-image support)
            click.echo(f"Loading {len(image_paths)} input images...")
            images = [Image.open(p).convert("RGB") for p in image_paths]
            for i, p in enumerate(image_paths):
                click.echo(f"  [image {i+1}] {p}")
            if is_flux2_klein:
                click.echo("Tip: Reference images in prompt using 'image 1', 'image 2', etc.")
        else:
            # Single image
            click.echo(f"Loading input image: {image_paths}")
            img = Image.open(image_paths).convert("RGB")
            # Qwen and FLUX.2 Klein expect a list even for single images
            images = [img] if (is_qwen or is_flux2_klein) else img
            if is_flux2_klein:
                click.echo("Tip: Reference this image in prompt using 'image 1' or describe what to edit.")

        # Handle seed
        seed = kwargs.pop("seed", None)
        if seed is not None and "generator" not in kwargs:
            # Qwen uses CPU generator for reproducibility
            if is_qwen:
                kwargs["generator"] = torch.manual_seed(seed)
            else:
                # FLUX.2 Klein and others use device-specific generator
                device = "cuda" if torch.cuda.is_available() else "cpu"
                kwargs["generator"] = torch.Generator(device=device).manual_seed(seed)

        # Get model defaults and merge
        defaults = self.get_default_kwargs()
        inference_kwargs = {**defaults, **kwargs}

        # Remove any None values
        inference_kwargs = {k: v for k, v in inference_kwargs.items() if v is not None}

        # Filter out incompatible parameters for specific pipelines
        if is_flux2_klein:
            # FLUX.2 Klein doesn't use these SDXL parameters
            for param in ["strength", "negative_prompt", "true_cfg_scale"]:
                inference_kwargs.pop(param, None)
        elif is_qwen:
            # Qwen doesn't use strength
            inference_kwargs.pop("strength", None)

        click.echo(f"Prompt: {prompt or '(none - refining only)'}")

        if is_qwen:
            click.echo(f"True CFG Scale: {inference_kwargs.get('true_cfg_scale', 4.0)}")
            click.echo(f"Steps: {inference_kwargs.get('num_inference_steps', 40)}")
        elif is_flux2_klein:
            click.echo(f"Steps: {inference_kwargs.get('num_inference_steps', 4)}")
            click.echo(f"Size: {inference_kwargs.get('width', 1024)}x{inference_kwargs.get('height', 1024)}")
        else:
            click.echo(f"Strength: {inference_kwargs.get('strength', 0.5)}")
        
        # Clear CUDA cache before inference to maximize available memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Run inference with OOM handling
        try:
            with torch.inference_mode():
                result = pipeline(
                    prompt=prompt,
                    image=images,
                    **inference_kwargs
                )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                click.echo("", err=True)
                click.echo(click.style("Out of GPU memory!", fg="red"), err=True)
                if is_flux2_klein:
                    click.echo("FLUX.2 Klein requires significant VRAM for VAE decode.", err=True)
                    click.echo("Try one of these solutions:", err=True)
                    click.echo("  1. Lower resolution: Add height=768, width=768 to input", err=True)
                    click.echo("  2. Use CPU offload: HFTOOL_CPU_OFFLOAD=1 hftool -t i2i ...", err=True)
                    click.echo("  3. Disable multi-GPU: HFTOOL_MULTI_GPU=0 hftool -t i2i ...", err=True)
                else:
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
