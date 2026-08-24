"""Catalog-driven text-to-image task handler."""

from __future__ import annotations

from typing import Any, Dict, Optional

from hftool.io.output_handler import save_image
from hftool.tasks.base import BaseTask, TextInputMixin
from hftool.tasks.diffusion_utils import load_catalog_pipeline


class TextToImageTask(TextInputMixin, BaseTask):
    """Generate images with a catalog-selected Diffusers pipeline."""

    def load_pipeline(self, model: str, **kwargs: Any) -> Any:
        """Load the exact pipeline and optional profile declared by the catalog."""
        return load_catalog_pipeline(
            model,
            device=self.device,
            requested_dtype=self.dtype,
            load_kwargs=kwargs,
        )

    def get_default_kwargs(self) -> Dict[str, Any]:
        """Defaults are authoritative in the packaged model catalog."""
        return {}

    @staticmethod
    def _generator_for_pipeline(pipeline: Any, seed: int) -> Any:
        """Create a deterministic generator on a pipeline-compatible device."""
        import torch

        if type(pipeline).__name__.startswith("QwenImage"):
            return torch.manual_seed(seed)

        execution_device = getattr(pipeline, "_execution_device", None)
        if execution_device is None:
            component = getattr(pipeline, "transformer", None)
            if component is None:
                component = getattr(pipeline, "unet", None)
            if component is not None:
                try:
                    execution_device = next(component.parameters()).device
                except (AttributeError, StopIteration):
                    execution_device = None
        generator_device = str(execution_device) if execution_device is not None else "cpu"
        return torch.Generator(device=generator_device).manual_seed(seed)

    def run_inference(self, pipeline: Any, prompt: str, **kwargs: Any) -> Any:
        """Run text-to-image inference with deterministic seed and OOM guidance."""
        import click
        import torch

        seed = kwargs.pop("seed", None)
        if seed is not None and "generator" not in kwargs:
            kwargs["generator"] = self._generator_for_pipeline(pipeline, seed)

        inference_kwargs = {key: value for key, value in kwargs.items() if value is not None}
        try:
            with torch.inference_mode():
                result = pipeline(prompt=prompt, **inference_kwargs)
        except RuntimeError as error:
            error_message = str(error).lower()
            if "out of memory" not in error_message and "hip out of memory" not in error_message:
                raise

            height = inference_kwargs.get("height", 1024)
            width = inference_kwargs.get("width", 1024)
            click.echo(click.style("Out of GPU memory.", fg="red"), err=True)
            click.echo(
                f"Current request: {width}x{height}. Free GPU memory or lower the resolution, "
                "or retry with HFTOOL_CPU_OFFLOAD=1.",
                err=True,
            )
            raise

        return result.images[0]

    def save_output(self, result: Any, output_path: str, **kwargs: Any) -> str:
        """Save the generated image."""
        return save_image(result, output_path, **kwargs)


def create_task(device: str = "auto", dtype: Optional[str] = None) -> TextToImageTask:
    """Create a text-to-image task handler."""
    return TextToImageTask(device=device, dtype=dtype)
