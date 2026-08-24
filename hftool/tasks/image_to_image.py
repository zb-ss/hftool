"""Catalog-driven image-to-image task handler."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

from hftool.io.output_handler import save_image
from hftool.tasks.base import BaseTask
from hftool.tasks.diffusion_utils import load_catalog_pipeline


class ImageToImageTask(BaseTask):
    """Edit or refine one or more images with a catalog-selected pipeline."""

    def _parse_input(self, input_data: Any) -> tuple[Any, str]:
        """Extract image path(s) and prompt from a path or JSON object."""
        if isinstance(input_data, dict):
            return input_data.get("image"), input_data.get("prompt", "")
        if isinstance(input_data, str):
            if input_data.strip().startswith("{"):
                try:
                    data = json.loads(input_data)
                except json.JSONDecodeError:
                    return input_data, ""
                return data.get("image"), data.get("prompt", "")
            return input_data, ""
        raise ValueError(f"Invalid input format: {type(input_data).__name__}")

    def load_pipeline(self, model: str, **kwargs: Any) -> Any:
        """Load the exact pipeline declared by the model catalog."""
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
    def _is_qwen_pipeline(pipeline: Any) -> bool:
        return type(pipeline).__name__.startswith("QwenImageEdit")

    @staticmethod
    def _is_flux2_klein_pipeline(pipeline: Any) -> bool:
        return type(pipeline).__name__.startswith("Flux2Klein")

    @staticmethod
    def _load_images(
        image_paths: Union[str, List[str]],
        *,
        expects_list: bool,
    ) -> Any:
        from PIL import Image

        if isinstance(image_paths, list):
            return [Image.open(path).convert("RGB") for path in image_paths]
        image = Image.open(image_paths).convert("RGB")
        return [image] if expects_list else image

    def run_inference(self, pipeline: Any, input_data: Any, **kwargs: Any) -> Any:
        """Run editing/refinement with deterministic seeds and helpful OOM errors."""
        import click
        import torch

        image_paths, prompt = self._parse_input(input_data)
        if not image_paths:
            raise ValueError(
                "No image provided. Use: -i "
                "'{\"image\": \"path.png\", \"prompt\": \"edit\"}'"
            )

        is_qwen = self._is_qwen_pipeline(pipeline)
        is_flux2 = self._is_flux2_klein_pipeline(pipeline)
        images = self._load_images(
            image_paths,
            expects_list=is_qwen or is_flux2,
        )

        seed = kwargs.pop("seed", None)
        if seed is not None and "generator" not in kwargs:
            if is_qwen:
                kwargs["generator"] = torch.manual_seed(seed)
            else:
                generator_device = "cuda" if torch.cuda.is_available() else "cpu"
                kwargs["generator"] = torch.Generator(device=generator_device).manual_seed(seed)

        inference_kwargs = {key: value for key, value in kwargs.items() if value is not None}
        if is_flux2:
            for incompatible in ("strength", "negative_prompt", "true_cfg_scale"):
                inference_kwargs.pop(incompatible, None)
        elif is_qwen:
            inference_kwargs.pop("strength", None)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            with torch.inference_mode():
                result = pipeline(prompt=prompt, image=images, **inference_kwargs)
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            click.echo(click.style("Out of GPU memory.", fg="red"), err=True)
            click.echo(
                "Free GPU memory, lower the output resolution, or retry with "
                "HFTOOL_CPU_OFFLOAD=1.",
                err=True,
            )
            raise

        return result.images[0]

    def save_output(self, result: Any, output_path: str, **kwargs: Any) -> str:
        """Save the generated image."""
        return save_image(result, output_path, **kwargs)


def create_task(device: str = "auto", dtype: Optional[str] = None) -> ImageToImageTask:
    """Create an image-to-image task handler."""
    return ImageToImageTask(device=device, dtype=dtype)
