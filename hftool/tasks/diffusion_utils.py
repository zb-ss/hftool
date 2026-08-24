"""Shared Diffusers pipeline loading for image-generation tasks."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


def _is_enabled(name: str) -> bool:
    """Return whether an opt-in environment flag is enabled."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_dtype(torch: Any, device: str, requested: Optional[str], catalog_dtype: Optional[str]) -> Any:
    """Resolve frontend and catalog dtype names to a torch dtype."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if requested:
        if requested not in dtype_map:
            raise ValueError(f"Unsupported dtype '{requested}'. Use bfloat16, float16, or float32.")
        return dtype_map[requested]
    if device == "cpu":
        return torch.float32
    if catalog_dtype:
        if catalog_dtype not in dtype_map:
            raise ValueError(f"Catalog specifies unsupported dtype '{catalog_dtype}'")
        return dtype_map[catalog_dtype]

    from hftool.core.device import get_optimal_dtype

    return get_optimal_dtype(device)


def _configure_scheduler(diffusers: Any, class_name: Optional[str], config: Dict[str, Any]) -> Any:
    """Build a catalog-declared scheduler without importing arbitrary code."""
    if not class_name:
        return None
    scheduler_class = getattr(diffusers, class_name, None)
    if scheduler_class is None:
        raise RuntimeError(
            f"{class_name} is unavailable in Diffusers. Install the model's declared dependencies."
        )
    return scheduler_class.from_config(dict(config))


def _apply_adapter(
    pipeline: Any,
    adapter_path: Optional[str],
    weight_name: Optional[str],
    scale: float,
) -> None:
    """Load one exact, locally resolved LoRA weight into a pipeline."""
    if not adapter_path:
        return
    if not weight_name:
        raise RuntimeError("Catalog adapter profile is missing weight_name")
    if not hasattr(pipeline, "load_lora_weights"):
        raise RuntimeError(f"{type(pipeline).__name__} does not support LoRA adapters")

    adapter_name = "hftool_profile"
    pipeline.load_lora_weights(
        adapter_path,
        weight_name=weight_name,
        adapter_name=adapter_name,
    )
    if hasattr(pipeline, "set_adapters"):
        pipeline.set_adapters(adapter_name, adapter_weights=scale)


def _apply_opt_in_optimizations(pipeline: Any, click: Any, torch: Any) -> None:
    """Apply explicitly requested optimizations with safe fallbacks."""
    attention_backend = os.environ.get("HFTOOL_ATTENTION_BACKEND", "").strip()
    if attention_backend:
        transformer = getattr(pipeline, "transformer", None)
        if transformer is None or not hasattr(transformer, "set_attention_backend"):
            click.echo(
                f"Note: attention backend '{attention_backend}' is not supported by this pipeline; using native SDPA.",
                err=True,
            )
        else:
            try:
                transformer.set_attention_backend(attention_backend)
                click.echo(f"Attention backend: {attention_backend}")
            except Exception as error:
                click.echo(
                    f"Note: attention backend '{attention_backend}' failed ({error}); using native SDPA.",
                    err=True,
                )

    if _is_enabled("HFTOOL_TORCH_COMPILE"):
        denoiser = getattr(pipeline, "transformer", None)
        if denoiser is None:
            denoiser = getattr(pipeline, "unet", None)
        if denoiser is None:
            click.echo("Note: no denoiser found for torch.compile; continuing uncompiled.", err=True)
        else:
            try:
                compiled = torch.compile(denoiser, fullgraph=True)
                if hasattr(pipeline, "transformer"):
                    pipeline.transformer = compiled
                else:
                    pipeline.unet = compiled
                click.echo("torch.compile enabled (compare cold and warm runs separately)")
            except Exception as error:
                click.echo(f"Note: torch.compile failed ({error}); continuing uncompiled.", err=True)


def load_catalog_pipeline(
    model: str,
    *,
    device: str,
    requested_dtype: Optional[str],
    load_kwargs: Dict[str, Any],
) -> Any:
    """Load a catalog-resolved Diffusers pipeline and optional adapter profile."""
    import click

    from hftool.utils.deps import check_dependencies

    check_dependencies(["diffusers", "torch", "accelerate"], extra="with_t2i")

    import diffusers
    import torch

    from hftool.core.device import configure_rocm_env, detect_device, get_multi_gpu_kwargs
    from hftool.core.models import get_catalog_runtime_config

    configure_rocm_env()
    resolved_device = device if device != "auto" else detect_device()

    options = dict(load_kwargs)
    pipeline_class_name = options.pop("_pipeline_class", None) or "DiffusionPipeline"
    catalog_dtype = options.pop("_catalog_dtype", None)
    adapter_path = options.pop("_adapter_path", None)
    adapter_weight_name = options.pop("_adapter_weight_name", None)
    adapter_scale = float(options.pop("_adapter_scale", 1.0))
    scheduler_class_name = options.pop("_scheduler_class", None)
    scheduler_config = options.pop("_scheduler_config", {})

    dtype = _resolve_dtype(torch, resolved_device, requested_dtype, catalog_dtype)
    options.setdefault("torch_dtype", dtype)

    if _is_enabled("HFTOOL_PARALLEL_LOADING"):
        os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "YES")

    runtime = get_catalog_runtime_config("gpu_selection")
    reserve_gb = float(runtime.get("safety_reserve_gb", 0.0))
    gpu_config = get_multi_gpu_kwargs(reserve_per_gpu_gb=reserve_gb)
    if gpu_config["message"]:
        click.echo(gpu_config["message"])
    if gpu_config["use_multi_gpu"]:
        options.setdefault("device_map", gpu_config["device_map"])
        options.setdefault("max_memory", gpu_config["max_memory"])

    scheduler = _configure_scheduler(
        diffusers,
        scheduler_class_name,
        scheduler_config if isinstance(scheduler_config, dict) else {},
    )
    if scheduler is not None:
        options["scheduler"] = scheduler

    pipeline_class = getattr(diffusers, pipeline_class_name, None)
    if pipeline_class is None:
        raise RuntimeError(
            f"{pipeline_class_name} is unavailable in Diffusers {diffusers.__version__}. "
            "Install the dependencies shown by 'hftool info <model>'."
        )

    click.echo(f"Loading {pipeline_class_name}...")
    pipeline = pipeline_class.from_pretrained(model, **options)
    _apply_adapter(pipeline, adapter_path, adapter_weight_name, adapter_scale)

    for optimization in ("enable_vae_slicing", "enable_vae_tiling"):
        method = getattr(pipeline, optimization, None)
        if method is not None:
            try:
                method()
            except Exception as error:
                click.echo(f"Note: {optimization} unavailable ({error})", err=True)

    has_device_map = bool(getattr(pipeline, "hf_device_map", None))
    cpu_offload = os.environ.get("HFTOOL_CPU_OFFLOAD", "").strip().lower()
    if has_device_map:
        click.echo(f"Model distributed across devices: {pipeline.hf_device_map}")
    elif cpu_offload == "2" and hasattr(pipeline, "enable_sequential_cpu_offload"):
        click.echo("Enabling sequential CPU offload...")
        pipeline.enable_sequential_cpu_offload()
    elif cpu_offload in {"1", "true", "yes"} and hasattr(pipeline, "enable_model_cpu_offload"):
        click.echo("Enabling model CPU offload...")
        pipeline.enable_model_cpu_offload()
    else:
        click.echo(f"Loading model on {resolved_device}...")
        pipeline.to(resolved_device)

    _apply_opt_in_optimizations(pipeline, click, torch)
    return pipeline
