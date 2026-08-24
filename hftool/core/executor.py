"""Task execution engine — shared by CLI and TUI frontends.

Provides a frontend-agnostic interface for running hftool tasks:
    request = TaskRequest(task_name="text-to-image", input_data="A cat")
    result = execute_task(request)
"""

import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class TaskRequest:
    """Frontend-agnostic task execution request."""
    task_name: str
    model: Optional[str] = None
    input_data: str = ""
    output_path: Optional[str] = None
    device: str = "auto"
    dtype: Optional[str] = None
    gpu: Optional[str] = None
    seed: Optional[int] = None
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
    auto_download: bool = False
    embed_metadata: bool = True
    progress_callback: Optional[Callable] = None


@dataclass
class TaskResult:
    """Result of a task execution."""
    success: bool
    output_path: Optional[str] = None
    result_data: Any = None
    error: Optional[str] = None
    elapsed_s: float = 0.0
    task_name: str = ""
    model: Optional[str] = None
    seed: Optional[int] = None


@dataclass(frozen=True)
class TaskPreview:
    """A no-download, no-load execution preview shared by TUI and CLI."""

    task_name: str
    model_key: str
    model_name: str
    repo_id: str
    revision: Optional[str]
    profile: Optional[str]
    adapter_repo_id: Optional[str]
    adapter_revision: Optional[str]
    adapter_weight_name: Optional[str]
    status: str
    download_status: str
    total_download_gb: float
    min_vram_gb: Optional[float]
    recommended_vram_gb: Optional[float]
    dtype: Optional[str]
    license: Optional[str]
    commercial_use: Optional[bool]
    gated: bool
    dependencies: List[str]
    inference_defaults: Dict[str, Any]
    gpu_message: str
    gpu_adequate: Optional[bool]

    def as_dict(self) -> Dict[str, Any]:
        """Return a stable JSON-ready representation."""
        return {
            "task": self.task_name,
            "model": self.model_key,
            "model_name": self.model_name,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "profile": self.profile,
            "adapter_repo_id": self.adapter_repo_id,
            "adapter_revision": self.adapter_revision,
            "adapter_weight_name": self.adapter_weight_name,
            "catalog_status": self.status,
            "download_status": self.download_status,
            "total_download_gb": self.total_download_gb,
            "min_vram_gb": self.min_vram_gb,
            "recommended_vram_gb": self.recommended_vram_gb,
            "dtype": self.dtype,
            "license": self.license,
            "commercial_use": self.commercial_use,
            "gated": self.gated,
            "dependencies": list(self.dependencies),
            "inference_defaults": dict(self.inference_defaults),
            "gpu_message": self.gpu_message,
            "gpu_adequate": self.gpu_adequate,
        }


def _resolve_catalog_model(task_name: str, model: Optional[str]):
    """Resolve task/config aliases and return canonical model context."""
    from hftool.core.config import Config
    from hftool.core.models import (
        get_default_model_info,
        get_model_info,
        get_model_key,
    )
    from hftool.core.registry import TASK_ALIASES

    resolved_task = TASK_ALIASES.get(task_name, task_name)
    config = Config.get()
    resolved_model = model
    if resolved_model is None:
        resolved_model = config.get_value("model", task=resolved_task, default=None)
    if resolved_model:
        resolved_model = config.resolve_model_alias(resolved_model)
        info = get_model_info(resolved_task, resolved_model)
        model_key = get_model_key(resolved_task, resolved_model)
    else:
        info = get_default_model_info(resolved_task)
        model_key = get_model_key(resolved_task, info.repo_id)
    return resolved_task, model_key, info


def _select_gpu_for_model(model_info, gpu_arg: Optional[str]):
    """Select or validate one visible GPU for a catalog model."""
    from hftool.core.device import (
        get_all_gpus,
        parse_gpu_selection,
        select_compute_gpu,
    )

    gpus = get_all_gpus()
    if not gpus:
        return None
    if gpu_arg == "all":
        return None
    if gpu_arg and gpu_arg != "auto":
        visible_indices = parse_gpu_selection(gpu_arg, model_info.min_vram_gb)
        selected_gpus = [gpu for gpu in gpus if gpu.index in visible_indices]
        return select_compute_gpu(model_info.min_vram_gb, gpus=selected_gpus)
    return select_compute_gpu(model_info.min_vram_gb, gpus=gpus)


def _is_multi_gpu_enabled(gpu_arg: Optional[str]) -> bool:
    """Return whether this request or its container explicitly enabled sharding."""
    if gpu_arg == "all" or bool(gpu_arg and "," in gpu_arg):
        return True
    return os.environ.get("HFTOOL_MULTI_GPU", "").lower() in {
        "1",
        "true",
        "yes",
        "balanced",
    }


def preview_task(request: TaskRequest) -> TaskPreview:
    """Preview a catalog task without importing a pipeline or downloading data."""
    from hftool.core.download import get_model_download_status

    resolved_task, model_key, model_info = _resolve_catalog_model(
        request.task_name,
        request.model,
    )
    multi_gpu = _is_multi_gpu_enabled(request.gpu)
    selection = (
        None
        if request.device == "cpu" or multi_gpu
        else _select_gpu_for_model(model_info, request.gpu)
    )
    adapter_size = model_info.adapter.size_gb if model_info.adapter else 0.0
    return TaskPreview(
        task_name=resolved_task,
        model_key=model_key,
        model_name=model_info.name,
        repo_id=model_info.repo_id,
        revision=model_info.revision,
        profile=model_info.profile,
        adapter_repo_id=model_info.adapter.repo_id if model_info.adapter else None,
        adapter_revision=model_info.adapter.revision if model_info.adapter else None,
        adapter_weight_name=model_info.adapter.weight_name if model_info.adapter else None,
        status=model_info.status_label,
        download_status=get_model_download_status(model_info),
        total_download_gb=model_info.size_gb + adapter_size,
        min_vram_gb=model_info.min_vram_gb,
        recommended_vram_gb=model_info.recommended_vram_gb,
        dtype=request.dtype or model_info.dtype,
        license=model_info.license,
        commercial_use=model_info.commercial_use,
        gated=model_info.gated,
        dependencies=list(model_info.pip_dependencies),
        inference_defaults=dict(model_info.inference_defaults),
        gpu_message=(
            selection.format_message()
            if selection is not None
            else "Multi-GPU explicitly requested." if multi_gpu else "No GPU visible."
        ),
        gpu_adequate=selection.adequate if selection is not None else None,
    )


def build_model_runtime_kwargs(model_info, adapter_path: Optional[str] = None):
    """Build exact catalog-declared load and inference kwargs."""
    load_kwargs = dict(model_info.load_defaults)
    infer_kwargs = dict(model_info.inference_defaults)
    if model_info.pipeline_class:
        load_kwargs["_pipeline_class"] = model_info.pipeline_class
    if model_info.dtype:
        load_kwargs["_catalog_dtype"] = model_info.dtype
    if model_info.adapter:
        adapter_source = adapter_path
        if adapter_source and Path(adapter_source).name == model_info.adapter.weight_name:
            adapter_source = str(Path(adapter_source).parent)
        load_kwargs.update(
            {
                "_adapter_path": adapter_source,
                "_adapter_weight_name": model_info.adapter.weight_name,
                "_adapter_scale": model_info.adapter.scale,
                "_scheduler_class": model_info.adapter.scheduler_class,
                "_scheduler_config": dict(model_info.adapter.scheduler_config),
            }
        )
    return load_kwargs, infer_kwargs


def execute_task(request: TaskRequest) -> TaskResult:
    """Execute a task from any frontend (CLI, TUI, API).

    Handles: task alias resolution, config merging, model resolution,
    download check, dependency check, task loading, pipeline execution,
    output saving, and metadata embedding.
    """
    start_time = time.time()
    previous_multi_gpu = os.environ.get("HFTOOL_MULTI_GPU")
    has_multi_gpu_override = request.gpu == "all" or bool(
        request.gpu and "," in request.gpu
    )

    try:
        from hftool.core.registry import get_task_config, TASK_ALIASES
        from hftool.core.models import get_default_model_info, get_model_info
        from hftool.core.download import (
            ensure_model_available,
            ensure_model_file_available,
        )
        from hftool.core.config import Config
        from hftool.utils.deps import is_available, is_ffmpeg_available

        # Resolve task alias
        resolved_task = TASK_ALIASES.get(request.task_name, request.task_name)

        # Generate random seed if not provided
        seed = request.seed
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        extra_kwargs = dict(request.extra_kwargs)
        if "generator_seed" not in extra_kwargs and "seed" not in extra_kwargs:
            extra_kwargs["seed"] = seed

        # Load configuration
        config = Config.get()
        device = request.device
        dtype = request.dtype
        model = request.model

        if device == "auto":
            device = config.get_value("device", task=resolved_task, default="auto")
        if dtype is None:
            dtype = config.get_value("dtype", task=resolved_task, default=None)
        if model is None:
            model = config.get_value("model", task=resolved_task, default=None)
        if model:
            model = config.resolve_model_alias(model)

        # Merge config task-specific params
        task_params = {}
        if resolved_task in config._config:
            task_config_section = config._config[resolved_task]
            if isinstance(task_config_section, dict):
                reserved_keys = {'model', 'device', 'dtype'}
                task_params = {k: v for k, v in task_config_section.items() if k not in reserved_keys}
        extra_kwargs = {**task_params, **extra_kwargs}

        # Get task configuration
        task_config = get_task_config(resolved_task)

        # Check dependencies
        missing = []
        for dep in task_config.required_deps:
            if not is_available(dep):
                missing.append(dep)
        if missing:
            raise RuntimeError(f"Missing dependencies: {', '.join(missing)}. Install with: pip install {' '.join(missing)}")
        if task_config.requires_ffmpeg and not is_ffmpeg_available():
            raise RuntimeError("ffmpeg is required for this task but was not found.")

        # Resolve model
        model_info = None
        pip_dependencies = None
        model_gated = False

        if model is None:
            model_info = get_default_model_info(resolved_task)
            model_repo_id = model_info.repo_id
            model_size = model_info.size_gb
            model_name = model_info.name
            pip_dependencies = model_info.pip_dependencies
            model_gated = getattr(model_info, 'gated', False)
        else:
            if os.path.exists(model):
                model_repo_id = model
                model_size = 0
                model_name = os.path.basename(model)
            else:
                try:
                    model_info = get_model_info(resolved_task, model)
                    model_repo_id = model_info.repo_id
                    model_size = model_info.size_gb
                    model_name = model_info.name
                    pip_dependencies = model_info.pip_dependencies
                    model_gated = getattr(model_info, 'gated', False)
                except ValueError:
                    model_repo_id = model
                    model_size = 5.0
                    model_name = model.split("/")[-1] if "/" in model else model

        # Select using live free VRAM before any download or pipeline load.
        # Multi-GPU remains an explicit request and is evaluated by Accelerate.
        if has_multi_gpu_override:
            os.environ["HFTOOL_MULTI_GPU"] = "1"
        elif model_info and device != "cpu" and not _is_multi_gpu_enabled(request.gpu):
            selection = _select_gpu_for_model(model_info, request.gpu)
            if selection is not None:
                extra_kwargs["_gpu_indices"] = [selection.gpu.index]
                if device in ("auto", "cuda"):
                    device = f"cuda:{selection.gpu.index}"
                cpu_offload = os.environ.get("HFTOOL_CPU_OFFLOAD", "").lower()
                if (
                    resolved_task in {"text-to-image", "image-to-image"}
                    and not selection.adequate
                    and cpu_offload not in {"1", "2", "true", "yes"}
                ):
                    raise RuntimeError(
                        f"Insufficient live GPU headroom. {selection.format_message()} "
                        "Free VRAM, choose another physical GPU, use --gpu all, or "
                        "set HFTOOL_CPU_OFFLOAD=1."
                    )

        # Ensure model is available
        if not os.path.exists(model_repo_id):
            model_path = ensure_model_available(
                repo_id=model_repo_id,
                size_gb=model_size,
                task_name=resolved_task,
                model_name=model_name,
                pip_dependencies=pip_dependencies,
                gated=model_gated,
                auto_download=request.auto_download,
                revision=model_info.revision if model_info else None,
                ignore_patterns=model_info.ignore_patterns if model_info else None,
            )
            model_to_load = str(model_path)
        else:
            model_to_load = model_repo_id

        adapter_path = None
        if model_info and model_info.adapter:
            adapter = model_info.adapter
            adapter_path = ensure_model_file_available(
                repo_id=adapter.repo_id,
                filename=adapter.weight_name,
                size_gb=adapter.size_gb,
                task_name=resolved_task,
                model_name=model_name,
                auto_download=request.auto_download,
                revision=adapter.revision,
            )

        # Determine output path
        output_file = request.output_path
        if output_file is None:
            from hftool.io.output_handler import get_output_path, OutputType
            import json as json_module
            output_type_map = {
                "text": OutputType.TEXT,
                "image": OutputType.IMAGE,
                "audio": OutputType.AUDIO,
                "video": OutputType.VIDEO,
            }
            output_type = output_type_map.get(task_config.output_type, OutputType.TEXT)
            actual_input_path = None
            if task_config.input_type != "text":
                if request.input_data.strip().startswith("{"):
                    try:
                        data = json_module.loads(request.input_data)
                        img_path = data.get("image")
                        if isinstance(img_path, list):
                            actual_input_path = img_path[0] if img_path else None
                        else:
                            actual_input_path = img_path
                    except (json_module.JSONDecodeError, TypeError):
                        pass
                else:
                    actual_input_path = request.input_data
            output_file = get_output_path(input_path=actual_input_path, output_type=output_type)

        # Load and run task handler
        task_handler = _create_task_handler(resolved_task, task_config, device, dtype)

        load_kwargs, infer_kwargs = (
            build_model_runtime_kwargs(
                model_info,
                str(adapter_path) if adapter_path else None,
            )
            if model_info
            else ({}, {})
        )

        # Execute
        filtered_kwargs = {k: v for k, v in extra_kwargs.items() if not k.startswith("_")}
        result = task_handler.execute(
            model=model_to_load,
            input_data=request.input_data,
            output_path=output_file,
            load_kwargs=load_kwargs,
            infer_kwargs=infer_kwargs,
            **filtered_kwargs,
        )

        metadata_kwargs = dict(extra_kwargs)
        if model_info:
            metadata_kwargs["catalog"] = {
                "repo_id": model_info.repo_id,
                "revision": model_info.revision,
                "profile": model_info.profile,
                "dtype": dtype or model_info.dtype,
                "pipeline_class": model_info.pipeline_class,
                "inference_defaults": dict(model_info.inference_defaults),
                "adapter_repo_id": model_info.adapter.repo_id if model_info.adapter else None,
                "adapter_revision": model_info.adapter.revision if model_info.adapter else None,
                "adapter_weight_name": model_info.adapter.weight_name if model_info.adapter else None,
            }

        # Embed exact catalog/profile provenance with the generation settings.
        if output_file and request.embed_metadata and os.path.exists(output_file):
            _embed_output_metadata(
                output_file, resolved_task, model or model_repo_id,
                request.input_data, seed, metadata_kwargs,
            )

        elapsed = time.time() - start_time
        return TaskResult(
            success=True,
            output_path=output_file,
            result_data=result,
            elapsed_s=elapsed,
            task_name=resolved_task,
            model=model or model_repo_id,
            seed=seed,
        )

    except Exception as e:
        elapsed = time.time() - start_time
        return TaskResult(
            success=False,
            error=str(e),
            elapsed_s=elapsed,
            task_name=request.task_name,
            model=request.model,
            seed=request.seed,
        )
    finally:
        if has_multi_gpu_override:
            if previous_multi_gpu is None:
                os.environ.pop("HFTOOL_MULTI_GPU", None)
            else:
                os.environ["HFTOOL_MULTI_GPU"] = previous_multi_gpu


def _create_task_handler(task_name: str, task_config, device: str, dtype: Optional[str]):
    """Create the appropriate task handler for a given task name."""
    if task_name == "text-to-image":
        from hftool.tasks.text_to_image import create_task
        return create_task(device=device, dtype=dtype)
    elif task_name == "image-to-image":
        from hftool.tasks.image_to_image import create_task
        return create_task(device=device, dtype=dtype)
    elif task_name in ("text-to-video", "image-to-video"):
        from hftool.tasks.text_to_video import create_task
        mode = task_config.config.get("mode", "t2v")
        return create_task(device=device, dtype=dtype, mode=mode)
    elif task_name == "text-to-speech":
        from hftool.tasks.text_to_speech import create_task
        return create_task(device=device, dtype=dtype)
    elif task_name == "automatic-speech-recognition":
        from hftool.tasks.speech_to_text import create_task
        return create_task(device=device, dtype=dtype)
    else:
        from hftool.tasks.transformers_generic import create_task
        return create_task(task_name=task_name, device=device, dtype=dtype)


def _embed_output_metadata(
    output_file: str,
    task: str,
    model: str,
    input_data: str,
    seed: Optional[int],
    extra_kwargs: Dict[str, Any],
):
    """Embed generation metadata in an output file."""
    import json as json_module
    from hftool.core.metadata import embed_metadata

    prompt = None
    if isinstance(input_data, str):
        try:
            data = json_module.loads(input_data)
            if isinstance(data, dict):
                prompt = data.get("prompt") or data.get("text") or data.get("caption")
        except json_module.JSONDecodeError:
            prompt = input_data[:500] if len(input_data) > 500 else input_data

    embed_metadata(
        file_path=output_file,
        task=task,
        model=model,
        prompt=prompt,
        seed=seed,
        extra_params=extra_kwargs,
        verbose=False,
    )
