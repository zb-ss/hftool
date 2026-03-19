"""Task execution engine — shared by CLI and TUI frontends.

Provides a frontend-agnostic interface for running hftool tasks:
    request = TaskRequest(task_name="text-to-image", input_data="A cat")
    result = execute_task(request)
"""

import os
import random
import time
from dataclasses import dataclass, field
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


def execute_task(request: TaskRequest) -> TaskResult:
    """Execute a task from any frontend (CLI, TUI, API).

    Handles: task alias resolution, config merging, model resolution,
    download check, dependency check, task loading, pipeline execution,
    output saving, and metadata embedding.
    """
    start_time = time.time()

    try:
        from hftool.core.registry import get_task_config, TASK_ALIASES
        from hftool.core.models import get_default_model_info, get_model_info, find_model_by_repo_id
        from hftool.core.download import ensure_model_available
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

        # Parse GPU selection
        if request.gpu and request.device in ("auto", "cuda"):
            from hftool.core.device import parse_gpu_selection
            gpu_indices = parse_gpu_selection(request.gpu)
            if gpu_indices:
                extra_kwargs["_gpu_indices"] = gpu_indices

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

        # Ensure model is available
        if not os.path.exists(model_repo_id):
            model_path = ensure_model_available(
                repo_id=model_repo_id,
                size_gb=model_size,
                task_name=resolved_task,
                model_name=model_name,
                pip_dependencies=pip_dependencies,
                gated=model_gated,
            )
            model_to_load = str(model_path)
        else:
            model_to_load = model_repo_id

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

        # Separate load-time vs inference-time metadata
        model_metadata = model_info.metadata if model_info else {}
        load_param_names = {"subfolder", "revision", "variant", "torch_dtype", "use_safetensors"}
        load_kwargs = {}
        infer_kwargs = {}
        if model_metadata:
            for key, value in model_metadata.items():
                if key in load_param_names:
                    load_kwargs[key] = value
                else:
                    infer_kwargs[key] = value

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

        # Embed metadata
        if output_file and request.embed_metadata and os.path.exists(output_file):
            _embed_output_metadata(
                output_file, resolved_task, model or model_repo_id,
                request.input_data, seed, extra_kwargs,
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
