"""Core module for hftool - device detection, registry, and pipeline management."""

from hftool.core.device import detect_device, get_optimal_dtype, get_device_info
from hftool.core.registry import TASK_REGISTRY, TaskConfig, get_task_config, list_tasks
from hftool.core.models import (
    MODEL_REGISTRY,
    ModelInfo,
    ModelType,
    get_models_for_task,
    get_model_info,
    get_default_model_info,
)
from hftool.core.download import (
    get_models_dir,
    get_model_path,
    is_model_downloaded,
    download_model,
    download_model_with_progress,
    ensure_model_available,
)

__all__ = [
    # Device
    "detect_device",
    "get_optimal_dtype",
    "get_device_info",
    # Registry
    "TASK_REGISTRY",
    "TaskConfig",
    "get_task_config",
    "list_tasks",
    # Models
    "MODEL_REGISTRY",
    "ModelInfo",
    "ModelType",
    "get_models_for_task",
    "get_model_info",
    "get_default_model_info",
    # Download
    "get_models_dir",
    "get_model_path",
    "is_model_downloaded",
    "download_model",
    "download_model_with_progress",
    "ensure_model_available",
]
