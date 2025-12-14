"""Core module for hftool - device detection, registry, and pipeline management."""

from hftool.core.device import detect_device, get_optimal_dtype, get_device_info
from hftool.core.registry import TASK_REGISTRY, TaskConfig, get_task_config, list_tasks

__all__ = [
    "detect_device",
    "get_optimal_dtype",
    "get_device_info",
    "TASK_REGISTRY",
    "TaskConfig",
    "get_task_config",
    "list_tasks",
]
