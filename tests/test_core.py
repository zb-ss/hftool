"""Tests for core modules: device detection and task registry."""

import pytest


class TestDeviceDetection:
    """Tests for hftool.core.device module."""
    
    def test_detect_device_returns_valid_device(self):
        """detect_device should return 'cuda', 'mps', or 'cpu'."""
        from hftool.core.device import detect_device
        
        device = detect_device()
        assert device in ("cuda", "mps", "cpu")
    
    def test_get_device_info_returns_dataclass(self):
        """get_device_info should return a DeviceInfo dataclass."""
        from hftool.core.device import get_device_info, DeviceInfo
        
        info = get_device_info()
        assert isinstance(info, DeviceInfo)
        assert info.device in ("cuda", "mps", "cpu")
        assert isinstance(info.is_rocm, bool)
        assert isinstance(info.is_cuda, bool)
        assert isinstance(info.is_mps, bool)
    
    def test_is_rocm_returns_bool(self):
        """is_rocm should return a boolean."""
        from hftool.core.device import is_rocm
        
        result = is_rocm()
        assert isinstance(result, bool)
    
    def test_get_optimal_dtype_returns_dtype(self):
        """get_optimal_dtype should return a torch dtype."""
        from hftool.core.device import get_optimal_dtype
        import torch
        
        dtype = get_optimal_dtype()
        assert dtype in (torch.bfloat16, torch.float16, torch.float32)
    
    def test_get_device_map_returns_string(self):
        """get_device_map should return a valid device map string."""
        from hftool.core.device import get_device_map
        
        device_map = get_device_map()
        assert isinstance(device_map, str)
        assert device_map in ("cuda:0", "auto", "mps", "cpu")


class TestTaskRegistry:
    """Tests for hftool.core.registry module."""
    
    def test_task_registry_not_empty(self):
        """TASK_REGISTRY should contain tasks."""
        from hftool.core.registry import TASK_REGISTRY
        
        assert len(TASK_REGISTRY) > 0
    
    def test_task_registry_has_expected_tasks(self):
        """TASK_REGISTRY should contain the main tasks."""
        from hftool.core.registry import TASK_REGISTRY
        
        expected_tasks = [
            "text-to-image",
            "text-to-video",
            "text-to-speech",
            "automatic-speech-recognition",
        ]
        
        for task in expected_tasks:
            assert task in TASK_REGISTRY, f"Missing task: {task}"
    
    def test_get_task_config_valid_task(self):
        """get_task_config should return TaskConfig for valid tasks."""
        from hftool.core.registry import get_task_config, TaskConfig
        
        config = get_task_config("text-to-image")
        assert isinstance(config, TaskConfig)
        assert config.library == "diffusers"
        assert config.input_type == "text"
        assert config.output_type == "image"
    
    def test_get_task_config_alias(self):
        """get_task_config should resolve task aliases."""
        from hftool.core.registry import get_task_config
        
        # These should all resolve to the same config
        config_t2i = get_task_config("t2i")
        config_full = get_task_config("text-to-image")
        
        assert config_t2i.handler == config_full.handler
    
    def test_get_task_config_invalid_task(self):
        """get_task_config should raise ValueError for invalid tasks."""
        from hftool.core.registry import get_task_config
        
        with pytest.raises(ValueError, match="Unknown task"):
            get_task_config("invalid-task-name")
    
    def test_list_tasks_returns_dict(self):
        """list_tasks should return a dict of task names to descriptions."""
        from hftool.core.registry import list_tasks
        
        tasks = list_tasks()
        assert isinstance(tasks, dict)
        assert len(tasks) > 0
        
        # All values should be strings (descriptions)
        for name, desc in tasks.items():
            assert isinstance(name, str)
            assert isinstance(desc, str)
    
    def test_get_default_model_returns_string(self):
        """get_default_model should return a model name string."""
        from hftool.core.registry import get_default_model
        
        model = get_default_model("text-to-image")
        assert isinstance(model, str)
        assert len(model) > 0
    
    def test_task_config_has_required_fields(self):
        """TaskConfig should have all required fields."""
        from hftool.core.registry import TASK_REGISTRY, TaskConfig
        
        for name, config in TASK_REGISTRY.items():
            assert isinstance(config, TaskConfig)
            assert config.handler, f"Task {name} missing handler"
            assert config.library in ("diffusers", "transformers", "custom")
            assert config.input_type in ("text", "image", "audio", "video")
            assert config.output_type in ("text", "image", "audio", "video")
            assert isinstance(config.required_deps, list)
