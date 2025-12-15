"""Tests for core modules: device detection, task registry, models, and download."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

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


class TestModelRegistry:
    """Tests for hftool.core.models module."""
    
    def test_model_registry_not_empty(self):
        """MODEL_REGISTRY should contain models."""
        from hftool.core.models import MODEL_REGISTRY
        
        assert len(MODEL_REGISTRY) > 0
    
    def test_model_registry_has_expected_tasks(self):
        """MODEL_REGISTRY should contain the main tasks."""
        from hftool.core.models import MODEL_REGISTRY
        
        expected_tasks = [
            "text-to-image",
            "text-to-video",
            "text-to-speech",
            "automatic-speech-recognition",
        ]
        
        for task in expected_tasks:
            assert task in MODEL_REGISTRY, f"Missing task in model registry: {task}"
    
    def test_model_info_has_required_fields(self):
        """ModelInfo should have all required fields."""
        from hftool.core.models import MODEL_REGISTRY, ModelInfo
        
        for task_name, models in MODEL_REGISTRY.items():
            for model_name, info in models.items():
                assert isinstance(info, ModelInfo)
                assert info.repo_id, f"Model {model_name} missing repo_id"
                assert info.name, f"Model {model_name} missing name"
                assert info.size_gb > 0, f"Model {model_name} has invalid size"
    
    def test_get_models_for_task_valid(self):
        """get_models_for_task should return models for valid tasks."""
        from hftool.core.models import get_models_for_task
        
        models = get_models_for_task("text-to-image")
        assert isinstance(models, dict)
        assert len(models) > 0
    
    def test_get_models_for_task_alias(self):
        """get_models_for_task should resolve task aliases."""
        from hftool.core.models import get_models_for_task
        
        models_alias = get_models_for_task("t2i")
        models_full = get_models_for_task("text-to-image")
        
        assert models_alias == models_full
    
    def test_get_models_for_task_invalid(self):
        """get_models_for_task should raise ValueError for invalid tasks."""
        from hftool.core.models import get_models_for_task
        
        with pytest.raises(ValueError, match="Unknown task"):
            get_models_for_task("invalid-task-name")
    
    def test_get_model_info_valid(self):
        """get_model_info should return ModelInfo for valid model."""
        from hftool.core.models import get_model_info, ModelInfo
        
        info = get_model_info("text-to-image", "z-image-turbo")
        assert isinstance(info, ModelInfo)
        assert "Z-Image-Turbo" in info.repo_id
    
    def test_get_model_info_by_repo_id(self):
        """get_model_info should work with repo_id."""
        from hftool.core.models import get_model_info, ModelInfo
        
        info = get_model_info("automatic-speech-recognition", "openai/whisper-large-v3")
        assert isinstance(info, ModelInfo)
        assert info.repo_id == "openai/whisper-large-v3"
    
    def test_get_default_model_info(self):
        """get_default_model_info should return the default model."""
        from hftool.core.models import get_default_model_info
        
        info = get_default_model_info("text-to-image")
        assert info.is_default is True
    
    def test_each_task_has_default_model(self):
        """Each task should have exactly one default model."""
        from hftool.core.models import MODEL_REGISTRY
        
        for task_name, models in MODEL_REGISTRY.items():
            defaults = [m for m in models.values() if m.is_default]
            assert len(defaults) >= 1, f"Task {task_name} has no default model"
    
    def test_model_short_name_property(self):
        """ModelInfo.short_name should return last part of repo_id."""
        from hftool.core.models import get_model_info
        
        info = get_model_info("automatic-speech-recognition", "whisper-large-v3")
        assert info.short_name == "whisper-large-v3"
    
    def test_model_size_str_property(self):
        """ModelInfo.size_str should return human-readable size."""
        from hftool.core.models import ModelInfo, ModelType
        
        info = ModelInfo(
            repo_id="test/model",
            name="Test Model",
            model_type=ModelType.TRANSFORMERS,
            size_gb=2.5,
        )
        assert info.size_str == "2.5 GB"
        
        info_small = ModelInfo(
            repo_id="test/small",
            name="Small Model",
            model_type=ModelType.TRANSFORMERS,
            size_gb=0.5,
        )
        assert info_small.size_str == "512 MB"
    
    def test_find_model_by_repo_id(self):
        """find_model_by_repo_id should find models across all tasks."""
        from hftool.core.models import find_model_by_repo_id
        
        result = find_model_by_repo_id("openai/whisper-large-v3")
        assert result is not None
        task, short_name, info = result
        assert task == "automatic-speech-recognition"
        assert info.repo_id == "openai/whisper-large-v3"
    
    def test_find_model_by_repo_id_not_found(self):
        """find_model_by_repo_id should return None for unknown models."""
        from hftool.core.models import find_model_by_repo_id
        
        result = find_model_by_repo_id("nonexistent/model")
        assert result is None


class TestDownloadManager:
    """Tests for hftool.core.download module."""
    
    def test_get_models_dir_default(self):
        """get_models_dir should return ~/.hftool/models/ by default."""
        from hftool.core.download import get_models_dir
        
        with patch.dict(os.environ, {}, clear=True):
            # Clear HFTOOL_MODELS_DIR if set
            os.environ.pop("HFTOOL_MODELS_DIR", None)
            models_dir = get_models_dir()
            assert models_dir == Path.home() / ".hftool" / "models"
    
    def test_get_models_dir_env_var(self):
        """get_models_dir should use HFTOOL_MODELS_DIR if set."""
        from hftool.core.download import get_models_dir
        
        with patch.dict(os.environ, {"HFTOOL_MODELS_DIR": "/custom/path"}):
            models_dir = get_models_dir()
            assert models_dir == Path("/custom/path")
    
    def test_get_model_path(self):
        """get_model_path should return correct path for repo_id."""
        from hftool.core.download import get_model_path
        
        with patch.dict(os.environ, {"HFTOOL_MODELS_DIR": "/models"}):
            path = get_model_path("openai/whisper-large-v3")
            assert path == Path("/models/openai--whisper-large-v3")
    
    def test_is_model_downloaded_false(self):
        """is_model_downloaded should return False for non-existent models."""
        from hftool.core.download import is_model_downloaded
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"HFTOOL_MODELS_DIR": tmpdir}):
                assert is_model_downloaded("nonexistent/model") is False
    
    def test_is_model_downloaded_true(self):
        """is_model_downloaded should return True for existing models."""
        from hftool.core.download import is_model_downloaded, get_model_path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"HFTOOL_MODELS_DIR": tmpdir}):
                # Create a fake model directory
                model_path = get_model_path("test/model")
                model_path.mkdir(parents=True)
                (model_path / "config.json").write_text("{}")
                
                assert is_model_downloaded("test/model") is True
    
    def test_get_download_status(self):
        """get_download_status should return correct status."""
        from hftool.core.download import get_download_status, get_model_path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"HFTOOL_MODELS_DIR": tmpdir}):
                # Not downloaded
                assert get_download_status("test/not-exists") == "not downloaded"
                
                # Downloaded (with config.json)
                model_path = get_model_path("test/complete")
                model_path.mkdir(parents=True)
                (model_path / "config.json").write_text("{}")
                assert get_download_status("test/complete") == "downloaded"
                
                # Partial (directory exists but no config)
                partial_path = get_model_path("test/partial")
                partial_path.mkdir(parents=True)
                (partial_path / "some_file.bin").write_text("data")
                assert get_download_status("test/partial") == "partial"
    
    def test_list_downloaded_models(self):
        """list_downloaded_models should return list of downloaded repo_ids."""
        from hftool.core.download import list_downloaded_models, get_model_path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"HFTOOL_MODELS_DIR": tmpdir}):
                # Create two fake models
                for repo_id in ["test/model1", "test/model2"]:
                    model_path = get_model_path(repo_id)
                    model_path.mkdir(parents=True)
                    (model_path / "config.json").write_text("{}")
                
                downloaded = list_downloaded_models()
                assert "test/model1" in downloaded
                assert "test/model2" in downloaded
    
    def test_delete_model(self):
        """delete_model should remove model directory."""
        from hftool.core.download import delete_model, get_model_path, is_model_downloaded
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"HFTOOL_MODELS_DIR": tmpdir}):
                # Create a fake model
                model_path = get_model_path("test/to-delete")
                model_path.mkdir(parents=True)
                (model_path / "config.json").write_text("{}")
                
                assert is_model_downloaded("test/to-delete") is True
                
                result = delete_model("test/to-delete")
                assert result is True
                assert is_model_downloaded("test/to-delete") is False
    
    def test_delete_model_not_found(self):
        """delete_model should return False for non-existent models."""
        from hftool.core.download import delete_model
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"HFTOOL_MODELS_DIR": tmpdir}):
                result = delete_model("nonexistent/model")
                assert result is False
    
    def test_get_models_disk_usage(self):
        """get_models_disk_usage should return usage info."""
        from hftool.core.download import get_models_disk_usage, get_model_path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"HFTOOL_MODELS_DIR": tmpdir}):
                # Create a fake model with some content
                model_path = get_model_path("test/model")
                model_path.mkdir(parents=True)
                (model_path / "config.json").write_text("{}" * 100)
                (model_path / "model.bin").write_bytes(b"0" * 1024)
                
                usage = get_models_disk_usage()
                assert usage["total_bytes"] > 0
                assert len(usage["models"]) == 1
                assert usage["models"][0]["repo_id"] == "test/model"
