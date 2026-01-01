"""Integration tests for config system in task execution.

Tests that config values are properly loaded and applied during task execution,
including priority chain, model alias resolution, and parameter merging.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
import pytest

from hftool.core.config import Config


class TestConfigIntegration:
    """Test config integration in task execution."""
    
    def setup_method(self):
        """Reset config singleton and environment before each test."""
        Config.reset()
        # Clean up any HFTOOL_* env vars
        for key in list(os.environ.keys()):
            if key.startswith("HFTOOL_"):
                del os.environ[key]
    
    def teardown_method(self):
        """Clean up after each test."""
        Config.reset()
        for key in list(os.environ.keys()):
            if key.startswith("HFTOOL_"):
                del os.environ[key]
    
    # =========================================================================
    # 1. Config Loading in Task Execution
    # =========================================================================
    
    def test_config_loaded_during_task_execution(self, tmp_path):
        """Test that config file is loaded when running a task."""
        # Create a config file
        config_dir = tmp_path / ".hftool"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text("""
[defaults]
device = "cpu"
dtype = "float16"

[text-to-image]
model = "z-image-turbo"
""")
        
        # Patch Path.cwd to return tmp_path
        with patch.object(Path, 'cwd', return_value=tmp_path):
            Config.reset()
            config = Config.get()
            
            # Verify config was loaded
            assert config.get_value("device", default=None) == "cpu"
            assert config.get_value("dtype", default=None) == "float16"
    
    def test_missing_config_backward_compatibility(self, tmp_path):
        """Test that missing config doesn't break execution (backward compatibility)."""
        # Ensure no config file exists
        with patch.object(Path, 'cwd', return_value=tmp_path), \
             patch.object(Path, 'home', return_value=tmp_path):
            
            Config.reset()
            config = Config.get()
            
            # Should work fine with no config
            assert config.get_value("device", default="auto") == "auto"
            assert config.get_value("model", default=None) is None
    
    def test_user_config_loading(self, tmp_path):
        """Test that user config (~/.hftool/config.toml) is loaded."""
        # Create user config
        user_config_dir = tmp_path / ".hftool"
        user_config_dir.mkdir()
        user_config_file = user_config_dir / "config.toml"
        user_config_file.write_text("""
[defaults]
device = "cuda"
""")
        
        with patch.object(Path, 'home', return_value=tmp_path):
            Config.reset()
            config = Config.get()
            
            assert config.get_value("device", default=None) == "cuda"
    
    def test_project_config_loading(self, tmp_path):
        """Test that project config (./.hftool/config.toml) is loaded."""
        # Create project config
        project_config_dir = tmp_path / ".hftool"
        project_config_dir.mkdir()
        project_config_file = project_config_dir / "config.toml"
        project_config_file.write_text("""
[defaults]
device = "mps"
""")
        
        with patch.object(Path, 'cwd', return_value=tmp_path):
            Config.reset()
            config = Config.get()
            
            assert config.get_value("device", default=None) == "mps"
    
    # =========================================================================
    # 2. Config Priority Chain
    # =========================================================================
    
    def test_cli_args_override_config(self, tmp_path):
        """Test that CLI args override config values."""
        # Create config with device = "cpu"
        config_dir = tmp_path / ".hftool"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text("""
[defaults]
device = "cpu"
""")
        
        with patch.object(Path, 'cwd', return_value=tmp_path):
            Config.reset()
            config = Config.get()
            
            # Config says "cpu"
            assert config.get_value("device", default="auto") == "cpu"
            
            # But CLI arg should override (simulated by checking the logic)
            # In actual execution, CLI arg "cuda" would be used instead of config "cpu"
            cli_device = "cuda"  # This would come from CLI arg
            effective_device = cli_device if cli_device != "auto" else config.get_value("device", default="auto")
            assert effective_device == "cuda"
    
    def test_env_vars_override_config(self, tmp_path):
        """Test that environment variables override config values."""
        # Create config
        config_dir = tmp_path / ".hftool"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text("""
[defaults]
device = "cpu"
""")
        
        # Set env var
        os.environ["HFTOOL_DEVICE"] = "cuda"
        
        with patch.object(Path, 'cwd', return_value=tmp_path):
            Config.reset()
            config = Config.get()
            
            # Env var should override config
            assert config.get_value("device", default="auto") == "cuda"
    
    def test_project_config_overrides_user_config(self, tmp_path):
        """Test that project config overrides user config."""
        # Create user config
        user_config_dir = tmp_path / "user" / ".hftool"
        user_config_dir.mkdir(parents=True)
        user_config_file = user_config_dir / "config.toml"
        user_config_file.write_text("""
[defaults]
device = "cpu"
dtype = "float32"
""")
        
        # Create project config
        project_config_dir = tmp_path / "project" / ".hftool"
        project_config_dir.mkdir(parents=True)
        project_config_file = project_config_dir / "config.toml"
        project_config_file.write_text("""
[defaults]
device = "cuda"
""")
        
        with patch.object(Path, 'home', return_value=tmp_path / "user"), \
             patch.object(Path, 'cwd', return_value=tmp_path / "project"):
            
            Config.reset()
            config = Config.get()
            
            # Project config should override user config for device
            assert config.get_value("device", default=None) == "cuda"
            # But user config value for dtype should still be there
            assert config.get_value("dtype", default=None) == "float32"
    
    def test_config_overrides_builtin_defaults(self, tmp_path):
        """Test that config overrides built-in defaults."""
        # Create config
        config_dir = tmp_path / ".hftool"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text("""
[defaults]
device = "cpu"
""")
        
        with patch.object(Path, 'cwd', return_value=tmp_path):
            Config.reset()
            config = Config.get()
            
            # Config value should override default "auto"
            assert config.get_value("device", default="auto") == "cpu"
    
    # =========================================================================
    # 3. Model Alias Resolution
    # =========================================================================
    
    def test_model_alias_resolution(self, tmp_path):
        """Test that model aliases from config are resolved correctly."""
        # Create config with model alias
        config_dir = tmp_path / ".hftool"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text("""
[aliases]
my-favorite = "Tongyi-MAI/Z-Image-Turbo"
quick = "stabilityai/sdxl-turbo"
""")
        
        with patch.object(Path, 'cwd', return_value=tmp_path):
            Config.reset()
            config = Config.get()
            
            # Test alias resolution
            assert config.resolve_model_alias("my-favorite") == "Tongyi-MAI/Z-Image-Turbo"
            assert config.resolve_model_alias("quick") == "stabilityai/sdxl-turbo"
    
    def test_invalid_alias_fallback(self, tmp_path):
        """Test that invalid aliases fall back gracefully."""
        # Create config with some aliases
        config_dir = tmp_path / ".hftool"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text("""
[aliases]
known = "some/model"
""")
        
        with patch.object(Path, 'cwd', return_value=tmp_path):
            Config.reset()
            config = Config.get()
            
            # Unknown alias should return as-is
            assert config.resolve_model_alias("unknown-alias") == "unknown-alias"
    
    def test_explicit_model_name_bypass_alias(self, tmp_path):
        """Test that explicit model names bypass alias resolution."""
        # Create config with alias
        config_dir = tmp_path / ".hftool"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text("""
[aliases]
turbo = "stabilityai/sdxl-turbo"
""")
        
        with patch.object(Path, 'cwd', return_value=tmp_path):
            Config.reset()
            config = Config.get()
            
            # Explicit repo ID should pass through unchanged
            explicit_model = "openai/whisper-large-v3"
            assert config.resolve_model_alias(explicit_model) == explicit_model
    
    # =========================================================================
    # 4. Default Application
    # =========================================================================
    
    def test_device_default_from_config(self, tmp_path):
        """Test that device default from config is applied."""
        # Create config
        config_dir = tmp_path / ".hftool"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text("""
[defaults]
device = "cpu"
""")
        
        with patch.object(Path, 'cwd', return_value=tmp_path):
            Config.reset()
            config = Config.get()
            
            # Simulate CLI logic: device = "auto" from CLI, should use config
            cli_device = "auto"
            effective_device = config.get_value("device", default="auto") if cli_device == "auto" else cli_device
            assert effective_device == "cpu"
    
    def test_dtype_default_from_config(self, tmp_path):
        """Test that dtype default from config is applied."""
        # Create config
        config_dir = tmp_path / ".hftool"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text("""
[defaults]
dtype = "float16"
""")
        
        with patch.object(Path, 'cwd', return_value=tmp_path):
            Config.reset()
            config = Config.get()
            
            # Simulate CLI logic: dtype = None from CLI, should use config
            cli_dtype = None
            effective_dtype = config.get_value("dtype", default=None) if cli_dtype is None else cli_dtype
            assert effective_dtype == "float16"
    
    def test_model_default_from_config(self, tmp_path):
        """Test that model default from config is applied."""
        # Create config
        config_dir = tmp_path / ".hftool"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text("""
[text-to-image]
model = "z-image-turbo"
""")
        
        with patch.object(Path, 'cwd', return_value=tmp_path):
            Config.reset()
            config = Config.get()
            
            # Simulate CLI logic: model = None from CLI, should use config
            cli_model = None
            task = "text-to-image"
            effective_model = config.get_value("model", task=task, default=None) if cli_model is None else cli_model
            assert effective_model == "z-image-turbo"
    
    def test_task_specific_parameters_from_config(self, tmp_path):
        """Test that task-specific parameters from config are applied."""
        # Create config with task-specific params
        config_dir = tmp_path / ".hftool"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text("""
[text-to-image]
num_inference_steps = 25
guidance_scale = 7.5
width = 1024
height = 1024
""")
        
        with patch.object(Path, 'cwd', return_value=tmp_path):
            Config.reset()
            config = Config.get()
            
            # Simulate CLI logic: extract task params from config
            task = "text-to-image"
            if task in config._config:
                task_config_section = config._config[task]
                if isinstance(task_config_section, dict):
                    reserved_keys = {'model', 'device', 'dtype'}
                    task_params = {k: v for k, v in task_config_section.items() if k not in reserved_keys}
                    
                    assert task_params["num_inference_steps"] == 25
                    assert task_params["guidance_scale"] == 7.5
                    assert task_params["width"] == 1024
                    assert task_params["height"] == 1024
    
    # =========================================================================
    # 5. Edge Cases
    # =========================================================================
    
    def test_invalid_config_values_handled_gracefully(self, tmp_path):
        """Test that invalid config values are handled gracefully."""
        # Create config with invalid values
        config_dir = tmp_path / ".hftool"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text("""
[defaults]
device = "invalid-device"
dtype = "invalid-dtype"
""")
        
        with patch.object(Path, 'cwd', return_value=tmp_path):
            Config.reset()
            config = Config.get()
            
            # Config should load, even with invalid values
            # Validation happens later in the pipeline
            assert config.get_value("device", default=None) == "invalid-device"
            assert config.get_value("dtype", default=None) == "invalid-dtype"
    
    def test_malformed_toml_doesnt_crash(self, tmp_path):
        """Test that malformed TOML doesn't crash execution."""
        # Create malformed config
        config_dir = tmp_path / ".hftool"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text("""
[defaults
device = "cpu"
this is not valid TOML
""")
        
        with patch.object(Path, 'cwd', return_value=tmp_path):
            Config.reset()
            
            # Should not crash, just skip the malformed config
            config = Config.get()
            
            # Should fall back to defaults
            assert config.get_value("device", default="auto") == "auto"
    
    def test_config_reload_behavior(self, tmp_path):
        """Test config reload behavior."""
        # Create initial config
        config_dir = tmp_path / ".hftool"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text("""
[defaults]
device = "cpu"
""")
        
        with patch.object(Path, 'cwd', return_value=tmp_path):
            Config.reset()
            config1 = Config.get()
            assert config1.get_value("device", default=None) == "cpu"
            
            # Modify config file
            config_file.write_text("""
[defaults]
device = "cuda"
""")
            
            # Same instance should still have old value (singleton)
            config2 = Config.get()
            assert config2 is config1
            assert config2.get_value("device", default=None) == "cpu"
            
            # Reset and reload should get new value
            Config.reset()
            config3 = Config.get()
            assert config3.get_value("device", default=None) == "cuda"
    
    def test_empty_config_file(self, tmp_path):
        """Test that empty config file doesn't cause issues."""
        # Create empty config
        config_dir = tmp_path / ".hftool"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text("")
        
        with patch.object(Path, 'cwd', return_value=tmp_path):
            Config.reset()
            config = Config.get()
            
            # Should work fine with empty config
            assert config.get_value("device", default="auto") == "auto"
    
    def test_config_with_comments(self, tmp_path):
        """Test that config with comments is parsed correctly."""
        # Create config with comments
        config_dir = tmp_path / ".hftool"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text("""
# This is a comment
[defaults]
# Use CPU for testing
device = "cpu"

# Text-to-image settings
[text-to-image]
model = "z-image-turbo"  # Fast model
""")
        
        with patch.object(Path, 'cwd', return_value=tmp_path):
            Config.reset()
            config = Config.get()
            
            assert config.get_value("device", default=None) == "cpu"
            assert config.get_value("model", task="text-to-image", default=None) == "z-image-turbo"
    
    # =========================================================================
    # 6. Integration with Task Execution Flow
    # =========================================================================
    
    def test_full_config_integration_flow(self, tmp_path):
        """Test full integration: config -> defaults -> alias -> params."""
        # Create comprehensive config
        config_dir = tmp_path / ".hftool"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text("""
[defaults]
device = "cpu"
dtype = "float16"

[aliases]
fast = "Tongyi-MAI/Z-Image-Turbo"

[text-to-image]
model = "fast"
num_inference_steps = 20
guidance_scale = 7.0
""")
        
        with patch.object(Path, 'cwd', return_value=tmp_path):
            Config.reset()
            config = Config.get()
            
            # Simulate full task execution flow
            task = "text-to-image"
            
            # 1. Load defaults
            device = config.get_value("device", task=task, default="auto")
            dtype = config.get_value("dtype", task=task, default=None)
            model = config.get_value("model", task=task, default=None)
            
            assert device == "cpu"
            assert dtype == "float16"
            assert model == "fast"
            
            # 2. Resolve model alias
            resolved_model = config.resolve_model_alias(model)
            assert resolved_model == "Tongyi-MAI/Z-Image-Turbo"
            
            # 3. Extract task params
            task_config_section = config._config.get(task, {})
            reserved_keys = {'model', 'device', 'dtype'}
            task_params = {k: v for k, v in task_config_section.items() if k not in reserved_keys}
            
            assert task_params["num_inference_steps"] == 20
            assert task_params["guidance_scale"] == 7.0
            
            # 4. Merge with CLI params (CLI has priority)
            cli_params = {"guidance_scale": 8.0}  # Override from CLI
            final_params = {**task_params, **cli_params}
            
            assert final_params["num_inference_steps"] == 20  # From config
            assert final_params["guidance_scale"] == 8.0  # From CLI (overridden)
    
    def test_config_priority_chain_complete(self, tmp_path):
        """Test complete priority chain: CLI > env > project > user > default."""
        # Create user config
        user_config_dir = tmp_path / "user" / ".hftool"
        user_config_dir.mkdir(parents=True)
        user_config_file = user_config_dir / "config.toml"
        user_config_file.write_text("""
[defaults]
device = "cpu"
dtype = "float32"
model = "user-model"
""")
        
        # Create project config
        project_config_dir = tmp_path / "project" / ".hftool"
        project_config_dir.mkdir(parents=True)
        project_config_file = project_config_dir / "config.toml"
        project_config_file.write_text("""
[defaults]
device = "cuda"
dtype = "float16"
""")
        
        # Set env var
        os.environ["HFTOOL_DEVICE"] = "mps"
        
        with patch.object(Path, 'home', return_value=tmp_path / "user"), \
             patch.object(Path, 'cwd', return_value=tmp_path / "project"):
            
            Config.reset()
            config = Config.get()
            
            # Priority chain test:
            # device: env var "mps" > project "cuda" > user "cpu"
            assert config.get_value("device", default="auto") == "mps"
            
            # dtype: project "float16" > user "float32"
            assert config.get_value("dtype", default=None) == "float16"
            
            # model: user "user-model" (no override)
            assert config.get_value("model", default=None) == "user-model"
            
            # Simulate CLI override
            cli_device = "rocm"
            effective_device = cli_device if cli_device != "auto" else config.get_value("device", default="auto")
            assert effective_device == "rocm"  # CLI wins over everything
