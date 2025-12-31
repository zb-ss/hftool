"""Tests for configuration management."""

import os
import tempfile
from pathlib import Path
import pytest

from hftool.core.config import Config


class TestConfig:
    """Test configuration loading and management."""
    
    def setup_method(self):
        """Reset singleton before each test."""
        Config.reset()
    
    def test_singleton_pattern(self):
        """Test that Config is a singleton."""
        config1 = Config.get()
        config2 = Config.get()
        assert config1 is config2
    
    def test_empty_config(self):
        """Test that config works without any config files."""
        config = Config.get()
        assert config.get_value("nonexistent", default="test") == "test"
    
    def test_get_value_with_default(self):
        """Test getting value with default."""
        config = Config.get()
        assert config.get_value("test_key", default="default_value") == "default_value"
    
    def test_env_var_override(self):
        """Test that environment variables override config."""
        os.environ["HFTOOL_TEST_VAR"] = "env_value"
        try:
            config = Config.get()
            assert config.get_value("test_var", default="default") == "env_value"
        finally:
            del os.environ["HFTOOL_TEST_VAR"]
    
    def test_parse_env_bool(self):
        """Test parsing boolean environment variables."""
        config = Config()
        
        assert config._parse_env_value("true") is True
        assert config._parse_env_value("True") is True
        assert config._parse_env_value("1") is True
        assert config._parse_env_value("yes") is True
        
        assert config._parse_env_value("false") is False
        assert config._parse_env_value("False") is False
        assert config._parse_env_value("0") is False
        assert config._parse_env_value("no") is False
    
    def test_parse_env_int(self):
        """Test parsing integer environment variables."""
        config = Config()
        assert config._parse_env_value("42") == 42
        assert config._parse_env_value("-10") == -10
    
    def test_parse_env_float(self):
        """Test parsing float environment variables."""
        config = Config()
        assert config._parse_env_value("3.14") == 3.14
        assert config._parse_env_value("-2.5") == -2.5
    
    def test_parse_env_string(self):
        """Test parsing string environment variables."""
        config = Config()
        assert config._parse_env_value("hello") == "hello"
        assert config._parse_env_value("hello world") == "hello world"
    
    def test_resolve_model_alias(self):
        """Test model alias resolution."""
        config = Config()
        config._config = {
            "aliases": {
                "fast": "model/fast",
                "slow": "model/slow",
            }
        }
        
        assert config.resolve_model_alias("fast") == "model/fast"
        assert config.resolve_model_alias("slow") == "model/slow"
        assert config.resolve_model_alias("unknown") == "unknown"
    
    def test_get_path(self):
        """Test path expansion."""
        config = Config()
        config._config = {
            "defaults": {
                "test_path": "~/test",
            }
        }
        
        path = config.get_path("test_path")
        assert path == Path.home() / "test"
    
    def test_config_file_loading(self):
        """Test loading config from TOML file."""
        # Only run if tomllib/tomli is available
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                pytest.skip("tomllib/tomli not available")
        
        # Create a temporary config file
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".hftool"
            config_dir.mkdir()
            config_file = config_dir / "config.toml"
            
            config_content = """
[defaults]
device = "cuda"
verbose = true

[text-to-image]
model = "test-model"
num_inference_steps = 20
"""
            config_file.write_text(config_content)
            
            # Temporarily change HOME to tmpdir
            old_home = os.environ.get("HOME")
            os.environ["HOME"] = tmpdir
            
            try:
                Config.reset()
                config = Config.get()
                
                # Check that values were loaded
                assert config.get_value("device") == "cuda"
                assert config.get_value("verbose") is True
                assert config.get_value("model", task="text-to-image") == "test-model"
                assert config.get_value("num_inference_steps", task="text-to-image") == 20
            finally:
                if old_home:
                    os.environ["HOME"] = old_home
                else:
                    del os.environ["HOME"]
    
    def test_has_config_file(self):
        """Test checking if config file was loaded."""
        config = Config.get()
        # Without config files, should return False
        # (unless one exists in the real home directory)
        assert isinstance(config.has_config_file(), bool)
    
    def test_get_config_paths(self):
        """Test getting config file paths."""
        config = Config.get()
        paths = config.get_config_paths()
        
        assert "user" in paths
        assert "project" in paths
        # Paths should be None or Path objects
        assert paths["user"] is None or isinstance(paths["user"], Path)
        assert paths["project"] is None or isinstance(paths["project"], Path)


class TestConfigSecurity:
    """Test security fixes for configuration."""
    
    def setup_method(self):
        """Reset singleton before each test."""
        Config.reset()
    
    def test_path_traversal_prevention(self):
        """Test that get_path() prevents path traversal (H-1)."""
        config = Config()
        config._config = {
            "defaults": {
                "malicious_path": "/etc/passwd",
            }
        }
        
        # Should raise ValueError for path outside home/tmp
        with pytest.raises(ValueError, match="outside allowed directories"):
            config.get_path("malicious_path")
    
    def test_path_in_home_allowed(self):
        """Test that paths within home directory are allowed."""
        config = Config()
        config._config = {
            "defaults": {
                "safe_path": "~/projects/test",
            }
        }
        
        # Should work fine
        path = config.get_path("safe_path")
        assert path is not None
        assert path.is_relative_to(Path.home())
    
    def test_path_in_tmp_allowed(self):
        """Test that paths in /tmp are allowed."""
        config = Config()
        config._config = {
            "defaults": {
                "tmp_path": "/tmp/test.txt",
            }
        }
        
        # Should work fine
        path = config.get_path("tmp_path")
        assert path is not None
        assert "/tmp" in str(path)
    
    def test_env_var_validation(self):
        """Test environment variable name validation (M-2)."""
        config = Config()
        
        # Valid HFTOOL_ prefix
        os.environ["HFTOOL_VALID_VAR"] = "test_value"
        try:
            value = config.get_value("valid_var", env_var="HFTOOL_VALID_VAR")
            assert value == "test_value"
        finally:
            del os.environ["HFTOOL_VALID_VAR"]
        
        # Invalid env var name (no HFTOOL_ prefix)
        os.environ["MALICIOUS_VAR"] = "malicious_value"
        try:
            # Should not use the malicious env var
            value = config.get_value("test", default="default", env_var="MALICIOUS_VAR")
            # Should fall back to default since env var is rejected
            assert value == "default"
        finally:
            del os.environ["MALICIOUS_VAR"]
    
    def test_config_file_size_limit(self):
        """Test config file size limit (M-1)."""
        # Only run if tomllib/tomli is available
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                pytest.skip("tomllib/tomli not available")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".hftool"
            config_dir.mkdir()
            config_file = config_dir / "config.toml"
            
            # Create a large config file (>1MB)
            large_content = "[defaults]\n" + "x" * (1024 * 1024 + 1000)
            config_file.write_text(large_content)
            
            # Temporarily change HOME to tmpdir
            old_home = os.environ.get("HOME")
            os.environ["HOME"] = tmpdir
            
            try:
                Config.reset()
                config = Config.get()
                
                # Config should still initialize but large file should be rejected
                assert config is not None
                # The large config shouldn't be loaded
                assert not config.has_config_file()
            finally:
                if old_home:
                    os.environ["HOME"] = old_home
                else:
                    del os.environ["HOME"]
    
    def test_env_var_injection_special_chars(self):
        """Test that env var names with special characters are rejected."""
        config = Config()
        
        # Env var with special characters
        malicious_vars = [
            "HFTOOL_TEST;rm -rf /",
            "HFTOOL_TEST`whoami`",
            "HFTOOL_TEST$(malicious)",
            "HFTOOL_TEST/../../../etc/passwd",
        ]
        
        for var_name in malicious_vars:
            # These should be rejected by the regex validation
            value = config.get_value("test", default="safe", env_var=var_name)
            # Should return default since env var is rejected
            assert value == "safe"
