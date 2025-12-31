"""Tests for shell completion utilities."""

import os
import tempfile
from pathlib import Path
import pytest

from hftool.core.completion import (
    get_task_names,
    get_model_names,
    get_device_options,
    get_dtype_options,
    get_shell_name,
    get_completion_script,
    install_completion,
    TaskCompleter,
    ModelCompleter,
    DeviceCompleter,
    DtypeCompleter,
    FilePickerCompleter,
)


class TestCompletionHelpers:
    """Test completion helper functions."""
    
    def test_get_task_names(self):
        """Test getting task names for completion."""
        tasks = get_task_names()
        
        assert isinstance(tasks, list)
        assert len(tasks) > 0
        assert "text-to-image" in tasks
        assert "t2i" in tasks  # Alias
        assert sorted(tasks) == tasks  # Should be sorted
    
    def test_get_model_names_all(self):
        """Test getting all model names."""
        models = get_model_names()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert "z-image-turbo" in models
        assert "whisper-large-v3" in models
    
    def test_get_model_names_filtered(self):
        """Test getting model names for specific task."""
        models = get_model_names("text-to-image")
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert "z-image-turbo" in models
        
        # Should not contain models from other tasks
        assert "whisper-large-v3" not in models
    
    def test_get_model_names_with_alias(self):
        """Test getting model names with task alias."""
        models = get_model_names("t2i")
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert "z-image-turbo" in models
    
    def test_get_device_options(self):
        """Test getting device options."""
        devices = get_device_options()
        
        assert isinstance(devices, list)
        assert "auto" in devices
        assert "cuda" in devices
        assert "mps" in devices
        assert "cpu" in devices
    
    def test_get_dtype_options(self):
        """Test getting dtype options."""
        dtypes = get_dtype_options()
        
        assert isinstance(dtypes, list)
        assert "float32" in dtypes
        assert "float16" in dtypes
        assert "bfloat16" in dtypes


class TestShellDetection:
    """Test shell detection and script generation."""
    
    def test_get_shell_name_from_env(self):
        """Test shell detection from SHELL env var."""
        old_shell = os.environ.get("SHELL")
        
        try:
            os.environ["SHELL"] = "/bin/bash"
            assert get_shell_name() == "bash"
            
            os.environ["SHELL"] = "/usr/bin/zsh"
            assert get_shell_name() == "zsh"
            
            os.environ["SHELL"] = "/usr/bin/fish"
            assert get_shell_name() == "fish"
        finally:
            if old_shell:
                os.environ["SHELL"] = old_shell
            elif "SHELL" in os.environ:
                del os.environ["SHELL"]
    
    def test_get_completion_script_bash(self):
        """Test bash completion script generation."""
        script = get_completion_script("bash")
        
        assert "bash" in script
        assert "_HFTOOL_COMPLETE=bash_source" in script
        assert "hftool" in script
    
    def test_get_completion_script_zsh(self):
        """Test zsh completion script generation."""
        script = get_completion_script("zsh")
        
        assert "zsh" in script
        assert "_HFTOOL_COMPLETE=zsh_source" in script
        assert "hftool" in script
    
    def test_get_completion_script_fish(self):
        """Test fish completion script generation."""
        script = get_completion_script("fish")
        
        assert "fish" in script
        assert "_HFTOOL_COMPLETE=fish_source" in script
        assert "hftool" in script
    
    def test_get_completion_script_invalid(self):
        """Test error handling for invalid shell."""
        with pytest.raises(ValueError, match="Unsupported shell"):
            get_completion_script("tcsh")


class TestCompletionInstall:
    """Test completion installation."""
    
    def test_install_completion_bash(self):
        """Test bash completion installation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / ".bashrc"
            
            # Mock home directory
            original_home = Path.home
            Path.home = lambda: Path(tmpdir)
            
            try:
                # Install completion
                result = install_completion("bash")
                
                assert result is True
                assert config_file.exists()
                
                content = config_file.read_text()
                assert "# hftool completion for bash" in content
                assert "_HFTOOL_COMPLETE=bash_source" in content
                
                # Try installing again - should return False (already installed)
                result = install_completion("bash")
                assert result is False
            finally:
                Path.home = original_home
    
    def test_install_completion_zsh(self):
        """Test zsh completion installation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / ".zshrc"
            
            original_home = Path.home
            Path.home = lambda: Path(tmpdir)
            
            try:
                result = install_completion("zsh")
                
                assert result is True
                assert config_file.exists()
                
                content = config_file.read_text()
                assert "# hftool completion for zsh" in content
            finally:
                Path.home = original_home
    
    def test_install_completion_fish(self):
        """Test fish completion installation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".config" / "fish"
            config_file = config_dir / "config.fish"
            
            original_home = Path.home
            Path.home = lambda: Path(tmpdir)
            
            try:
                result = install_completion("fish")
                
                assert result is True
                assert config_file.exists()
                
                content = config_file.read_text()
                assert "# hftool completion for fish" in content
            finally:
                Path.home = original_home
    
    def test_install_completion_invalid(self):
        """Test error handling for invalid shell."""
        with pytest.raises(ValueError, match="Unsupported shell"):
            install_completion("tcsh")


def _get_completion_values(items):
    """Extract string values from CompletionItem objects."""
    return [item.value for item in items]


class TestCompleters:
    """Test Click completer classes."""
    
    def test_task_completer(self):
        """Test TaskCompleter."""
        completer = TaskCompleter()
        
        # Mock context
        class MockContext:
            params = {}
        
        class MockParam:
            pass
        
        ctx = MockContext()
        param = MockParam()
        
        # Test completion - completers now return CompletionItem objects
        matches = _get_completion_values(completer(ctx, param, "text"))
        assert "text-to-image" in matches
        
        matches = _get_completion_values(completer(ctx, param, "t2"))
        assert "t2i" in matches
        
        matches = _get_completion_values(completer(ctx, param, "xyz"))
        assert len(matches) == 0
    
    def test_model_completer(self):
        """Test ModelCompleter."""
        completer = ModelCompleter()
        
        class MockContext:
            params = {"task": "text-to-image"}
        
        class MockParam:
            pass
        
        ctx = MockContext()
        param = MockParam()
        
        matches = _get_completion_values(completer(ctx, param, "z-"))
        assert "z-image-turbo" in matches
        
        matches = _get_completion_values(completer(ctx, param, "whisper"))
        assert len(matches) == 0  # Not in text-to-image task
    
    def test_model_completer_no_task(self):
        """Test ModelCompleter without task context."""
        completer = ModelCompleter()
        
        class MockContext:
            params = {}
        
        class MockParam:
            pass
        
        ctx = MockContext()
        param = MockParam()
        
        # Should return all models
        matches = _get_completion_values(completer(ctx, param, "whisper"))
        assert "whisper-large-v3" in matches
    
    def test_device_completer(self):
        """Test DeviceCompleter."""
        completer = DeviceCompleter()
        
        class MockContext:
            params = {}
        
        class MockParam:
            pass
        
        ctx = MockContext()
        param = MockParam()
        
        matches = _get_completion_values(completer(ctx, param, "cu"))
        assert "cuda" in matches
        assert "cuda:0" in matches
        
        matches = _get_completion_values(completer(ctx, param, "m"))
        assert "mps" in matches
    
    def test_dtype_completer(self):
        """Test DtypeCompleter."""
        completer = DtypeCompleter()
        
        class MockContext:
            params = {}
        
        class MockParam:
            pass
        
        ctx = MockContext()
        param = MockParam()
        
        matches = _get_completion_values(completer(ctx, param, "float"))
        assert "float32" in matches
        assert "float16" in matches
        
        matches = _get_completion_values(completer(ctx, param, "bf"))
        assert "bfloat16" in matches
    
    def test_file_picker_completer(self):
        """Test FilePickerCompleter."""
        completer = FilePickerCompleter()
        
        class MockContext:
            params = {}
        
        class MockParam:
            pass
        
        ctx = MockContext()
        param = MockParam()
        
        # Test @ syntax completion
        matches = _get_completion_values(completer(ctx, param, "@"))
        assert "@" in matches
        assert "@?" in matches
        assert "@." in matches
        assert "@~" in matches
        assert "@@" in matches
        
        matches = _get_completion_values(completer(ctx, param, "@?"))
        assert "@?" in matches
        
        # Non-@ input should return empty
        matches = _get_completion_values(completer(ctx, param, "file"))
        assert len(matches) == 0
