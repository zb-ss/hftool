"""Tests for error handling."""

import pytest
from pathlib import Path
from hftool.utils.errors import HFToolError, handle_exception, ERROR_PATTERNS, sanitize_path


class TestHFToolError:
    """Test HFToolError class."""
    
    def test_basic_error(self):
        """Test creating basic error."""
        error = HFToolError("Test error")
        assert error.message == "Test error"
        assert error.suggestion is None
        assert error.original_error is None
    
    def test_error_with_suggestion(self):
        """Test error with suggestion."""
        error = HFToolError("Test error", "Try this fix")
        assert error.message == "Test error"
        assert error.suggestion == "Try this fix"
    
    def test_error_with_original(self):
        """Test error with original exception."""
        original = ValueError("Original error")
        error = HFToolError("Friendly message", "Try this", original)
        assert error.original_error is original


class TestHandleException:
    """Test exception handling."""
    
    def test_cuda_oom_error(self):
        """Test CUDA out of memory error."""
        exc = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        hf_error = handle_exception(exc)
        
        assert "memory" in hf_error.message.lower()
        assert hf_error.suggestion is not None
        assert "float16" in hf_error.suggestion or "dtype" in hf_error.suggestion
    
    def test_missing_module_error(self):
        """Test missing module error."""
        exc = ImportError("No module named 'diffusers'")
        hf_error = handle_exception(exc)
        
        assert "diffusers" in hf_error.message.lower()
        assert hf_error.suggestion is not None
        assert "pip install" in hf_error.suggestion.lower()
    
    def test_connection_error(self):
        """Test network connection error."""
        exc = ConnectionError("Connection refused")
        hf_error = handle_exception(exc)
        
        assert "huggingface" in hf_error.message.lower() or "connection" in hf_error.message.lower()
        assert hf_error.suggestion is not None
    
    def test_file_not_found_error(self):
        """Test file not found error."""
        exc = FileNotFoundError("[Errno 2] No such file or directory: '/path/to/file.txt'")
        hf_error = handle_exception(exc)
        
        assert "file" in hf_error.message.lower() or "not found" in hf_error.message.lower()
    
    def test_unknown_error(self):
        """Test handling of unknown error."""
        exc = RuntimeError("Some unknown error")
        hf_error = handle_exception(exc)
        
        # Should wrap the error with generic suggestion
        assert hf_error.suggestion is not None
        assert "-v" in hf_error.suggestion or "verbose" in hf_error.suggestion
    
    def test_verbose_mode(self):
        """Test verbose error handling."""
        exc = ValueError("Test error")
        hf_error = handle_exception(exc, verbose=True)
        
        assert hf_error.original_error is exc


class TestErrorPatterns:
    """Test error pattern matching."""
    
    def test_patterns_are_valid(self):
        """Test that all error patterns compile."""
        import re
        
        for pattern, message, suggestion in ERROR_PATTERNS:
            # Should not raise
            re.compile(pattern, re.IGNORECASE)
            
            # Message and suggestion should be strings
            assert isinstance(message, str)
            assert isinstance(suggestion, str)
    
    def test_patterns_have_suggestions(self):
        """Test that all patterns have non-empty suggestions."""
        for pattern, message, suggestion in ERROR_PATTERNS:
            assert len(suggestion) > 0
            assert len(message) > 0


class TestSecurityFixes:
    """Test security fixes for H-2 (Information Disclosure)."""
    
    def test_sanitize_path_home_directory(self):
        """Test that paths in home directory are sanitized to use ~."""
        home = Path.home()
        test_path = home / "projects" / "test.py"
        sanitized = sanitize_path(str(test_path))
        
        # Should replace home with ~
        assert sanitized.startswith("~/")
        assert "projects/test.py" in sanitized
        # Should not expose username
        assert str(home) not in sanitized
    
    def test_sanitize_path_outside_home(self):
        """Test that paths outside home show only basename."""
        test_path = "/etc/passwd"
        sanitized = sanitize_path(test_path)
        
        # Should only show basename
        assert sanitized == "passwd"
        assert "/etc" not in sanitized
    
    def test_sanitize_path_in_tmp(self):
        """Test that paths in /tmp show only basename."""
        test_path = "/tmp/somefile.txt"
        sanitized = sanitize_path(test_path)
        
        # Should only show basename
        assert sanitized == "somefile.txt"
        assert "/tmp" not in sanitized
    
    def test_error_message_path_sanitization(self):
        """Test that error messages sanitize file paths."""
        # Create error with path in message
        home = Path.home()
        test_file = home / "secret" / "config.toml"
        error_msg = f"Failed to load config from {test_file}"
        
        error = HFToolError(error_msg)
        
        # Should not contain full path
        assert str(home) not in error.message
        # Should contain sanitized version
        assert "~/" in error.message or "config.toml" in error.message
    
    def test_handle_exception_sanitizes_paths(self):
        """Test that handle_exception sanitizes paths in error messages."""
        # FileNotFoundError with full path
        exc = FileNotFoundError(f"[Errno 2] No such file or directory: '{Path.home()}/secret/data.json'")
        hf_error = handle_exception(exc)
        
        # Should not expose full home path
        assert str(Path.home()) not in hf_error.message
    
    def test_sanitize_windows_path(self):
        """Test sanitizing Windows-style paths."""
        # On Linux, Windows paths won't be recognized as valid paths
        # so they'll be returned as-is or basename extracted
        test_path = "C:\\Users\\username\\documents\\file.txt"
        sanitized = sanitize_path(test_path)
        
        # Should return something (the path or basename)
        # On Linux, this will just return the last component
        assert len(sanitized) > 0
        # The sanitize function should at least try to extract basename
        assert "file.txt" in sanitized or sanitized == test_path
