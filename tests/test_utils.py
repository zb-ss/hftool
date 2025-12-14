"""Tests for utility modules."""

import pytest


class TestDependencyChecking:
    """Tests for hftool.utils.deps module."""
    
    def test_is_available_returns_bool(self):
        """is_available should return boolean."""
        from hftool.utils.deps import is_available
        
        # torch should be available
        assert is_available("torch") is True
        
        # non-existent package
        assert is_available("non_existent_package_xyz") is False
    
    def test_check_dependency_raises_on_missing(self):
        """check_dependency should raise DependencyError for missing packages."""
        from hftool.utils.deps import check_dependency, DependencyError
        
        with pytest.raises(DependencyError):
            check_dependency("non_existent_package_xyz")
    
    def test_check_dependency_succeeds_for_available(self):
        """check_dependency should return True for available packages."""
        from hftool.utils.deps import check_dependency
        
        # torch is a core dependency
        assert check_dependency("torch") is True
    
    def test_dependency_error_message_contains_package(self):
        """DependencyError should have helpful message."""
        from hftool.utils.deps import DependencyError
        
        error = DependencyError("mypackage", extra="with_something")
        assert "mypackage" in str(error)
        assert "with_something" in str(error)
    
    def test_is_ffmpeg_available_returns_bool(self):
        """is_ffmpeg_available should return boolean."""
        from hftool.utils.deps import is_ffmpeg_available
        
        result = is_ffmpeg_available()
        assert isinstance(result, bool)
    
    def test_torch_available_constant(self):
        """TORCH_AVAILABLE should be True (torch is a dependency)."""
        from hftool.utils.deps import TORCH_AVAILABLE
        
        assert TORCH_AVAILABLE is True
    
    def test_check_dependencies_multiple(self):
        """check_dependencies should check multiple packages."""
        from hftool.utils.deps import check_dependencies
        
        # These should all be available
        assert check_dependencies(["torch", "click"]) is True
    
    def test_check_dependencies_raises_on_first_missing(self):
        """check_dependencies should raise on first missing package."""
        from hftool.utils.deps import check_dependencies, DependencyError
        
        with pytest.raises(DependencyError):
            check_dependencies(["torch", "non_existent_xyz", "click"])
