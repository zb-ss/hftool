"""Optional dependency checking for hftool."""

import importlib
import sys
from typing import List, Optional, Dict


class DependencyError(ImportError):
    """Raised when a required dependency is missing."""
    
    def __init__(self, package: str, extra: Optional[str] = None, pip_name: Optional[str] = None):
        self.package = package
        self.extra = extra
        self.pip_name = pip_name or package
        
        if extra:
            message = (
                f"'{package}' is required for this task. "
                f"Install with: pip install hftool[{extra}]"
            )
        else:
            message = (
                f"'{package}' is required for this task. "
                f"Install with: pip install {self.pip_name}"
            )
        super().__init__(message)


# Cache for checked dependencies
_DEPENDENCY_CACHE: Dict[str, bool] = {}


def check_dependency(package: str, extra: Optional[str] = None, pip_name: Optional[str] = None) -> bool:
    """Check if a package is available.
    
    Args:
        package: The Python package name to import
        extra: The hftool extra that provides this dependency
        pip_name: The pip package name if different from import name
    
    Returns:
        True if the package is available
    
    Raises:
        DependencyError: If the package is not available
    """
    if package in _DEPENDENCY_CACHE:
        if not _DEPENDENCY_CACHE[package]:
            raise DependencyError(package, extra, pip_name)
        return True
    
    try:
        importlib.import_module(package)
        _DEPENDENCY_CACHE[package] = True
        return True
    except ImportError:
        _DEPENDENCY_CACHE[package] = False
        raise DependencyError(package, extra, pip_name)


def check_dependencies(packages: List[str], extra: Optional[str] = None) -> bool:
    """Check if multiple packages are available.
    
    Args:
        packages: List of package names to check
        extra: The hftool extra that provides these dependencies
    
    Returns:
        True if all packages are available
    
    Raises:
        DependencyError: If any package is not available
    """
    for package in packages:
        check_dependency(package, extra)
    return True


def is_available(package: str) -> bool:
    """Check if a package is available without raising an error.
    
    Args:
        package: The Python package name to import
    
    Returns:
        True if the package is available, False otherwise
    """
    if package in _DEPENDENCY_CACHE:
        return _DEPENDENCY_CACHE[package]
    
    try:
        importlib.import_module(package)
        _DEPENDENCY_CACHE[package] = True
        return True
    except ImportError:
        _DEPENDENCY_CACHE[package] = False
        return False


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available on the system.
    
    Returns:
        True if ffmpeg is available
    
    Raises:
        DependencyError: If ffmpeg is not available
    """
    import shutil
    if shutil.which("ffmpeg") is None:
        raise DependencyError(
            "ffmpeg",
            extra=None,
            pip_name="ffmpeg (system package, not pip)"
        )
    return True


def is_ffmpeg_available() -> bool:
    """Check if ffmpeg is available without raising an error.
    
    Returns:
        True if ffmpeg is available, False otherwise
    """
    import shutil
    return shutil.which("ffmpeg") is not None


# Pre-check common optional dependencies
TORCH_AVAILABLE = is_available("torch")
TRANSFORMERS_AVAILABLE = is_available("transformers")
DIFFUSERS_AVAILABLE = is_available("diffusers")
PIL_AVAILABLE = is_available("PIL")
SOUNDFILE_AVAILABLE = is_available("soundfile")
ACCELERATE_AVAILABLE = is_available("accelerate")
