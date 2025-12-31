"""Error handling and user-friendly error messages for hftool.

Provides pattern matching for common errors and actionable suggestions.
"""

import re
import click
from pathlib import Path
from typing import Optional, Tuple, List


def sanitize_path(path_str: str) -> str:
    """Sanitize file paths to prevent information disclosure.
    
    Security: Replaces home directory with ~ and shows only basename for other paths (H-2).
    
    Args:
        path_str: Path string to sanitize
    
    Returns:
        Sanitized path string
    """
    try:
        path = Path(path_str)
        home = Path.home()
        
        # Try to make relative to home
        try:
            rel_path = path.relative_to(home)
            return f"~/{rel_path}"
        except ValueError:
            # Not in home directory - return basename only
            return path.name if path.name else str(path)
    except Exception:
        # If path parsing fails, return basename or original
        try:
            return Path(path_str).name
        except Exception:
            return path_str


class HFToolError(Exception):
    """Base exception with user-friendly message and suggestion."""
    
    def __init__(self, message: str, suggestion: Optional[str] = None, original_error: Optional[Exception] = None):
        """Initialize HFToolError.
        
        Args:
            message: User-friendly error message
            suggestion: Actionable suggestion to fix the error
            original_error: Original exception that was caught
        """
        # Security: Sanitize paths in message (H-2)
        self.message = self._sanitize_message(message)
        self.suggestion = suggestion
        self.original_error = original_error
        super().__init__(self.message)
    
    @staticmethod
    def _sanitize_message(message: str) -> str:
        """Sanitize file paths in error message.
        
        Args:
            message: Original error message
        
        Returns:
            Sanitized message
        """
        # Replace common absolute path patterns
        # Match paths like /home/username/... or C:\Users\username\...
        path_pattern = r'(?:/[\w.-]+)+(?:/[\w.-]+)*|(?:[A-Z]:\\(?:[\w.-]+\\)*[\w.-]*)'
        
        def replace_path(match):
            return sanitize_path(match.group(0))
        
        return re.sub(path_pattern, replace_path, message)
    
    def display(self, verbose: bool = False) -> None:
        """Display the error to the user.
        
        Args:
            verbose: Whether to show full traceback
        """
        click.echo(click.style(f"Error: {self.message}", fg="red"), err=True)
        
        if self.suggestion:
            click.echo(click.style(f"Suggestion: {self.suggestion}", fg="yellow"), err=True)
        
        if verbose and self.original_error:
            click.echo("\nOriginal error:", err=True)
            click.echo(str(self.original_error), err=True)


# Pattern matching for common errors
# Format: (regex_pattern, message_template, suggestion)
ERROR_PATTERNS: List[Tuple[str, str, str]] = [
    # GPU/CUDA errors
    (
        r"CUDA out of memory|out of memory|OOM",
        "Not enough GPU memory for this operation",
        "Try: --dtype float16, use a smaller model, or set HFTOOL_CPU_OFFLOAD=1"
    ),
    (
        r"CUDA error|HIP error",
        "GPU error occurred during execution",
        "Check GPU status with nvidia-smi or rocm-smi. Try restarting the process."
    ),
    (
        r"No CUDA GPUs are available|CUDA is not available",
        "No GPU detected or CUDA/ROCm not properly configured",
        "Install appropriate PyTorch version: run 'hftool setup' or check https://pytorch.org"
    ),
    
    # Missing dependencies
    (
        r"No module named ['\"](\w+)['\"]",
        "Missing required dependency: {0}",
        "Install with: pip install {0} or pip install hftool[all]"
    ),
    (
        r"ImportError.*(?:diffusers|transformers|accelerate)",
        "Missing required ML library",
        "Install with: pip install hftool[with_t2i] or pip install hftool[all]"
    ),
    (
        r"cannot import name.*from.*diffusers",
        "Incompatible diffusers version",
        "Update diffusers: pip install --upgrade diffusers>=0.36.0"
    ),
    
    # Network errors
    (
        r"Connection refused|ConnectionError|Timeout|URLError",
        "Cannot reach HuggingFace Hub",
        "Check internet connection or use --offline mode with pre-downloaded models"
    ),
    (
        r"401|403|Unauthorized|Forbidden",
        "Access denied to HuggingFace model",
        "Check if model requires authentication: huggingface-cli login"
    ),
    (
        r"404|Not Found",
        "Model or file not found on HuggingFace Hub",
        "Verify model name with: hftool models"
    ),
    
    # File errors
    (
        r"FileNotFoundError.*['\"]([^'\"]+)['\"]",
        "File not found: {0}",
        "Check the file path. Use absolute paths or verify the file exists."
    ),
    (
        r"PermissionError|Permission denied",
        "Cannot write to output location",
        "Check file/directory permissions or use a different output path"
    ),
    (
        r"IsADirectoryError",
        "Output path is a directory",
        "Specify a file name, not a directory: -o /path/to/output.png"
    ),
    
    # Input validation errors
    (
        r"Invalid JSON|JSONDecodeError",
        "Invalid JSON input format",
        "Check JSON syntax or use --interactive mode for guided input"
    ),
    (
        r"Expected.*input.*got",
        "Invalid input type or format",
        "Check input format for this task. Use --help to see examples."
    ),
    
    # Model errors
    (
        r"Model.*not found|Unknown model",
        "Model not found in registry",
        "Run 'hftool models' to see available models"
    ),
    (
        r"cannot load model|Failed to load",
        "Failed to load model",
        "Model may be corrupted. Try: hftool download -t <task> --force"
    ),
    
    # ffmpeg errors
    (
        r"ffmpeg.*not found|Cannot find ffmpeg",
        "ffmpeg is required but not installed",
        "Install ffmpeg: https://ffmpeg.org/download.html"
    ),
    
    # ROCm-specific errors
    (
        r"hipBLASLt|HSA_OVERRIDE_GFX_VERSION",
        "ROCm configuration issue",
        "Add to ~/.hftool/.env: HSA_OVERRIDE_GFX_VERSION=11.0.0 (for RX 7900 XTX)"
    ),
]


def handle_exception(exc: Exception, verbose: bool = False) -> HFToolError:
    """Convert exception to user-friendly error.
    
    Args:
        exc: Original exception
        verbose: Whether to include verbose details
    
    Returns:
        HFToolError with user-friendly message and suggestion
    """
    error_str = str(exc)
    
    # Try to match against known patterns
    for pattern, message_template, suggestion in ERROR_PATTERNS:
        match = re.search(pattern, error_str, re.IGNORECASE)
        if match:
            # Format message with captured groups
            if match.groups():
                try:
                    # Security: Sanitize captured paths (H-2)
                    sanitized_groups = [sanitize_path(g) if '/' in g or '\\' in g else g for g in match.groups()]
                    message = message_template.format(*sanitized_groups)
                    # Also format suggestion if it has placeholders
                    suggestion = suggestion.format(*sanitized_groups)
                except (IndexError, KeyError, ValueError):
                    message = message_template
            else:
                message = message_template
            
            return HFToolError(message, suggestion, exc if verbose else None)
    
    # Unknown error - return generic message
    return HFToolError(
        str(exc),
        "Run with -v for more details or check the documentation",
        exc if verbose else None
    )


def format_error_for_display(error: Exception, verbose: bool = False) -> str:
    """Format an error for display without raising.
    
    Args:
        error: Exception to format
        verbose: Whether to include verbose details
    
    Returns:
        Formatted error string
    """
    if isinstance(error, HFToolError):
        hf_error = error
    else:
        hf_error = handle_exception(error, verbose)
    
    parts = [click.style(f"Error: {hf_error.message}", fg="red")]
    
    if hf_error.suggestion:
        parts.append(click.style(f"Suggestion: {hf_error.suggestion}", fg="yellow"))
    
    if verbose and hf_error.original_error:
        parts.append("\nOriginal error:")
        parts.append(str(hf_error.original_error))
    
    return "\n".join(parts)
