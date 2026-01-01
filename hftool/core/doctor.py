"""System diagnostics for hftool.

Provides the `hftool doctor` command to check system health:
- Python version
- PyTorch installation
- GPU availability
- System dependencies (ffmpeg)
- Network connectivity
- Optional features
- Configuration status
"""

import os
import sys
import subprocess
import platform
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class CheckStatus(Enum):
    """Status of a diagnostic check."""
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    INFO = "info"


@dataclass
class CheckResult:
    """Result of a single diagnostic check."""
    name: str
    status: CheckStatus
    message: str
    details: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class DoctorReport:
    """Complete diagnostic report."""
    checks: List[CheckResult] = field(default_factory=list)
    
    def add_check(self, check: CheckResult) -> None:
        """Add a check result to the report."""
        self.checks.append(check)
    
    def has_errors(self) -> bool:
        """Check if any errors were found."""
        return any(c.status == CheckStatus.ERROR for c in self.checks)
    
    def has_warnings(self) -> bool:
        """Check if any warnings were found."""
        return any(c.status == CheckStatus.WARNING for c in self.checks)
    
    def get_exit_code(self) -> int:
        """Get appropriate exit code.
        
        Returns:
            0 if OK, 1 if warnings, 2 if errors
        """
        if self.has_errors():
            return 2
        elif self.has_warnings():
            return 1
        return 0
    
    def to_dict(self) -> Dict:
        """Convert report to dictionary for JSON output."""
        return {
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "details": c.details,
                    "suggestions": c.suggestions,
                }
                for c in self.checks
            ],
            "summary": {
                "total": len(self.checks),
                "ok": sum(1 for c in self.checks if c.status == CheckStatus.OK),
                "warnings": sum(1 for c in self.checks if c.status == CheckStatus.WARNING),
                "errors": sum(1 for c in self.checks if c.status == CheckStatus.ERROR),
            }
        }


def check_python_version() -> CheckResult:
    """Check Python version.
    
    Returns:
        CheckResult with Python version info
    """
    version_info = sys.version_info
    version_str = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    
    details = [
        f"Version: {version_str}",
        f"Executable: {sys.executable}",
        f"Platform: {platform.platform()}",
    ]
    
    # hftool requires Python 3.10+
    if version_info >= (3, 10):
        return CheckResult(
            name="Python Version",
            status=CheckStatus.OK,
            message=f"Python {version_str} (OK)",
            details=details,
        )
    else:
        return CheckResult(
            name="Python Version",
            status=CheckStatus.ERROR,
            message=f"Python {version_str} (too old)",
            details=details,
            suggestions=[
                "hftool requires Python 3.10 or newer",
                "Upgrade Python: https://www.python.org/downloads/",
            ]
        )


def check_pytorch() -> CheckResult:
    """Check PyTorch installation.
    
    Returns:
        CheckResult with PyTorch info
    """
    try:
        import torch
        
        version = torch.__version__
        details = [f"Version: {version}"]
        
        # Check for GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            details.append(f"GPU: {gpu_name}")
            details.append(f"GPU Count: {gpu_count}")
            
            # Detect ROCm vs CUDA
            if hasattr(torch.version, 'hip') and torch.version.hip:
                details.append("Backend: ROCm")
            else:
                cuda_version = torch.version.cuda
                details.append(f"Backend: CUDA {cuda_version}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            details.append("GPU: Apple Silicon (MPS)")
        else:
            details.append("GPU: Not available (CPU mode)")
        
        return CheckResult(
            name="PyTorch",
            status=CheckStatus.OK,
            message=f"PyTorch {version} installed",
            details=details,
        )
    
    except ImportError:
        return CheckResult(
            name="PyTorch",
            status=CheckStatus.ERROR,
            message="PyTorch not installed",
            suggestions=[
                "Run: hftool setup",
                "Or install manually:",
                "  NVIDIA: pip install torch torchvision torchaudio",
                "  AMD ROCm: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2",
                "  CPU: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
            ]
        )


def check_gpu_availability() -> CheckResult:
    """Check GPU availability and compatibility.
    
    Returns:
        CheckResult with GPU info
    """
    try:
        import torch
    except ImportError:
        return CheckResult(
            name="GPU Availability",
            status=CheckStatus.INFO,
            message="PyTorch not installed - cannot check GPU",
        )
    
    details = []
    suggestions = []
    
    # Check CUDA
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024**3)
            details.append(f"GPU {i}: {gpu_name}")
            details.append(f"  VRAM: {vram_gb:.1f} GB")
            details.append(f"  Compute: {props.major}.{props.minor}")
        
        return CheckResult(
            name="GPU Availability",
            status=CheckStatus.OK,
            message=f"{gpu_count} GPU(s) available",
            details=details,
        )
    
    # Check MPS (Apple Silicon)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        details.append("Apple Silicon GPU detected")
        details.append("Using MPS backend")
        
        return CheckResult(
            name="GPU Availability",
            status=CheckStatus.OK,
            message="Apple Silicon GPU available",
            details=details,
        )
    
    # No GPU
    else:
        details.append("No GPU detected")
        details.append("Running in CPU mode")
        
        suggestions.append("For better performance, use a GPU-enabled system")
        suggestions.append("Or use a cloud GPU service (Google Colab, AWS, etc.)")
        
        return CheckResult(
            name="GPU Availability",
            status=CheckStatus.WARNING,
            message="No GPU available (CPU mode)",
            details=details,
            suggestions=suggestions,
        )


def check_ffmpeg() -> CheckResult:
    """Check ffmpeg availability.
    
    Returns:
        CheckResult with ffmpeg info
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        
        if result.returncode == 0:
            # Parse version from first line
            first_line = result.stdout.split("\n")[0]
            version = first_line.split()[2] if len(first_line.split()) > 2 else "unknown"
            
            return CheckResult(
                name="ffmpeg",
                status=CheckStatus.OK,
                message=f"ffmpeg {version} found",
                details=["Required for text-to-video and audio output"],
            )
        else:
            return CheckResult(
                name="ffmpeg",
                status=CheckStatus.WARNING,
                message="ffmpeg not working properly",
                details=["Required for text-to-video and audio conversion"],
                suggestions=[
                    "Reinstall ffmpeg:",
                    "  Ubuntu/Debian: sudo apt install ffmpeg",
                    "  macOS: brew install ffmpeg",
                    "  Windows: Download from https://ffmpeg.org/",
                ]
            )
    
    except FileNotFoundError:
        return CheckResult(
            name="ffmpeg",
            status=CheckStatus.WARNING,
            message="ffmpeg not found",
            details=["Required for text-to-video and MP3 audio output"],
            suggestions=[
                "Install ffmpeg:",
                "  Ubuntu/Debian: sudo apt install ffmpeg",
                "  macOS: brew install ffmpeg",
                "  Windows: Download from https://ffmpeg.org/",
            ]
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            name="ffmpeg",
            status=CheckStatus.WARNING,
            message="ffmpeg check timed out",
        )


def check_network() -> CheckResult:
    """Check network connectivity to HuggingFace.
    
    Returns:
        CheckResult with network status
    """
    import socket
    
    try:
        # Try to resolve huggingface.co
        socket.create_connection(("huggingface.co", 443), timeout=5)
        
        return CheckResult(
            name="Network",
            status=CheckStatus.OK,
            message="HuggingFace Hub reachable",
            details=["Required for downloading models"],
        )
    
    except (socket.timeout, socket.gaierror, OSError):
        return CheckResult(
            name="Network",
            status=CheckStatus.WARNING,
            message="Cannot reach HuggingFace Hub",
            details=["Required for downloading models"],
            suggestions=[
                "Check your internet connection",
                "Check firewall settings",
                "Use --offline mode if models are already downloaded",
            ]
        )


def check_optional_features() -> CheckResult:
    """Check optional feature dependencies.
    
    Returns:
        CheckResult with feature availability
    """
    from hftool.utils.deps import is_available
    
    features = {
        "diffusers": "Text-to-image, image-to-image, text-to-video",
        "Pillow": "Image handling",
        "soundfile": "Audio handling",
        "InquirerPy": "Interactive file picker",
        "requests": "HTTP/URL inputs",
        "librosa": "Advanced audio processing",
        "opencv-python": "Advanced image processing",
    }
    
    available = []
    missing = []
    
    for package, description in features.items():
        if is_available(package):
            available.append(f"✓ {package}: {description}")
        else:
            missing.append(f"✗ {package}: {description}")
    
    details = []
    if available:
        details.append("Installed features:")
        details.extend(available)
    
    if missing:
        details.append("")
        details.append("Optional features not installed:")
        details.extend(missing)
    
    suggestions = []
    if missing:
        suggestions.append("Install optional features:")
        suggestions.append("  pip install hftool[all]")
        suggestions.append("Or install specific features:")
        suggestions.append("  pip install hftool[with_t2i,with_interactive]")
    
    if len(missing) == 0:
        status = CheckStatus.OK
        message = "All optional features installed"
    elif len(available) > len(missing):
        status = CheckStatus.INFO
        message = f"{len(available)}/{len(features)} optional features installed"
    else:
        status = CheckStatus.INFO
        message = f"{len(available)}/{len(features)} optional features installed"
    
    return CheckResult(
        name="Optional Features",
        status=status,
        message=message,
        details=details,
        suggestions=suggestions,
    )


def check_configuration() -> CheckResult:
    """Check configuration files.
    
    Returns:
        CheckResult with configuration status
    """
    from hftool.core.config import Config
    from pathlib import Path
    
    config = Config.get()
    
    details = []
    
    # Check for config files
    user_config = Path.home() / ".hftool" / "config.toml"
    project_config = Path.cwd() / ".hftool" / "config.toml"
    
    if user_config.exists():
        details.append(f"✓ User config: {user_config}")
    else:
        details.append(f"  User config: Not found ({user_config})")
    
    if project_config.exists():
        details.append(f"✓ Project config: {project_config}")
    else:
        details.append(f"  Project config: Not found ({project_config})")
    
    # Check models directory
    from hftool.core.download import get_models_dir
    models_dir = get_models_dir()
    details.append(f"Models directory: {models_dir}")
    
    if models_dir.exists():
        details.append(f"  Status: Exists")
    else:
        details.append(f"  Status: Will be created on first download")
    
    suggestions = []
    if not user_config.exists() and not project_config.exists():
        suggestions.append("Create config file: hftool config init")
        suggestions.append("Edit config: hftool config edit")
    
    return CheckResult(
        name="Configuration",
        status=CheckStatus.INFO,
        message="Configuration status",
        details=details,
        suggestions=suggestions,
    )


def run_doctor_checks() -> DoctorReport:
    """Run all diagnostic checks.
    
    Returns:
        DoctorReport with all check results
    """
    report = DoctorReport()
    
    # System checks
    report.add_check(check_python_version())
    report.add_check(check_pytorch())
    report.add_check(check_gpu_availability())
    
    # Environment checks
    report.add_check(check_ffmpeg())
    report.add_check(check_network())
    
    # Feature checks
    report.add_check(check_optional_features())
    
    # Configuration checks
    report.add_check(check_configuration())
    
    return report


def format_doctor_report(report: DoctorReport, use_color: bool = True) -> str:
    """Format doctor report for console output.
    
    Args:
        report: DoctorReport to format
        use_color: Whether to use ANSI colors
    
    Returns:
        Formatted report string (empty string if using rich output)
    """
    # Try to use rich for colored output
    console = None
    has_rich = False
    
    if use_color:
        try:
            from rich.console import Console
            console = Console()
            has_rich = True
        except ImportError:
            pass
    
    # If using rich, print directly to console
    if has_rich and console:
        console.print("")
        console.print("=" * 60)
        console.print("  hftool doctor - System Diagnostics")
        console.print("=" * 60)
        console.print("")
        
        for check in report.checks:
            # Status indicator
            if check.status == CheckStatus.OK:
                status_icon = "✓"
                status_color = "green"
            elif check.status == CheckStatus.WARNING:
                status_icon = "⚠"
                status_color = "yellow"
            elif check.status == CheckStatus.ERROR:
                status_icon = "✗"
                status_color = "red"
            else:  # INFO
                status_icon = "ℹ"
                status_color = "cyan"
            
            console.print(f"[{status_color}]{status_icon}[/{status_color}] {check.name}: {check.message}")
            
            for detail in check.details:
                console.print(f"  {detail}", style="dim")
            
            for suggestion in check.suggestions:
                console.print(f"  → {suggestion}", style="italic")
            
            console.print("")
        
        console.print("=" * 60)
        summary = report.to_dict()["summary"]
        console.print(f"Summary: [green]{summary['ok']} OK[/green], [yellow]{summary['warnings']} warnings[/yellow], [red]{summary['errors']} errors[/red]")
        console.print("")
        
        return ""  # Already printed
    
    # Fallback: plain text output
    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("  hftool doctor - System Diagnostics")
    lines.append("=" * 60)
    lines.append("")
    
    for check in report.checks:
        # Status indicator
        if check.status == CheckStatus.OK:
            status_icon = "[OK]"
        elif check.status == CheckStatus.WARNING:
            status_icon = "[WARNING]"
        elif check.status == CheckStatus.ERROR:
            status_icon = "[ERROR]"
        else:  # INFO
            status_icon = "[INFO]"
        
        lines.append(f"{status_icon} {check.name}: {check.message}")
        
        for detail in check.details:
            lines.append(f"  {detail}")
        
        for suggestion in check.suggestions:
            lines.append(f"  → {suggestion}")
        
        lines.append("")
    
    lines.append("=" * 60)
    summary = report.to_dict()["summary"]
    lines.append(f"Summary: {summary['ok']} OK, {summary['warnings']} warnings, {summary['errors']} errors")
    lines.append("")
    
    return "\n".join(lines)
