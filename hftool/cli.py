#!/usr/bin/env python3
"""hftool CLI - Command-line interface for Hugging Face models.

Supports:
- Text-to-Image (Z-Image, SDXL, FLUX)
- Text-to-Video (HunyuanVideo, CogVideoX, Wan2.2)
- Text-to-Speech (VibeVoice, Bark, MMS-TTS)
- Speech-to-Text (Whisper)
- And other transformers pipeline tasks
"""

import os
import sys
import warnings
import logging
from typing import Optional

# Load .env file FIRST (before ROCm setup) so env vars can be configured there
try:
    from dotenv import load_dotenv
    # Load from current directory, then home directory
    load_dotenv()  # .env in current directory
    load_dotenv(os.path.expanduser("~/.hftool/.env"))  # ~/.hftool/.env
except ImportError:
    pass  # python-dotenv not installed, skip

# =============================================================================
# Warning and Logging Configuration
# =============================================================================
# By default, suppress noisy warnings from dependencies (diffusers, transformers, torch)
# Enable debug mode with HFTOOL_DEBUG=1 in .env or environment to see all warnings
# Optionally log to file with HFTOOL_LOG_FILE=~/.hftool/hftool.log
_debug_mode = os.environ.get("HFTOOL_DEBUG", "").lower() in ("1", "true", "yes")
_log_file = os.environ.get("HFTOOL_LOG_FILE", "")

# Setup file logging if configured
_file_handler = None
if _log_file:
    _log_file = os.path.expanduser(_log_file)
    os.makedirs(os.path.dirname(_log_file) or ".", exist_ok=True)
    
    # Create file handler for capturing everything
    _file_handler = logging.FileHandler(_log_file, mode="a", encoding="utf-8")
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    
    # Add to root logger
    logging.getLogger().addHandler(_file_handler)
    logging.getLogger().setLevel(logging.DEBUG)
    
    # Capture warnings to the log file
    logging.captureWarnings(True)
    
    # Log startup
    _logger = logging.getLogger("hftool")
    _logger.info(f"hftool started - logging to {_log_file}")
    _logger.debug(f"Debug mode: {_debug_mode}")
    _logger.debug(f"Python: {sys.version}")
    _logger.debug(f"Working dir: {os.getcwd()}")

if not _debug_mode:
    # Suppress common non-breaking warnings from console (still logged to file if enabled)
    warnings.filterwarnings("ignore", message=".*expandable_segments not supported.*")
    warnings.filterwarnings("ignore", message=".*hipBLASLt on an unsupported architecture.*")
    warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")
    warnings.filterwarnings("ignore", message=".*config attributes.*were passed to.*but are not expected.*")
    warnings.filterwarnings("ignore", message=".*guidance_scale.*is passed.*but ignored.*")
    warnings.filterwarnings("ignore", message=".*Some parameters are on the meta device.*")
    
    # Suppress transformers/diffusers logging to console but allow file logging
    for _lib_name in ("transformers", "diffusers"):
        _lib_logger = logging.getLogger(_lib_name)
        _lib_logger.setLevel(logging.DEBUG)  # Capture everything
        # Remove any existing console handlers and add a NullHandler for console
        _lib_logger.handlers = []
        if _file_handler:
            _lib_logger.addHandler(_file_handler)
        # Add null handler to prevent "No handler found" warnings
        _lib_logger.addHandler(logging.NullHandler())
    
    # Set environment variables to suppress library-specific console output
    # These are checked by diffusers/transformers before printing warnings
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")

# =============================================================================
# ROCm Setup (for AMD GPU users without system-wide ROCm)
# =============================================================================
# Enable by setting HFTOOL_ROCM_PATH in your .env file or environment.
# If Ollama is installed, you can use its bundled ROCm libraries:
#   HFTOOL_ROCM_PATH=/usr/local/lib/ollama/rocm
#   HSA_OVERRIDE_GFX_VERSION=11.0.0  # For RX 7900 XTX/XT (gfx1100)
#
# GFX versions: gfx1100=RX 7900, gfx1101=RX 7800/7700, gfx1102=RX 7600
#               gfx1030=RX 6900/6800, gfx1031=RX 6700, gfx1032=RX 6600
_rocm_path = os.environ.get("HFTOOL_ROCM_PATH", "")
if _rocm_path and os.path.isdir(_rocm_path):
    _ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if _rocm_path not in _ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{_rocm_path}:{_ld_path}".rstrip(":")

# Configure ROCm/HIP memory allocation early (before PyTorch is imported)
# This helps prevent OOM errors with large images
if "PYTORCH_HIP_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import click

# Suppress known harmless warnings from dependencies
# - PyTorch CUDA warning when using ROCm or CPU
# - Deprecation warnings from diffusers internals
warnings.filterwarnings("ignore", message="User provided device_type of 'cuda', but CUDA is not available")
warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated")


# =============================================================================
# PyTorch Setup Check
# =============================================================================

def _check_pytorch_setup() -> dict:
    """Check PyTorch installation status.
    
    Returns:
        dict with keys: installed, version, gpu_available, gpu_type, gpu_name, needs_setup
    """
    result = {
        "installed": False,
        "version": None,
        "gpu_available": False,
        "gpu_type": None,  # "cuda", "rocm", "mps", or None
        "gpu_name": None,
        "needs_setup": False,
    }
    
    try:
        import torch
        result["installed"] = True
        result["version"] = torch.__version__
        
        # Check for GPU
        if torch.cuda.is_available():
            result["gpu_available"] = True
            result["gpu_name"] = torch.cuda.get_device_name(0)
            # Detect if ROCm or CUDA
            if torch.version.hip:
                result["gpu_type"] = "rocm"
            else:
                result["gpu_type"] = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            result["gpu_available"] = True
            result["gpu_type"] = "mps"
            result["gpu_name"] = "Apple Silicon"
        
    except ImportError:
        result["needs_setup"] = True
    
    return result


def _detect_system_gpu() -> dict:
    """Detect available GPUs on the system (independent of PyTorch).
    
    Returns:
        dict with keys: has_nvidia, has_amd, has_mps, amd_gpu_name
    """
    import subprocess
    import platform
    
    result = {
        "has_nvidia": False,
        "has_amd": False,
        "has_mps": False,
        "amd_gpu_name": None,
    }
    
    # Check for Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        result["has_mps"] = True
        return result
    
    # Check for NVIDIA GPU (lspci or nvidia-smi)
    try:
        output = subprocess.run(
            ["lspci"], capture_output=True, text=True, timeout=5
        )
        if "NVIDIA" in output.stdout:
            result["has_nvidia"] = True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Check for AMD GPU
    try:
        output = subprocess.run(
            ["lspci"], capture_output=True, text=True, timeout=5
        )
        for line in output.stdout.split("\n"):
            if "AMD" in line and ("VGA" in line or "Display" in line or "3D" in line):
                result["has_amd"] = True
                # Extract GPU name
                if "Radeon" in line:
                    result["amd_gpu_name"] = line.split(":")[-1].strip()
                break
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    return result


def _run_interactive_setup() -> bool:
    """Run interactive PyTorch setup wizard.
    
    Returns:
        True if setup was successful, False otherwise
    """
    import subprocess
    
    click.echo("")
    click.echo("=" * 60)
    click.echo("  hftool - First Time Setup")
    click.echo("=" * 60)
    click.echo("")
    
    # Check current PyTorch status
    pytorch_status = _check_pytorch_setup()
    system_gpu = _detect_system_gpu()
    
    if pytorch_status["installed"] and pytorch_status["gpu_available"]:
        click.echo(click.style("PyTorch is already configured correctly!", fg="green"))
        click.echo(f"  Version: {pytorch_status['version']}")
        click.echo(f"  GPU: {pytorch_status['gpu_name']} ({pytorch_status['gpu_type']})")
        click.echo("")
        return True
    
    if pytorch_status["installed"] and not pytorch_status["gpu_available"]:
        click.echo(click.style("PyTorch is installed but no GPU detected.", fg="yellow"))
        click.echo(f"  Version: {pytorch_status['version']}")
        click.echo("")
        click.echo("This could mean:")
        click.echo("  1. Wrong PyTorch version (CUDA vs ROCm vs CPU)")
        click.echo("  2. GPU drivers not installed")
        click.echo("  3. Running on CPU-only system")
        click.echo("")
    else:
        click.echo(click.style("PyTorch is not installed.", fg="yellow"))
        click.echo("")
    
    # Show detected hardware
    click.echo("Detected hardware:")
    if system_gpu["has_nvidia"]:
        click.echo(click.style("  [✓] NVIDIA GPU detected", fg="green"))
    if system_gpu["has_amd"]:
        gpu_name = system_gpu["amd_gpu_name"] or "AMD GPU"
        click.echo(click.style(f"  [✓] AMD GPU detected: {gpu_name}", fg="green"))
    if system_gpu["has_mps"]:
        click.echo(click.style("  [✓] Apple Silicon detected", fg="green"))
    if not any([system_gpu["has_nvidia"], system_gpu["has_amd"], system_gpu["has_mps"]]):
        click.echo("  [ ] No GPU detected (CPU mode)")
    click.echo("")
    
    # Determine recommended option
    if system_gpu["has_amd"]:
        recommended = "2"
    elif system_gpu["has_nvidia"]:
        recommended = "1"
    elif system_gpu["has_mps"]:
        recommended = "3"
    else:
        recommended = "4"
    
    # Show options
    click.echo("Select PyTorch version to install:")
    click.echo("")
    options = [
        ("1", "NVIDIA GPU (CUDA)", "pip install torch torchvision torchaudio"),
        ("2", "AMD GPU (ROCm 6.2)", "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2"),
        ("3", "Apple Silicon (MPS)", "pip install torch torchvision torchaudio"),
        ("4", "CPU only", "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"),
        ("5", "Skip (install manually later)", None),
    ]
    
    for opt, name, cmd in options:
        rec = " (recommended)" if opt == recommended else ""
        click.echo(f"  [{opt}] {name}{click.style(rec, fg='cyan')}")
    
    click.echo("")
    
    try:
        choice = click.prompt(
            "Your choice",
            default=recommended,
            type=click.Choice(["1", "2", "3", "4", "5"]),
        )
    except click.Abort:
        click.echo("\nSetup cancelled.")
        return False
    
    if choice == "5":
        click.echo("")
        click.echo("Skipping PyTorch installation.")
        click.echo("Install manually with one of these commands:")
        for opt, name, cmd in options[:4]:
            click.echo(f"  # {name}")
            click.echo(f"  {cmd}")
            click.echo("")
        return False
    
    # Get the pip command
    _, name, pip_cmd = options[int(choice) - 1]
    
    # Detect if running in pipx
    executable = sys.executable
    is_pipx = "pipx" in executable
    
    if is_pipx:
        # Convert pip command to pipx runpip
        pip_cmd = pip_cmd.replace("pip install", "pipx runpip hftool install")
        click.echo("")
        click.echo(f"Detected pipx environment.")
    
    click.echo("")
    click.echo(f"Installing PyTorch for {name}...")
    click.echo(f"Running: {pip_cmd}")
    click.echo("")
    
    # If AMD ROCm, show additional setup needed
    if choice == "2":
        click.echo(click.style("Note for AMD GPUs:", fg="yellow"))
        click.echo("  After installation, add these to your ~/.hftool/.env file:")
        click.echo("")
        click.echo("  # Use Ollama's ROCm libraries (if Ollama is installed)")
        click.echo("  HFTOOL_ROCM_PATH=/usr/local/lib/ollama/rocm")
        click.echo("")
        click.echo("  # Set your GPU architecture:")
        click.echo("  # RX 7900 XTX/XT: 11.0.0, RX 7800/7700: 11.0.1, RX 7600: 11.0.2")
        click.echo("  # RX 6900/6800: 10.3.0, RX 6700: 10.3.1, RX 6600: 10.3.2")
        click.echo("  HSA_OVERRIDE_GFX_VERSION=11.0.0")
        click.echo("")
        
        if not click.confirm("Continue with installation?", default=True):
            return False
    
    # Run the installation
    try:
        # Parse the command
        cmd_parts = pip_cmd.split()
        
        result = subprocess.run(
            cmd_parts,
            check=True,
            text=True,
        )
        
        click.echo("")
        click.echo(click.style("PyTorch installed successfully!", fg="green"))
        
        # Verify installation
        click.echo("")
        click.echo("Verifying installation...")
        
        # We need to re-exec since torch was just installed
        verify_cmd = [sys.executable, "-c", """
import torch
print(f"PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print("GPU available: Apple Silicon (MPS)")
else:
    print("GPU: Not available (CPU mode)")
"""]
        subprocess.run(verify_cmd)
        click.echo("")
        
        return True
        
    except subprocess.CalledProcessError as e:
        click.echo("")
        click.echo(click.style(f"Installation failed: {e}", fg="red"), err=True)
        click.echo("")
        click.echo("Try running the command manually:")
        click.echo(f"  {pip_cmd}")
        return False
    except FileNotFoundError:
        click.echo("")
        click.echo(click.style("pip/pipx not found in PATH", fg="red"), err=True)
        return False


def _ensure_pytorch_ready() -> bool:
    """Ensure PyTorch is installed and ready. Run setup wizard if needed.
    
    Returns:
        True if PyTorch is ready, False otherwise
    """
    status = _check_pytorch_setup()
    
    if status["installed"]:
        return True
    
    # PyTorch not installed - run interactive setup
    click.echo("")
    click.echo(click.style("PyTorch is required but not installed.", fg="yellow"))
    click.echo("")
    
    if click.confirm("Would you like to run the interactive setup wizard?", default=True):
        return _run_interactive_setup()
    else:
        click.echo("")
        click.echo("Install PyTorch manually:")
        click.echo("  # NVIDIA: pip install torch torchvision torchaudio")
        click.echo("  # AMD ROCm: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2")
        click.echo("  # CPU: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
        click.echo("")
        return False


# =============================================================================
# CLI GROUP
# =============================================================================

def _extract_extra_args():
    """Extract arguments after -- from sys.argv before Click processes them."""
    try:
        idx = sys.argv.index("--")
        extra = sys.argv[idx + 1:]
        # Remove -- and everything after from sys.argv so Click doesn't see them
        sys.argv = sys.argv[:idx]
        return extra
    except ValueError:
        return []

# Extract extra args BEFORE Click parses (this modifies sys.argv)
_EXTRA_ARGS_CACHE = _extract_extra_args()


@click.group(invoke_without_command=True)
@click.option("--task", "-t", default=None, help="Task to perform (e.g., text-to-image, tts, asr)")
@click.option("--model", "-m", default=None, help="Model name or path (uses task default if not specified)")
@click.option("--input", "-i", "input_data", default=None, help="Input data (text, file path, or URL)")
@click.option("--output-file", "-o", default=None, help="Output file path")
@click.option("--device", "-d", default="auto", help="Device to use (auto, cuda, mps, cpu)")
@click.option("--dtype", default=None, help="Data type (bfloat16, float16, float32)")
@click.option("--open/--no-open", default=None, help="Open output file with default application (auto-detected by default)")
@click.option("--list-tasks", is_flag=True, help="List all available tasks")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def main(
    ctx: click.Context,
    task: Optional[str],
    model: Optional[str],
    input_data: Optional[str],
    output_file: Optional[str],
    device: str,
    dtype: Optional[str],
    open: Optional[bool],
    list_tasks: bool,
    verbose: bool,
):
    """hftool - Run Hugging Face models from the command line.
    
    \b
    QUICK START:
      hftool -t t2i -i "A cat in space" -o cat.png
      hftool -t tts -i "Hello world" -o hello.wav
      hftool -t asr -i recording.wav -o transcript.txt
    
    \b
    MANAGE MODELS:
      hftool models                    # List available models
      hftool models -t text-to-image   # List models for a task
      hftool download -t t2i           # Download default model for task
      hftool download -t t2i -m sdxl   # Download specific model
    
    \b
    EXAMPLES:
      # Text-to-Image with Z-Image
      hftool -t text-to-image -i "A cat in space" -o cat.png
    
      # Text-to-Video with HunyuanVideo
      hftool -t text-to-video -i "A person walking" -o video.mp4
    
      # Speech-to-Text with Whisper
      hftool -t asr -i recording.wav -o transcript.txt
    
      # Pass extra arguments (after --)
      hftool -t t2i -i "A cat" -o cat.png -- --num_inference_steps 20
    """
    # Store options in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["open"] = open
    ctx.obj["extra_args"] = tuple(_EXTRA_ARGS_CACHE)
    
    # Handle --list-tasks
    if list_tasks:
        _list_tasks()
        return
    
    # If no subcommand and no task, show help
    if ctx.invoked_subcommand is None and task is None:
        click.echo(ctx.get_help())
        return
    
    # If subcommand is invoked, let it handle everything
    if ctx.invoked_subcommand is not None:
        return
    
    # Run task (legacy behavior for -t flag)
    if task is not None:
        if input_data is None:
            click.echo("Error: Missing option '--input' / '-i'.", err=True)
            sys.exit(1)
        
        # Ensure PyTorch is installed before running tasks
        if not _ensure_pytorch_ready():
            sys.exit(1)
        
        _run_task_command(
            ctx=ctx,
            task=task,
            model=model,
            input_data=input_data,
            output_file=output_file,
            device=device,
            dtype=dtype,
            verbose=verbose,
            open_output=open,
        )


# =============================================================================
# SETUP COMMAND
# =============================================================================

@main.command("setup")
@click.pass_context
def setup_command(ctx: click.Context):
    """Run interactive setup wizard for PyTorch installation.
    
    \b
    This wizard helps you install the correct PyTorch version for your hardware:
      - NVIDIA GPU (CUDA)
      - AMD GPU (ROCm)
      - Apple Silicon (MPS)
      - CPU only
    
    \b
    The wizard will:
      1. Detect your hardware
      2. Check current PyTorch installation
      3. Install/reinstall PyTorch if needed
      4. Provide ROCm configuration tips for AMD GPUs
    """
    _run_interactive_setup()


# =============================================================================
# RUN COMMAND (explicit subcommand alternative)
# =============================================================================

@main.command("run")
@click.option("--task", "-t", required=True, help="Task to perform")
@click.option("--model", "-m", default=None, help="Model name or path")
@click.option("--input", "-i", "input_data", required=True, help="Input data")
@click.option("--output-file", "-o", default=None, help="Output file path")
@click.option("--device", "-d", default="auto", help="Device to use")
@click.option("--dtype", default=None, help="Data type")
@click.option("--open/--no-open", default=None, help="Open output file with default application")
@click.pass_context
def run_command(
    ctx: click.Context,
    task: str,
    model: Optional[str],
    input_data: str,
    output_file: Optional[str],
    device: str,
    dtype: Optional[str],
    open: Optional[bool],
):
    """Run a task with the specified model."""
    # Ensure PyTorch is ready
    if not _ensure_pytorch_ready():
        sys.exit(1)
    
    verbose = ctx.obj.get("verbose", False)
    # Use command-level --open if specified, otherwise use global
    open_output = open if open is not None else ctx.obj.get("open")
    _run_task_command(ctx, task, model, input_data, output_file, device, dtype, verbose, open_output)


# =============================================================================
# MODELS COMMAND
# =============================================================================

@main.command("models")
@click.option("--task", "-t", default=None, help="Filter by task (e.g., t2i, tts)")
@click.option("--downloaded", "-d", is_flag=True, help="Show only downloaded models")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def models_command(
    ctx: click.Context,
    task: Optional[str],
    downloaded: bool,
    as_json: bool,
):
    """List available models for tasks.
    
    \b
    Examples:
      hftool models                      # List all models
      hftool models -t text-to-image     # List models for T2I
      hftool models -t t2i               # Same (using alias)
      hftool models --downloaded         # Show downloaded models only
    """
    from hftool.core.models import MODEL_REGISTRY, get_models_for_task
    from hftool.core.download import get_download_status, get_models_dir
    from hftool.core.registry import TASK_ALIASES
    
    verbose = ctx.obj.get("verbose", False)
    
    if verbose:
        click.echo(f"Models directory: {get_models_dir()}")
        click.echo("")
    
    if as_json:
        import json
        output = {}
        
        tasks_to_show = [TASK_ALIASES.get(task, task)] if task else list(MODEL_REGISTRY.keys())
        
        for task_name in tasks_to_show:
            if task_name not in MODEL_REGISTRY:
                continue
            models = MODEL_REGISTRY[task_name]
            output[task_name] = {}
            for short_name, info in models.items():
                status = get_download_status(info.repo_id)
                if downloaded and status != "downloaded":
                    continue
                output[task_name][short_name] = {
                    "repo_id": info.repo_id,
                    "name": info.name,
                    "size_gb": info.size_gb,
                    "is_default": info.is_default,
                    "status": status,
                    "description": info.description,
                }
        
        click.echo(json.dumps(output, indent=2))
        return
    
    # Text output
    if task:
        resolved_task = TASK_ALIASES.get(task, task)
        if resolved_task not in MODEL_REGISTRY:
            click.echo(f"Unknown task: {task}", err=True)
            click.echo(f"Available tasks: {', '.join(MODEL_REGISTRY.keys())}", err=True)
            sys.exit(1)
        tasks_to_show = [(resolved_task, MODEL_REGISTRY[resolved_task])]
    else:
        tasks_to_show = list(MODEL_REGISTRY.items())
    
    for task_name, models in tasks_to_show:
        click.echo(f"\n{task_name}:")
        click.echo("-" * 60)
        
        for short_name, info in models.items():
            status = get_download_status(info.repo_id)
            
            if downloaded and status != "downloaded":
                continue
            
            # Status indicator
            if status == "downloaded":
                status_str = click.style("[✓]", fg="green")
            elif status == "partial":
                status_str = click.style("[~]", fg="yellow")
            else:
                status_str = click.style("[ ]", fg="white")
            
            # Default indicator
            default_str = click.style(" (default)", fg="cyan") if info.is_default else ""
            
            click.echo(f"  {status_str} {short_name}{default_str}")
            click.echo(f"      {info.name} ({info.size_str})")
            click.echo(f"      {info.repo_id}")
            if info.description:
                click.echo(f"      {info.description}")
    
    click.echo("")
    click.echo("Legend: [✓] downloaded  [~] partial  [ ] not downloaded")
    click.echo(f"Models directory: {get_models_dir()}")


# =============================================================================
# DOWNLOAD COMMAND
# =============================================================================

@main.command("download")
@click.option("--task", "-t", default=None, help="Download default model for task")
@click.option("--model", "-m", default=None, help="Specific model to download (short name or repo_id)")
@click.option("--all", "download_all", is_flag=True, help="Download default models for all tasks")
@click.option("--force", "-f", is_flag=True, help="Re-download even if already exists")
@click.pass_context
def download_command(
    ctx: click.Context,
    task: Optional[str],
    model: Optional[str],
    download_all: bool,
    force: bool,
):
    """Download models from HuggingFace Hub.
    
    \b
    Examples:
      hftool download -t text-to-image        # Download default T2I model
      hftool download -t t2i -m sdxl          # Download specific model
      hftool download -m openai/whisper-large-v3  # Download by repo_id
      hftool download --all                   # Download all default models
    
    \b
    Environment Variables:
      HFTOOL_MODELS_DIR    Custom directory for model storage
                           Default: ~/.hftool/models/
    """
    from hftool.core.models import (
        MODEL_REGISTRY, get_models_for_task, get_default_model_info,
        get_model_info, find_model_by_repo_id
    )
    from hftool.core.download import download_model_with_progress, get_models_dir
    from hftool.core.registry import TASK_ALIASES
    
    verbose = ctx.obj.get("verbose", False)
    
    click.echo(f"Models directory: {get_models_dir()}")
    click.echo("")
    
    if download_all:
        # Download default model for each task
        click.echo("Downloading default models for all tasks...")
        click.echo("")
        
        for task_name in MODEL_REGISTRY.keys():
            try:
                info = get_default_model_info(task_name)
                click.echo(f"[{task_name}]")
                download_model_with_progress(
                    repo_id=info.repo_id,
                    size_gb=info.size_gb,
                    force=force,
                    pip_dependencies=info.pip_dependencies if info.pip_dependencies else None,
                )
                click.echo("")
            except Exception as e:
                click.echo(f"  Error: {e}", err=True)
        return
    
    if model:
        # Download specific model
        # First check if it's a repo_id
        found = find_model_by_repo_id(model)
        if found:
            task_name, short_name, info = found
        elif task:
            # Look up by short name within task
            resolved_task = TASK_ALIASES.get(task, task)
            try:
                info = get_model_info(resolved_task, model)
            except ValueError as e:
                click.echo(f"Error: {e}", err=True)
                sys.exit(1)
        else:
            # Try to find model across all tasks
            for task_name, models in MODEL_REGISTRY.items():
                if model in models:
                    info = models[model]
                    break
            else:
                click.echo(f"Error: Model '{model}' not found.", err=True)
                click.echo("Specify a task with -t or use full repo_id.", err=True)
                sys.exit(1)
        
        download_model_with_progress(
            repo_id=info.repo_id,
            size_gb=info.size_gb,
            force=force,
            pip_dependencies=info.pip_dependencies if info.pip_dependencies else None,
        )
        return
    
    if task:
        # Download default model for task
        resolved_task = TASK_ALIASES.get(task, task)
        try:
            info = get_default_model_info(resolved_task)
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        
        click.echo(f"Downloading default model for {resolved_task}...")
        download_model_with_progress(
            repo_id=info.repo_id,
            size_gb=info.size_gb,
            force=force,
            pip_dependencies=info.pip_dependencies if info.pip_dependencies else None,
        )
        return
    
    # No arguments - show help
    click.echo("Please specify what to download:")
    click.echo("  hftool download -t <task>           # Download default model for task")
    click.echo("  hftool download -t <task> -m <model> # Download specific model")
    click.echo("  hftool download --all               # Download all default models")
    click.echo("")
    click.echo("Available tasks:")
    for task_name in MODEL_REGISTRY.keys():
        click.echo(f"  {task_name}")


# =============================================================================
# STATUS COMMAND
# =============================================================================

@main.command("status")
@click.pass_context
def status_command(ctx: click.Context):
    """Show download status and disk usage.
    
    Displays information about:
    - Models directory location
    - Downloaded models and their sizes
    - Total disk usage
    """
    from hftool.core.download import get_models_dir, get_models_disk_usage, list_downloaded_models
    from hftool.core.models import find_model_by_repo_id
    
    models_dir = get_models_dir()
    click.echo(f"Models directory: {models_dir}")
    click.echo("")
    
    usage = get_models_disk_usage()
    
    if not usage["models"]:
        click.echo("No models downloaded yet.")
        click.echo("")
        click.echo("To download models, run:")
        click.echo("  hftool download -t <task>")
        return
    
    click.echo("Downloaded models:")
    click.echo("-" * 60)
    
    for model_info in usage["models"]:
        repo_id = model_info["repo_id"]
        size_str = model_info["size_str"]
        
        # Try to find model info
        found = find_model_by_repo_id(repo_id)
        if found:
            task_name, short_name, info = found
            click.echo(f"  {info.name}")
            click.echo(f"    Task: {task_name}")
            click.echo(f"    Size: {size_str}")
            click.echo(f"    Repo: {repo_id}")
        else:
            click.echo(f"  {repo_id}")
            click.echo(f"    Size: {size_str}")
            click.echo(f"    (Custom/unknown model)")
        click.echo("")
    
    click.echo("-" * 60)
    click.echo(f"Total disk usage: {usage['total_str']}")


# =============================================================================
# CLEAN COMMAND
# =============================================================================

@main.command("clean")
@click.option("--model", "-m", "models", multiple=True, help="Delete specific model(s) - can be used multiple times")
@click.option("--all", "delete_all", is_flag=True, help="Delete all downloaded models")
@click.option("--select", "-s", is_flag=True, help="Interactive selection mode")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.pass_context
def clean_command(
    ctx: click.Context,
    models: tuple,
    delete_all: bool,
    select: bool,
    yes: bool,
):
    """Delete downloaded models to free disk space.
    
    \b
    Examples:
      hftool clean                          # Interactive selection
      hftool clean -s                       # Same as above (explicit)
      hftool clean -m whisper-large-v3      # Delete specific model
      hftool clean -m model1 -m model2      # Delete multiple models
      hftool clean --all                    # Delete all models
      hftool clean --all -y                 # Delete without confirmation
    """
    from hftool.core.download import delete_model, list_downloaded_models, get_models_disk_usage
    from hftool.core.models import find_model_by_repo_id
    
    # Delete all models
    if delete_all:
        usage = get_models_disk_usage()
        if not usage["models"]:
            click.echo("No models to delete.")
            return
        
        click.echo(f"This will delete {len(usage['models'])} models ({usage['total_str']})")
        
        if not yes:
            if not click.confirm("Are you sure?"):
                click.echo("Cancelled.")
                return
        
        for model_info in usage["models"]:
            repo_id = model_info["repo_id"]
            if delete_model(repo_id):
                click.echo(f"Deleted: {repo_id}")
        
        click.echo("Done.")
        return
    
    # Delete specific model(s) by name
    if models:
        for model in models:
            repo_id = _resolve_model_to_repo_id(model)
            
            if not yes:
                if not click.confirm(f"Delete {repo_id}?"):
                    click.echo(f"Skipped: {repo_id}")
                    continue
            
            if delete_model(repo_id):
                click.echo(f"Deleted: {repo_id}")
            else:
                click.echo(f"Model not found: {repo_id}")
        return
    
    # Interactive selection mode (default when no arguments)
    usage = get_models_disk_usage()
    if not usage["models"]:
        click.echo("No models downloaded.")
        click.echo("")
        click.echo("To download models, run:")
        click.echo("  hftool download -t <task>")
        return
    
    # Show interactive selection
    selected = _interactive_model_select(usage["models"])
    
    if not selected:
        click.echo("No models selected.")
        return
    
    # Calculate total size
    total_size = sum(m["size_bytes"] for m in selected)
    total_str = _format_size(total_size)
    
    click.echo("")
    click.echo(f"Selected {len(selected)} model(s) to delete ({total_str}):")
    for model_info in selected:
        click.echo(f"  - {model_info['repo_id']} ({model_info['size_str']})")
    click.echo("")
    
    if not yes:
        if not click.confirm("Delete these models?"):
            click.echo("Cancelled.")
            return
    
    # Delete selected models
    for model_info in selected:
        repo_id = model_info["repo_id"]
        if delete_model(repo_id):
            click.echo(f"Deleted: {repo_id}")
    
    click.echo("Done.")


def _resolve_model_to_repo_id(model: str) -> str:
    """Resolve a model name/shortname to repo_id."""
    from hftool.core.models import find_model_by_repo_id, MODEL_REGISTRY
    
    # Try to find by repo_id
    found = find_model_by_repo_id(model)
    if found:
        return found[2].repo_id
    
    # Try to find by short name across all tasks
    for task_name, models in MODEL_REGISTRY.items():
        if model in models:
            return models[model].repo_id
    
    # Assume it's a repo_id directly
    return model


def _format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size = size / 1024
    return f"{size:.1f} PB"


def _interactive_model_select(models: list) -> list:
    """Interactive model selection using numbered list.
    
    Args:
        models: List of model info dicts from get_models_disk_usage()
    
    Returns:
        List of selected model info dicts
    """
    from hftool.core.models import find_model_by_repo_id
    
    click.echo("")
    click.echo("Downloaded models:")
    click.echo("-" * 60)
    
    # Display numbered list
    for i, model_info in enumerate(models, 1):
        repo_id = model_info["repo_id"]
        size_str = model_info["size_str"]
        
        # Try to get friendly name
        found = find_model_by_repo_id(repo_id)
        if found:
            task_name, short_name, info = found
            display_name = f"{info.name} ({task_name})"
        else:
            display_name = repo_id
        
        click.echo(f"  [{i:2d}] {display_name}")
        click.echo(f"       {repo_id} - {size_str}")
    
    click.echo("-" * 60)
    click.echo("")
    click.echo("Enter model numbers to delete (comma-separated, ranges with -, or 'all'):")
    click.echo("Examples: 1,3,5  or  1-3  or  1,3-5,7  or  all")
    click.echo("")
    
    try:
        selection = click.prompt("Selection", default="").strip()
    except click.Abort:
        return []
    
    if not selection:
        return []
    
    # Parse selection
    selected_indices = set()
    
    if selection.lower() == "all":
        return models
    
    try:
        parts = selection.split(",")
        for part in parts:
            part = part.strip()
            if "-" in part:
                # Range (e.g., "1-5")
                start, end = part.split("-", 1)
                start_idx = int(start.strip())
                end_idx = int(end.strip())
                for i in range(start_idx, end_idx + 1):
                    if 1 <= i <= len(models):
                        selected_indices.add(i - 1)
            else:
                # Single number
                idx = int(part)
                if 1 <= idx <= len(models):
                    selected_indices.add(idx - 1)
    except ValueError:
        click.echo("Invalid selection format.", err=True)
        return []
    
    return [models[i] for i in sorted(selected_indices)]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _list_tasks():
    """Print list of available tasks."""
    from hftool.core.registry import list_tasks, TASK_ALIASES
    
    click.echo("Available tasks:")
    click.echo("")
    
    tasks = list_tasks()
    for name, description in sorted(tasks.items()):
        click.echo(f"  {name}")
        click.echo(f"    {description}")
    
    click.echo("")
    click.echo("Task aliases:")
    for alias, target in sorted(TASK_ALIASES.items()):
        click.echo(f"  {alias} -> {target}")


def _run_task_command(
    ctx: click.Context,
    task: str,
    model: Optional[str],
    input_data: str,
    output_file: Optional[str],
    device: str,
    dtype: Optional[str],
    verbose: bool,
    open_output: Optional[bool] = None,
):
    """Execute a task (internal helper)."""
    # Parse extra arguments (after --)
    extra_args = ctx.obj.get("extra_args", ()) if ctx.obj else ()
    extra_kwargs = _parse_extra_args(list(extra_args))
    
    if verbose:
        click.echo(f"Task: {task}")
        click.echo(f"Model: {model or '(default)'}")
        click.echo(f"Input: {input_data}")
        click.echo(f"Output: {output_file or '(auto)'}")
        click.echo(f"Device: {device}")
        if extra_kwargs:
            click.echo(f"Extra args: {extra_kwargs}")
    
    try:
        # Import here to avoid slow startup for --help
        from hftool.core.registry import get_task_config, TASK_ALIASES
        from hftool.core.models import get_default_model_info, get_model_info, find_model_by_repo_id
        from hftool.core.download import ensure_model_available, is_model_downloaded, get_model_path
        
        # Resolve task alias
        resolved_task = TASK_ALIASES.get(task, task)
        
        # Get task configuration
        task_config = get_task_config(resolved_task)
        
        # Determine which model to use
        model_info = None
        pip_dependencies = None
        
        if model is None:
            # Use default model
            model_info = get_default_model_info(resolved_task)
            model_repo_id = model_info.repo_id
            model_size = model_info.size_gb
            model_name = model_info.name
            pip_dependencies = model_info.pip_dependencies
        else:
            # Check if model is a local path
            if os.path.exists(model):
                model_repo_id = model
                model_size = 0
                model_name = os.path.basename(model)
            else:
                # Try to find model info
                try:
                    model_info = get_model_info(resolved_task, model)
                    model_repo_id = model_info.repo_id
                    model_size = model_info.size_gb
                    model_name = model_info.name
                    pip_dependencies = model_info.pip_dependencies
                except ValueError:
                    # Not in registry - assume it's a HuggingFace repo_id
                    model_repo_id = model
                    model_size = 5.0  # Estimate
                    model_name = model.split("/")[-1] if "/" in model else model
        
        if verbose:
            click.echo(f"Using model: {model_repo_id}")
            if pip_dependencies:
                click.echo(f"Model dependencies: {', '.join(pip_dependencies)}")
        
        # Ensure model is available (prompts to download if needed)
        if not os.path.exists(model_repo_id):
            model_path = ensure_model_available(
                repo_id=model_repo_id,
                size_gb=model_size,
                task_name=resolved_task,
                model_name=model_name,
                pip_dependencies=pip_dependencies,
            )
            # Use the local path for loading
            model_to_load = str(model_path)
        else:
            model_to_load = model_repo_id
        
        # Check dependencies
        _check_task_deps(task_config, verbose)
        
        # Run the task
        result = _run_task(
            task_name=resolved_task,
            task_config=task_config,
            model=model_to_load,
            input_data=input_data,
            output_file=output_file,
            device=device,
            dtype=dtype,
            verbose=verbose,
            **extra_kwargs
        )
        
        # Print result summary
        if output_file:
            click.echo(f"Output saved to: {output_file}")
            
            # Determine if we should open the file
            should_open = _should_open_output(
                open_output=open_output,
                output_file=output_file,
                output_type=task_config.output_type,
            )
            
            if should_open:
                _open_file(output_file, verbose)
        elif isinstance(result, str):
            click.echo(result)
        elif isinstance(result, dict) and "text" in result:
            click.echo(result["text"])
        
    except SystemExit:
        raise
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _parse_extra_args(args: list) -> dict:
    """Parse extra arguments passed after --.
    
    Converts --arg value pairs to a dictionary.
    Handles boolean flags (--flag with no value).
    """
    kwargs = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            key = arg[2:].replace("-", "_")
            
            # Check if next arg is a value or another flag
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                value = args[i + 1]
                # Try to parse as number or boolean
                value = _parse_value(value)
                kwargs[key] = value
                i += 2
            else:
                # Boolean flag
                kwargs[key] = True
                i += 1
        else:
            i += 1
    
    return kwargs


def _parse_value(value: str):
    """Parse a string value to appropriate type."""
    # Boolean
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False
    
    # Integer
    try:
        return int(value)
    except ValueError:
        pass
    
    # Float
    try:
        return float(value)
    except ValueError:
        pass
    
    # String
    return value


def _should_open_output(
    open_output: Optional[bool],
    output_file: str,
    output_type: str,
) -> bool:
    """Determine if we should open the output file.
    
    Args:
        open_output: User's explicit preference (True/False/None)
        output_file: Path to the output file
        output_type: Type of output ("image", "audio", "video", "text")
    
    Returns:
        True if we should attempt to open the file
    """
    # If user explicitly specified, use that
    if open_output is not None:
        return open_output
    
    # Check environment variable
    env_open = os.environ.get("HFTOOL_AUTO_OPEN", "").lower()
    if env_open in ("1", "true", "yes"):
        return True
    if env_open in ("0", "false", "no"):
        return False
    
    # Auto-detect based on output type
    # By default, open image, audio, and video files (not text)
    openable_types = {"image", "audio", "video"}
    return output_type in openable_types


def _open_file(file_path: str, verbose: bool = False) -> bool:
    """Open a file with the system's default application.
    
    Args:
        file_path: Path to the file to open
        verbose: Whether to print detailed messages
    
    Returns:
        True if the file was opened successfully
    """
    import platform
    import subprocess
    
    if not os.path.exists(file_path):
        click.echo(f"Cannot open file: {file_path} (file not found)", err=True)
        return False
    
    system = platform.system().lower()
    
    try:
        if system == "darwin":  # macOS
            cmd = ["open", file_path]
        elif system == "windows":
            # On Windows, use os.startfile (no subprocess needed)
            os.startfile(file_path)  # type: ignore
            if verbose:
                click.echo(f"Opened: {file_path}")
            return True
        elif system == "linux":
            # Try xdg-open first (standard), then common alternatives
            openers = ["xdg-open", "gnome-open", "kde-open", "exo-open"]
            cmd = None
            
            for opener in openers:
                import shutil
                if shutil.which(opener):
                    cmd = [opener, file_path]
                    break
            
            if cmd is None:
                click.echo(
                    f"Cannot open file: no file opener found. "
                    f"Install xdg-utils or open manually: {file_path}",
                    err=True
                )
                return False
        else:
            click.echo(f"Cannot open file: unsupported platform '{system}'", err=True)
            return False
        
        # Execute the open command
        if verbose:
            click.echo(f"Opening: {file_path}")
        
        # Use Popen to not block and detach the process
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        
        if verbose:
            click.echo(f"Opened: {file_path}")
        
        return True
        
    except FileNotFoundError as e:
        click.echo(f"Cannot open file: application not found ({e})", err=True)
        return False
    except PermissionError as e:
        click.echo(f"Cannot open file: permission denied ({e})", err=True)
        return False
    except Exception as e:
        click.echo(f"Cannot open file: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def _check_task_deps(task_config, verbose: bool):
    """Check if required dependencies are installed."""
    from hftool.utils.deps import is_available, is_ffmpeg_available
    
    missing = []
    for dep in task_config.required_deps:
        if not is_available(dep):
            missing.append(dep)
    
    if missing:
        click.echo(f"Missing dependencies: {', '.join(missing)}", err=True)
        click.echo(f"Install with: pip install {' '.join(missing)}", err=True)
        sys.exit(1)
    
    if task_config.requires_ffmpeg and not is_ffmpeg_available():
        click.echo("ffmpeg is required for this task but was not found.", err=True)
        click.echo("Please install ffmpeg: https://ffmpeg.org/download.html", err=True)
        sys.exit(1)
    
    if verbose:
        click.echo("All dependencies satisfied.")


def _run_task(
    task_name: str,
    task_config,
    model: str,
    input_data: str,
    output_file: Optional[str],
    device: str,
    dtype: Optional[str],
    verbose: bool,
    **kwargs
):
    """Run the specified task."""
    from hftool.io.input_loader import load_input, detect_input_type, InputType
    from hftool.io.output_handler import get_output_path, OutputType
    
    # Map task output types to OutputType
    output_type_map = {
        "text": OutputType.TEXT,
        "image": OutputType.IMAGE,
        "audio": OutputType.AUDIO,
        "video": OutputType.VIDEO,
    }
    
    # Determine output path if not specified
    if output_file is None:
        import json as json_module
        output_type = output_type_map.get(task_config.output_type, OutputType.TEXT)
        
        # Extract actual file path from JSON input if needed (for i2i tasks)
        actual_input_path = None
        if task_config.input_type != "text":
            # Try to parse JSON input to extract image path
            if input_data.strip().startswith("{"):
                try:
                    data = json_module.loads(input_data)
                    img_path = data.get("image")
                    # Handle both single path and list of paths
                    if isinstance(img_path, list):
                        actual_input_path = img_path[0] if img_path else None
                    else:
                        actual_input_path = img_path
                except (json_module.JSONDecodeError, TypeError):
                    pass
            else:
                actual_input_path = input_data
        
        output_file = get_output_path(
            input_path=actual_input_path,
            output_type=output_type,
        )
    
    # Load and run task handler
    if task_name == "text-to-image":
        from hftool.tasks.text_to_image import create_task
        task_handler = create_task(device=device, dtype=dtype)
    elif task_name == "image-to-image":
        from hftool.tasks.image_to_image import create_task
        task_handler = create_task(device=device, dtype=dtype)
    elif task_name in ("text-to-video", "image-to-video"):
        from hftool.tasks.text_to_video import create_task
        mode = task_config.config.get("mode", "t2v")
        task_handler = create_task(device=device, dtype=dtype, mode=mode)
    elif task_name == "text-to-speech":
        from hftool.tasks.text_to_speech import create_task
        task_handler = create_task(device=device, dtype=dtype)
    elif task_name == "automatic-speech-recognition":
        from hftool.tasks.speech_to_text import create_task
        task_handler = create_task(device=device, dtype=dtype)
    else:
        # Fallback to generic transformers pipeline
        from hftool.tasks.transformers_generic import create_task
        task_handler = create_task(task_name=task_name, device=device, dtype=dtype)
    
    if verbose:
        click.echo(f"Loading model: {model}")
    
    # Execute task
    result = task_handler.execute(
        model=model,
        input_data=input_data,
        output_path=output_file,
        **kwargs
    )
    
    return result


if __name__ == "__main__":
    main()
