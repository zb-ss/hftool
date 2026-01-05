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
from typing import Any, Dict, Optional

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

# Import shell completion functions
from hftool.core.completion import (
    complete_tasks,
    complete_models,
    complete_devices,
    complete_dtypes,
    complete_input,
)

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
@click.option("--task", "-t", default=None, shell_complete=complete_tasks, help="Task to perform (e.g., text-to-image, tts, asr)")
@click.option("--model", "-m", default=None, shell_complete=complete_models, help="Model name or path (uses task default if not specified)")
@click.option("--input", "-i", "input_data", default=None, shell_complete=complete_input, help="Input data (text, file path, @ reference, @? for interactive, @*.ext for glob)")
@click.option("--output-file", "-o", default=None, help="Output file path (auto-generated if omitted)")
@click.option("--device", "-d", default="auto", shell_complete=complete_devices, help="Device to use (auto, cuda, mps, cpu)")
@click.option("--dtype", default=None, shell_complete=complete_dtypes, help="Data type (bfloat16, float16, float32)")
@click.option("--seed", type=int, default=None, help="Random seed for reproducible generation")
@click.option("--interactive", is_flag=True, help="Interactive mode for complex inputs (JSON builder)")
@click.option("-I", "--interactive-wizard", "wizard", is_flag=True, help="Full interactive wizard (select task, model, input, etc.)")
@click.option("--dry-run", is_flag=True, help="Preview operation without executing (shows model info, VRAM estimate, parameters)")
@click.option("--batch", default=None, help="Batch mode: process multiple inputs from file or directory")
@click.option("--batch-json", default=None, help="Batch mode: process inputs from JSON array file")
@click.option("--batch-output-dir", default=None, help="Output directory for batch processing")
@click.option("--open/--no-open", default=None, help="Open output file with default application (auto-detected by default)")
@click.option("--list-tasks", is_flag=True, help="List all available tasks")
@click.option("--quiet", "-q", is_flag=True, help="Quiet mode (only output file path)")
@click.option("--json", "output_json", is_flag=True, help="Output result as JSON")
@click.option("--embed-metadata/--no-embed-metadata", default=True, help="Embed generation metadata in output files (default: enabled)")
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
    seed: Optional[int],
    interactive: bool,
    wizard: bool,
    dry_run: bool,
    batch: Optional[str],
    batch_json: Optional[str],
    batch_output_dir: Optional[str],
    open: Optional[bool],
    list_tasks: bool,
    quiet: bool,
    output_json: bool,
    embed_metadata: bool,
    verbose: bool,
):
    """hftool - Run Hugging Face models from the command line.
    
    \b
    QUICK START:
      hftool -t t2i -i "A cat in space" -o cat.png
      hftool -t t2i -i @ -o cat.png              # Interactive file picker
      hftool -t asr -i @*.wav -o transcript.txt  # Glob pattern
    
    \b
    CONFIGURATION:
      hftool config init                         # Create default config
      hftool config show                         # View current config
      hftool config edit                         # Edit in $EDITOR
    
    \b
    PREVIEW & HISTORY:
      hftool -t t2i -i "A cat" --dry-run         # Preview without running
      hftool history                             # View command history
      hftool history --rerun 5                   # Re-run command #5
    
    \b
    FILE PICKER (@ syntax):
      @           Interactive file picker
      @?          Interactive with fuzzy search
      @.          Pick from current directory
      @~          Pick from home directory
      @/path/     Pick from specific directory
      @*.wav      Files matching glob pattern
      @@          Recent files from history
    
    \b
    INTERACTIVE MODE:
      hftool -I                                  # Full interactive wizard
      hftool --interactive-wizard                # Same as above
      hftool -t i2i --interactive                # Guided JSON builder for input
    
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
    
      # Interactive file selection
      hftool -t t2i -i @ -o output.png
    
      # Pass extra arguments (after --)
      hftool -t t2i -i "A cat" -o cat.png -- --num_inference_steps 20
      
      # Reproducible generation with seed
      hftool -t t2i -i "A cat" -o cat.png --seed 42
    """
    # Store options in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["open"] = open
    ctx.obj["seed"] = seed
    ctx.obj["interactive"] = interactive
    ctx.obj["quiet"] = quiet
    ctx.obj["output_json"] = output_json
    ctx.obj["embed_metadata"] = embed_metadata
    ctx.obj["batch"] = batch
    ctx.obj["batch_json"] = batch_json
    ctx.obj["batch_output_dir"] = batch_output_dir
    ctx.obj["extra_args"] = tuple(_EXTRA_ARGS_CACHE)
    
    # Handle --list-tasks
    if list_tasks:
        _list_tasks()
        return
    
    # Handle --interactive-wizard / -I (full wizard mode)
    if wizard:
        from hftool.io.interactive_mode import run_interactive_mode, check_interactive_mode
        
        # Ensure PyTorch is installed before running wizard
        if not _ensure_pytorch_ready():
            sys.exit(1)
        
        try:
            params = run_interactive_mode(quiet=quiet, output_json=output_json)
            
            # Run the task with wizard parameters
            _run_task_command(
                ctx=ctx,
                task=params["task"],
                model=params["model"],
                input_data=params["input_data"],
                output_file=params["output_file"],
                device=params["device"],
                dtype=params["dtype"],
                seed=params["seed"],
                interactive=False,
                verbose=verbose,
                quiet=params.get("quiet", quiet),
                output_json=params.get("output_json", output_json),
                embed_metadata=embed_metadata,
                open_output=open,
                wizard_extra_kwargs=params.get("extra_kwargs"),
            )
        except click.Abort:
            sys.exit(0)
        return
    
    # Check if interactive mode should be auto-enabled via config/env
    if ctx.invoked_subcommand is None and task is None:
        from hftool.io.interactive_mode import check_interactive_mode
        
        if check_interactive_mode(ctx, wizard):
            # Recursively call with wizard enabled
            ctx.invoke(main, wizard=True, quiet=quiet, output_json=output_json, 
                      verbose=verbose, embed_metadata=embed_metadata, open=open)
            return
        
        # Show help if not in interactive mode
        click.echo(ctx.get_help())
        return
    
    # If subcommand is invoked (like 'models', 'download', etc.), let it handle everything
    if ctx.invoked_subcommand is not None:
        return
    
    # Run task (legacy behavior for -t flag)
    if task is not None:
        # Handle interactive mode or missing input
        if input_data is None and not interactive:
            click.echo("Error: Missing option '--input' / '-i' (or use --interactive).", err=True)
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
            seed=seed,
            interactive=interactive,
            verbose=verbose,
            quiet=quiet,
            output_json=output_json,
            embed_metadata=embed_metadata,
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
@click.option("--task", "-t", required=True, shell_complete=complete_tasks, help="Task to perform")
@click.option("--model", "-m", default=None, shell_complete=complete_models, help="Model name or path")
@click.option("--input", "-i", "input_data", default=None, shell_complete=complete_input, help="Input data (@ references or @? for interactive)")
@click.option("--output-file", "-o", default=None, help="Output file path")
@click.option("--device", "-d", default="auto", shell_complete=complete_devices, help="Device to use")
@click.option("--dtype", default=None, shell_complete=complete_dtypes, help="Data type")
@click.option("--seed", type=int, default=None, help="Random seed for reproducibility")
@click.option("--interactive", is_flag=True, help="Interactive JSON builder mode")
@click.option("--open/--no-open", default=None, help="Open output file with default application")
@click.pass_context
def run_command(
    ctx: click.Context,
    task: str,
    model: Optional[str],
    input_data: Optional[str],
    output_file: Optional[str],
    device: str,
    dtype: Optional[str],
    seed: Optional[int],
    interactive: bool,
    open: Optional[bool],
):
    """Run a task with the specified model."""
    # Ensure PyTorch is ready
    if not _ensure_pytorch_ready():
        sys.exit(1)
    
    verbose = ctx.obj.get("verbose", False)
    # Use command-level --open if specified, otherwise use global
    open_output = open if open is not None else ctx.obj.get("open")
    # Use command-level --seed if specified, otherwise use global
    final_seed = seed if seed is not None else ctx.obj.get("seed")
    # Use command-level --interactive if specified, otherwise use global
    final_interactive = interactive or ctx.obj.get("interactive", False)
    
    _run_task_command(ctx, task, model, input_data, output_file, device, dtype, final_seed, final_interactive, verbose, open_output)


# =============================================================================
# HISTORY COMMAND
# =============================================================================

@main.command("history")
@click.option("--clear", is_flag=True, help="Clear all history")
@click.option("--rerun", type=int, metavar="ID", help="Re-run command from history")
@click.option("--limit", "-n", type=int, default=10, help="Number of entries to show")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def history_command(
    ctx: click.Context,
    clear: bool,
    rerun: Optional[int],
    limit: int,
    as_json: bool,
):
    """View and manage command history.
    
    \b
    Examples:
        hftool history                 # Show recent history
        hftool history -n 20           # Show last 20 commands
        hftool history --rerun 42      # Re-run command #42
        hftool history --clear         # Clear all history
    """
    from hftool.core.history import History
    
    history = History.get()
    
    # Clear history
    if clear:
        if click.confirm("Clear all command history?"):
            history.clear()
            click.echo("History cleared.")
        return
    
    # Re-run command
    if rerun is not None:
        entry = history.get_by_id(rerun)
        if entry is None:
            click.echo(f"Error: No history entry with ID {rerun}", err=True)
            sys.exit(1)
        
        click.echo(f"Re-running command #{entry.id} from {entry.get_timestamp_str()}:")
        click.echo(f"  {entry.to_command()}")
        click.echo("")
        
        if not click.confirm("Continue?", default=True):
            return
        
        # Extract parameters and re-run
        _run_task_command(
            ctx=ctx,
            task=entry.task,
            model=entry.model,
            input_data=entry.input_data,
            output_file=entry.output_file,
            device=entry.device,
            dtype=entry.dtype,
            seed=entry.seed,
            interactive=False,
            verbose=ctx.obj.get("verbose", False),
            open_output=ctx.obj.get("open"),
        )
        return
    
    # Show history
    entries = history.get_recent(limit=limit)
    
    if not entries:
        click.echo("No command history yet.")
        return
    
    if as_json:
        import json
        from dataclasses import asdict
        output = [asdict(entry) for entry in entries]
        click.echo(json.dumps(output, indent=2))
        return
    
    # Text output
    click.echo("")
    click.echo("Recent command history:")
    click.echo("=" * 80)
    
    for entry in entries:
        # Status indicator
        status = click.style("✓", fg="green") if entry.success else click.style("✗", fg="red")
        
        # Header
        click.echo(f"\n[{entry.id}] {status} {entry.get_timestamp_str()} - {entry.task}")
        
        # Details
        if entry.model:
            click.echo(f"    Model: {entry.model}")
        
        # Show input (truncate if too long)
        input_display = entry.input_data
        if len(input_display) > 60:
            input_display = input_display[:57] + "..."
        click.echo(f"    Input: {input_display}")
        
        if entry.output_file:
            click.echo(f"    Output: {entry.output_file}")
        
        if entry.seed is not None:
            click.echo(f"    Seed: {entry.seed}")
        
        if not entry.success and entry.error_message:
            error_display = entry.error_message
            if len(error_display) > 60:
                error_display = error_display[:57] + "..."
            click.echo(click.style(f"    Error: {error_display}", fg="red"))
        
        # Show command for reproduction
        click.echo(click.style(f"    Rerun: hftool history --rerun {entry.id}", fg="cyan"))
    
    click.echo("")
    click.echo("=" * 80)
    click.echo(f"Showing {len(entries)} most recent commands")


# =============================================================================
# MODELS COMMAND
# =============================================================================

@main.command("models")
@click.option("--task", "-t", default=None, shell_complete=complete_tasks, help="Filter by task (e.g., t2i, tts)")
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
@click.option("--task", "-t", default=None, shell_complete=complete_tasks, help="Download default model for task")
@click.option("--model", "-m", default=None, shell_complete=complete_models, help="Specific model to download (short name or repo_id)")
@click.option("--all", "download_all", is_flag=True, help="Download default models for all tasks")
@click.option("--force", "-f", is_flag=True, help="Re-download even if already exists")
@click.option("--resume/--no-resume", default=True, help="Resume partial downloads (default: enabled)")
@click.pass_context
def download_command(
    ctx: click.Context,
    task: Optional[str],
    model: Optional[str],
    download_all: bool,
    force: bool,
    resume: bool,
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
                    resume=resume,
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
            resume=resume,
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
            resume=resume,
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
    - Partial downloads (resumable)
    - Total disk usage
    """
    from hftool.core.download import get_models_dir, get_models_disk_usage, list_downloaded_models, get_partial_downloads
    from hftool.core.models import find_model_by_repo_id
    
    models_dir = get_models_dir()
    click.echo(f"Models directory: {models_dir}")
    click.echo("")
    
    # Check for partial downloads
    partial_downloads = get_partial_downloads()
    if partial_downloads:
        click.echo(click.style("Partial downloads (resumable):", fg="yellow"))
        click.echo("-" * 60)
        for partial in partial_downloads:
            repo_id = partial["repo_id"]
            click.echo(f"  {repo_id}")
            click.echo(f"    Resume: hftool download -m {repo_id}")
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
# INFO COMMAND
# =============================================================================

@main.command("info")
@click.argument("model_name")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def info_command(ctx: click.Context, model_name: str, as_json: bool):
    """Show detailed information about a model.
    
    MODEL_NAME can be a short name (e.g., 'whisper-large-v3'), 
    full repo ID (e.g., 'openai/whisper-large-v3'), or any model identifier.
    
    \b
    Examples:
      hftool info whisper-large-v3
      hftool info openai/whisper-large-v3
      hftool info z-image-turbo
      hftool info stabilityai/stable-diffusion-xl-base-1.0
    """
    from hftool.core.models import find_model_by_repo_id, MODEL_REGISTRY
    from hftool.core.download import get_download_status, get_models_dir, get_model_path
    import json as json_module
    
    # Try to find the model
    found = find_model_by_repo_id(model_name)
    
    if not found:
        # Try to find by short name across all tasks
        found_by_short = None
        for task_name, models in MODEL_REGISTRY.items():
            if model_name in models:
                info = models[model_name]
                found_by_short = (task_name, model_name, info)
                break
        
        if found_by_short:
            found = found_by_short
        else:
            click.echo(f"Error: Model '{model_name}' not found.", err=True)
            click.echo("", err=True)
            click.echo("Use 'hftool models' to see available models.", err=True)
            sys.exit(1)
    
    task_name, short_name, info = found
    
    # Get download status
    status = get_download_status(info.repo_id)
    is_downloaded = status == "downloaded"
    
    # Get local path if downloaded
    local_path = None
    if is_downloaded:
        local_path = str(get_model_path(info.repo_id))
    
    # Estimate VRAM for different resolutions (for image/video models)
    vram_estimates = {}
    if task_name in ("text-to-image", "image-to-image"):
        # Rough VRAM estimates for image generation
        # Base VRAM is model size + overhead
        base_vram = info.size_gb * 1.2  # 20% overhead for pipeline
        vram_estimates = {
            "512x512": f"{base_vram + 2:.1f} GB",
            "1024x1024": f"{base_vram + 4:.1f} GB",
            "2048x2048": f"{base_vram + 12:.1f} GB",
        }
    elif task_name in ("text-to-video", "image-to-video"):
        base_vram = info.size_gb * 1.2
        vram_estimates = {
            "480p (24 frames)": f"{base_vram + 6:.1f} GB",
            "720p (24 frames)": f"{base_vram + 12:.1f} GB",
            "1080p (24 frames)": f"{base_vram + 24:.1f} GB",
        }
    
    # Get recommended settings from metadata
    recommended_settings = info.metadata.copy() if info.metadata else {}
    
    # Generate HuggingFace URL
    hf_url = f"https://huggingface.co/{info.repo_id}"
    
    if as_json:
        # JSON output
        output = {
            "name": info.name,
            "short_name": short_name,
            "repo_id": info.repo_id,
            "task": task_name,
            "type": info.model_type.value,
            "size_gb": info.size_gb,
            "size_str": info.size_str,
            "is_default": info.is_default,
            "description": info.description,
            "status": status,
            "is_downloaded": is_downloaded,
            "local_path": local_path,
            "recommended_settings": recommended_settings,
            "vram_estimates": vram_estimates,
            "huggingface_url": hf_url,
        }
        
        if info.pip_dependencies:
            output["pip_dependencies"] = info.pip_dependencies
        
        click.echo(json_module.dumps(output, indent=2))
    else:
        # Text output
        click.echo("")
        click.echo(click.style(info.name, fg="cyan", bold=True))
        click.echo("=" * 60)
        click.echo("")
        
        click.echo(click.style("Basic Information", bold=True))
        click.echo(f"  Repository:     {info.repo_id}")
        click.echo(f"  Short Name:     {short_name}")
        click.echo(f"  Task:           {task_name}")
        click.echo(f"  Type:           {info.model_type.value}")
        click.echo(f"  Size:           {info.size_str}")
        click.echo(f"  Default:        {'Yes' if info.is_default else 'No'}")
        
        if info.description:
            click.echo(f"  Description:    {info.description}")
        
        click.echo("")
        
        # Download status
        click.echo(click.style("Download Status", bold=True))
        if is_downloaded:
            click.echo(click.style(f"  Status:         ✓ Downloaded", fg="green"))
            click.echo(f"  Location:       {local_path}")
        else:
            click.echo(click.style(f"  Status:         Not downloaded", fg="yellow"))
            click.echo(f"  To download:    hftool download -m {short_name}")
        
        click.echo("")
        
        # Recommended settings
        if recommended_settings:
            click.echo(click.style("Recommended Settings", bold=True))
            for key, value in recommended_settings.items():
                # Format key nicely
                display_key = key.replace("_", " ").title()
                click.echo(f"  {display_key + ':':<20} {value}")
            click.echo("")
        
        # VRAM estimates
        if vram_estimates:
            click.echo(click.style("VRAM Estimates", bold=True))
            for resolution, vram in vram_estimates.items():
                click.echo(f"  {resolution + ':':<20} {vram}")
            click.echo("")
        
        # Dependencies
        if info.pip_dependencies:
            click.echo(click.style("Dependencies", bold=True))
            for dep in info.pip_dependencies:
                click.echo(f"  - {dep}")
            click.echo("")
        
        # Links
        click.echo(click.style("Links", bold=True))
        click.echo(f"  HuggingFace:    {hf_url}")
        click.echo("")


# =============================================================================
# BENCHMARK COMMAND
# =============================================================================

@main.command("benchmark")
@click.option("--task", "-t", required=False, shell_complete=complete_tasks, help="Task to benchmark")
@click.option("--model", "-m", required=False, shell_complete=complete_models, help="Model to benchmark")
@click.option("--all", "benchmark_all", is_flag=True, help="Benchmark all downloaded models")
@click.option("--device", "-d", default="auto", shell_complete=complete_devices, help="Device to use")
@click.option("--dtype", default=None, shell_complete=complete_dtypes, help="Data type")
@click.option("--skip-large", is_flag=True, help="Skip models larger than 15GB")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def benchmark_command(
    ctx: click.Context,
    task: Optional[str],
    model: Optional[str],
    benchmark_all: bool,
    device: str,
    dtype: Optional[str],
    skip_large: bool,
    as_json: bool,
):
    """Benchmark model performance (load time, inference time, VRAM).
    
    \b
    Examples:
      hftool benchmark -t text-to-image -m z-image-turbo
      hftool benchmark -t asr -m whisper-large-v3
      hftool benchmark --all                    # Benchmark all downloaded models
      hftool benchmark --all --skip-large       # Skip models >15GB
    
    Results are cached in ~/.hftool/benchmarks.json for reference.
    """
    from hftool.core.benchmark import run_benchmark, save_benchmark, load_benchmarks, get_benchmarks_file
    from hftool.core.models import MODEL_REGISTRY, get_default_model_info
    from hftool.core.download import get_download_status
    import json as json_module
    
    verbose = ctx.obj.get("verbose", False) and not as_json
    
    if benchmark_all:
        # Benchmark all downloaded models
        click.echo("Benchmarking all downloaded models...")
        click.echo("")
        
        results = []
        for task_name, models in MODEL_REGISTRY.items():
            for short_name, info in models.items():
                status = get_download_status(info.repo_id)
                
                if status != "downloaded":
                    continue
                
                if skip_large and info.size_gb > 15:
                    if verbose:
                        click.echo(f"Skipping {info.name} ({info.size_str}) - too large")
                    continue
                
                click.echo(f"Benchmarking {info.name} ({task_name})...")
                
                result = run_benchmark(
                    task=task_name,
                    model=short_name,
                    device=device,
                    dtype=dtype,
                    verbose=verbose,
                )
                
                save_benchmark(result)
                results.append(result)
                click.echo("")
        
        if as_json:
            output = [
                {
                    "task": r.task,
                    "model": r.model,
                    "repo_id": r.repo_id,
                    "load_time": r.load_time,
                    "inference_time": r.inference_time,
                    "total_time": r.total_time,
                    "vram_peak": r.vram_peak,
                    "vram_allocated": r.vram_allocated,
                    "success": r.success,
                    "error": r.error,
                }
                for r in results
            ]
            click.echo(json_module.dumps(output, indent=2))
        else:
            click.echo("=" * 60)
            click.echo(f"Benchmarked {len(results)} models")
            click.echo(f"Results saved to: {get_benchmarks_file()}")
        
        return
    
    if not task or not model:
        click.echo("Error: Must specify --task and --model, or use --all", err=True)
        click.echo("", err=True)
        click.echo("Examples:", err=True)
        click.echo("  hftool benchmark -t text-to-image -m z-image-turbo", err=True)
        click.echo("  hftool benchmark --all", err=True)
        sys.exit(1)
    
    # Benchmark specific model
    result = run_benchmark(
        task=task,
        model=model,
        device=device,
        dtype=dtype,
        verbose=verbose,
    )
    
    # Save result
    save_benchmark(result)
    
    # Output result
    if as_json:
        output = {
            "task": result.task,
            "model": result.model,
            "repo_id": result.repo_id,
            "timestamp": result.timestamp,
            "device": result.device,
            "dtype": result.dtype,
            "load_time": result.load_time,
            "inference_time": result.inference_time,
            "total_time": result.total_time,
            "vram_peak": result.vram_peak,
            "vram_allocated": result.vram_allocated,
            "test_prompt": result.test_prompt,
            "test_params": result.test_params,
            "success": result.success,
            "error": result.error,
        }
        click.echo(json_module.dumps(output, indent=2))
    else:
        click.echo("")
        click.echo("=" * 60)
        click.echo("Benchmark Results")
        click.echo("=" * 60)
        click.echo(f"Task:            {result.task}")
        click.echo(f"Model:           {result.model}")
        click.echo(f"Device:          {result.device}")
        
        if result.success:
            click.echo(click.style(f"Status:          ✓ Success", fg="green"))
            click.echo("")
            click.echo(f"Load time:       {result.load_time:.2f}s")
            click.echo(f"Inference time:  {result.inference_time:.2f}s")
            click.echo(f"Total time:      {result.total_time:.2f}s")
            
            if result.vram_peak:
                click.echo("")
                click.echo(f"VRAM peak:       {result.vram_peak:.2f} GB")
                click.echo(f"VRAM allocated:  {result.vram_allocated:.2f} GB")
        else:
            click.echo(click.style(f"Status:          ✗ Failed", fg="red"))
            click.echo(f"Error:           {result.error}")
        
        click.echo("")
        click.echo(f"Results saved to: {get_benchmarks_file()}")


# =============================================================================
# COMPLETION COMMAND
# =============================================================================

@main.command("completion")
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]), required=False)
@click.option("--install", is_flag=True, help="Install completion for current shell")
@click.pass_context
def completion_command(ctx: click.Context, shell: Optional[str], install: bool):
    """Show or install shell completion scripts.
    
    \b
    Examples:
      hftool completion bash               # Show bash completion script
      hftool completion zsh                # Show zsh completion script
      hftool completion fish               # Show fish completion script
      hftool completion --install          # Auto-detect and install
      hftool completion bash --install     # Install bash completion
    
    \b
    After installation, restart your shell or run:
      source ~/.bashrc    # for bash
      source ~/.zshrc     # for zsh
      # fish completion loads automatically
    """
    from hftool.core.completion import (
        get_shell_name, 
        get_completion_script, 
        install_completion
    )
    
    # Auto-detect shell if not specified
    if shell is None:
        shell = get_shell_name()
        if shell is None:
            click.echo("Error: Could not detect shell. Please specify: bash, zsh, or fish", err=True)
            sys.exit(1)
    
    # Install completion
    if install:
        try:
            if install_completion(shell):
                click.echo(f"Completion installed for {shell}")
                click.echo("")
                click.echo("Restart your shell or run:")
                if shell == "bash":
                    click.echo("  source ~/.bashrc")
                elif shell == "zsh":
                    click.echo("  source ~/.zshrc")
                elif shell == "fish":
                    click.echo("  # fish completion loads automatically")
            else:
                click.echo(f"Completion already installed for {shell}")
        except Exception as e:
            click.echo(f"Error installing completion: {e}", err=True)
            sys.exit(1)
        return
    
    # Show completion script
    try:
        script = get_completion_script(shell)
        click.echo(script)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# =============================================================================
# DOCTOR COMMAND
# =============================================================================

@main.command("doctor")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def doctor_command(ctx: click.Context, as_json: bool):
    """Run system diagnostics and check hftool health.
    
    \b
    Checks:
      - Python version
      - PyTorch installation
      - GPU availability
      - ffmpeg (required for video/audio)
      - Network connectivity
      - Optional features
      - Configuration files
    
    \b
    Examples:
      hftool doctor              # Run all checks
      hftool doctor --json       # Output as JSON
    
    \b
    Exit codes:
      0 = All checks passed
      1 = Warnings found
      2 = Errors found
    """
    from hftool.core.doctor import run_doctor_checks, format_doctor_report
    import json as json_module
    
    # Run all checks
    report = run_doctor_checks()
    
    # Output results
    if as_json:
        output = report.to_dict()
        click.echo(json_module.dumps(output, indent=2))
    else:
        output = format_doctor_report(report, use_color=True)
        if output:  # Plain text fallback
            click.echo(output)
        # Otherwise rich already printed to console
    
    # Exit with appropriate code
    sys.exit(report.get_exit_code())


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
    input_data: Optional[str],
    output_file: Optional[str],
    device: str,
    dtype: Optional[str],
    seed: Optional[int],
    interactive: bool,
    verbose: bool,
    quiet: bool = False,
    output_json: bool = False,
    embed_metadata: bool = True,
    open_output: Optional[bool] = None,
    wizard_extra_kwargs: Optional[Dict[str, Any]] = None,
):
    """Execute a task (internal helper)."""
    import random
    import json as json_module
    
    # Parse extra arguments (after --)
    extra_args = ctx.obj.get("extra_args", ()) if ctx.obj else ()
    extra_kwargs = _parse_extra_args(list(extra_args))
    
    # Merge wizard extra_kwargs (from interactive mode) with CLI extra args
    # CLI args take priority over wizard params
    if wizard_extra_kwargs:
        extra_kwargs = {**wizard_extra_kwargs, **extra_kwargs}
    
    # Generate random seed if not provided
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    # Add seed to extra_kwargs (will be passed to model if supported)
    if "generator_seed" not in extra_kwargs and "seed" not in extra_kwargs:
        extra_kwargs["seed"] = seed
    
    # Quiet and JSON modes suppress verbose output
    if verbose and not quiet and not output_json:
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
        from hftool.core.config import Config
        
        # Resolve task alias
        resolved_task = TASK_ALIASES.get(task, task)
        
        # Handle batch mode
        batch_source = ctx.obj.get("batch") if ctx.obj else None
        batch_json_file = ctx.obj.get("batch_json") if ctx.obj else None
        batch_output_dir = ctx.obj.get("batch_output_dir") if ctx.obj else None
        
        if batch_source or batch_json_file:
            from hftool.core.batch import load_batch_inputs, load_batch_json, process_batch
            
            if not quiet and not output_json:
                click.echo("Running in batch mode...")
                click.echo("")
            
            # Load inputs
            if batch_json_file:
                # Load from JSON file
                batch_entries = load_batch_json(batch_json_file)
                
                # For JSON batch, we don't use the simple file list processing
                # Instead, each entry can have its own params
                if not quiet and not output_json:
                    click.echo(f"Loaded {len(batch_entries)} entries from JSON batch file")
                    click.echo("")
                
                # Process each entry with its own params
                results = []
                success_count = 0
                failure_count = 0
                
                for i, entry in enumerate(batch_entries):
                    entry_input = entry["input"]
                    entry_output = entry.get("output")
                    entry_params = entry.get("params", {})
                    
                    # Merge params (entry params override command-line params)
                    merged_kwargs = {**extra_kwargs, **entry_params}
                    
                    if not quiet and not output_json:
                        click.echo(f"[{i+1}/{len(batch_entries)}] Processing: {entry_input}")
                    
                    # Run single task
                    try:
                        _run_task_command(
                            ctx=ctx,
                            task=task,
                            model=model,
                            input_data=entry_input,
                            output_file=entry_output,
                            device=device,
                            dtype=dtype,
                            seed=seed,
                            interactive=False,
                            verbose=False,  # Suppress verbose for batch
                            quiet=True,  # Suppress output
                            output_json=False,
                            embed_metadata=embed_metadata,
                            open_output=False,  # Don't open files in batch
                        )
                        success_count += 1
                        if not quiet and not output_json:
                            click.echo(f"  ✓ Success")
                    except Exception as e:
                        failure_count += 1
                        if not quiet and not output_json:
                            click.echo(click.style(f"  ✗ Failed: {e}", fg="red"), err=True)
                
                # Print summary
                if not quiet and not output_json:
                    click.echo("")
                    click.echo("=" * 60)
                    click.echo(f"Batch processing complete: {success_count} succeeded, {failure_count} failed")
                elif output_json:
                    result_data = {
                        "success": True,
                        "batch_mode": "json",
                        "total": len(batch_entries),
                        "succeeded": success_count,
                        "failed": failure_count,
                    }
                    click.echo(json_module.dumps(result_data, indent=2))
                
                return
            
            else:
                # Load from file/directory
                inputs = load_batch_inputs(batch_source)
                
                if not inputs:
                    click.echo(f"No inputs found in: {batch_source}", err=True)
                    sys.exit(1)
                
                if not quiet and not output_json:
                    click.echo(f"Loaded {len(inputs)} inputs")
                    click.echo("")
                
                # Determine output extension based on task
                task_config = get_task_config(resolved_task)
                output_ext_map = {
                    "image": ".png",
                    "audio": ".wav",
                    "video": ".mp4",
                    "text": ".txt",
                }
                output_extension = output_ext_map.get(task_config.output_type, ".out")
                
                # Process batch
                results, success_count, failure_count = process_batch(
                    task=task,
                    inputs=inputs,
                    model=model,
                    device=device,
                    dtype=dtype,
                    output_dir=batch_output_dir,
                    output_extension=output_extension,
                    extra_kwargs=extra_kwargs,
                    verbose=not quiet and not output_json,
                )
                
                # Print summary
                if not quiet and not output_json:
                    click.echo("")
                    click.echo("=" * 60)
                    click.echo(f"Batch processing complete: {success_count} succeeded, {failure_count} failed")
                elif output_json:
                    result_data = {
                        "success": True,
                        "batch_mode": "file",
                        "total": len(inputs),
                        "succeeded": success_count,
                        "failed": failure_count,
                        "results": [
                            {
                                "input": r.input_file,
                                "output": r.output_file,
                                "success": r.success,
                                "error": r.error,
                                "execution_time": r.execution_time,
                            }
                            for r in results
                        ],
                    }
                    click.echo(json_module.dumps(result_data, indent=2))
                
                return
        
        # Load configuration early
        config = Config.get()
        
        # Apply config defaults if CLI args not provided
        # Device: use config if still "auto"
        if device == "auto":
            device = config.get_value("device", task=resolved_task, default="auto")
        
        # Dtype: use config if None
        if dtype is None:
            dtype = config.get_value("dtype", task=resolved_task, default=None)
        
        # Model: use config if None
        if model is None:
            model = config.get_value("model", task=resolved_task, default=None)
        
        # Resolve model alias if model is set
        if model:
            model = config.resolve_model_alias(model)
        
        # Merge config task-specific params (lower priority than extra_kwargs)
        # Get task-specific config section as dict
        task_params = {}
        if resolved_task in config._config:
            task_config_section = config._config[resolved_task]
            if isinstance(task_config_section, dict):
                # Extract only parameter-like keys (not 'model', 'device', 'dtype')
                reserved_keys = {'model', 'device', 'dtype'}
                task_params = {k: v for k, v in task_config_section.items() if k not in reserved_keys}
        
        # Merge: config params < extra_kwargs (CLI has priority)
        extra_kwargs = {**task_params, **extra_kwargs}
        
        # Handle interactive mode and file references
        if interactive or (input_data and input_data == "@?"):
            # Interactive JSON builder
            try:
                from hftool.io.interactive_input import build_interactive_input
                input_data = build_interactive_input(resolved_task)
            except ValueError as e:
                click.echo(f"Error: {e}", err=True)
                sys.exit(1)
        elif input_data and input_data.startswith("@"):
            # Resolve @ file reference
            try:
                from hftool.io.file_picker import resolve_file_reference
                input_data = resolve_file_reference(input_data, task=resolved_task)
                if verbose:
                    click.echo(f"Resolved file reference to: {input_data}")
            except ValueError as e:
                click.echo(f"Error resolving file reference: {e}", err=True)
                sys.exit(1)
        elif input_data is None and interactive:
            # Interactive mode but no schema available - build basic JSON
            try:
                from hftool.io.interactive_input import build_interactive_input
                input_data = build_interactive_input(resolved_task)
            except ValueError as e:
                # Fall back to text prompt
                try:
                    input_data = click.prompt("Enter input data")
                except click.Abort:
                    click.echo("Input cancelled", err=True)
                    sys.exit(1)
        
        # At this point input_data must be set
        if input_data is None:
            click.echo("Error: No input data provided", err=True)
            sys.exit(1)
        
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
        
        # Run the task (quiet mode suppresses progress bars)
        result = _run_task(
            task_name=resolved_task,
            task_config=task_config,
            model=model_to_load,
            input_data=input_data,
            output_file=output_file,
            device=device,
            dtype=dtype,
            verbose=verbose and not quiet and not output_json,
            **extra_kwargs
        )
        
        # Embed metadata in output file
        if output_file and embed_metadata and os.path.exists(output_file):
            from hftool.core.metadata import embed_metadata as do_embed_metadata
            
            # Extract prompt from input_data
            prompt = None
            if isinstance(input_data, str):
                # Try to parse JSON first
                try:
                    data = json_module.loads(input_data)
                    if isinstance(data, dict):
                        # Look for common prompt keys
                        prompt = data.get("prompt") or data.get("text") or data.get("caption")
                except json_module.JSONDecodeError:
                    # Not JSON, use as-is (but limit length)
                    prompt = input_data[:500] if len(input_data) > 500 else input_data
            
            do_embed_metadata(
                file_path=output_file,
                task=resolved_task,
                model=model or model_repo_id,
                prompt=prompt,
                seed=seed,
                extra_params=extra_kwargs,
                verbose=verbose and not quiet and not output_json,
            )
        
        # Print result summary based on output mode
        if output_json:
            # JSON output mode
            result_data = {
                "success": True,
                "task": resolved_task,
                "model": model or model_repo_id,
                "input": input_data,
                "output": output_file,
                "seed": seed,
                "device": device,
                "dtype": dtype,
            }
            
            # Add result text if available
            if isinstance(result, str):
                result_data["text"] = result
            elif isinstance(result, dict) and "text" in result:
                result_data["text"] = result["text"]
            
            click.echo(json_module.dumps(result_data, indent=2))
        elif quiet:
            # Quiet mode - only output file path
            if output_file:
                click.echo(output_file)
            elif isinstance(result, str):
                click.echo(result)
            elif isinstance(result, dict) and "text" in result:
                click.echo(result["text"])
        else:
            # Normal output mode
            if output_file:
                click.echo(f"Output saved to: {output_file}")
                
                # Show reproduction command
                if verbose or seed is not None:
                    repro_parts = ["hftool", "-t", resolved_task]
                    if model:
                        repro_parts.extend(["-m", model])
                    repro_parts.extend(["-i", f'"{input_data}"'])
                    if output_file:
                        repro_parts.extend(["-o", output_file])
                    if seed is not None:
                        repro_parts.extend(["--seed", str(seed)])
                    click.echo(f"Seed: {seed}")
                    click.echo(f"To reproduce: {' '.join(repro_parts)}")
                
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
        
        # Record to history (success)
        from hftool.core.history import History
        history = History.get()
        history.add(
            task=resolved_task,
            model=model,
            input_data=input_data,
            output_file=output_file,
            device=device,
            dtype=dtype,
            seed=seed,
            extra_args=extra_kwargs,
            success=True,
        )
        
    except SystemExit:
        raise
    except Exception as e:
        # Record to history (failure)
        from hftool.core.history import History
        history = History.get()
        history.add(
            task=task if 'resolved_task' not in locals() else resolved_task,
            model=model,
            input_data=input_data or "",
            output_file=output_file,
            device=device,
            dtype=dtype,
            seed=seed,
            extra_args=extra_kwargs if 'extra_kwargs' in locals() else {},
            success=False,
            error_message=str(e),
        )
        
        # Handle error output based on mode
        if output_json:
            error_data = {
                "success": False,
                "error": str(e),
                "task": task if 'resolved_task' not in locals() else resolved_task,
                "model": model,
                "input": input_data or "",
                "output": output_file,
            }
            click.echo(json_module.dumps(error_data, indent=2))
        else:
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
    from pathlib import Path
    
    # Security: Validate file path (M-3)
    try:
        path = Path(file_path).resolve()
        
        # Check file exists
        if not path.exists():
            click.echo(f"Cannot open file: {file_path} (file not found)", err=True)
            return False
        
        # Check it's a regular file (not a directory, symlink to dangerous location, etc.)
        if not path.is_file():
            click.echo(f"Cannot open file: {file_path} (not a regular file)", err=True)
            return False
        
        # Use the validated absolute path
        file_path = str(path)
        
    except Exception as e:
        click.echo(f"Cannot open file: invalid path ({e})", err=True)
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
