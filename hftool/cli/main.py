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

# =============================================================================
# GPU Selection Re-exec
# =============================================================================
# CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES must be set BEFORE torch is imported.
# Since torch may be imported by other modules, we parse --gpu early and re-exec
# if needed, so the env vars are set before any Python code runs.
#
# This is skipped in Docker (HFTOOL_IN_DOCKER=1) because Docker controls GPU
# visibility through device passthrough.

def _gpu_reexec_if_needed():
    """Check for --gpu argument and re-exec with GPU env vars if needed."""
    # Skip in Docker - Docker handles GPU visibility
    if os.environ.get("HFTOOL_IN_DOCKER"):
        return

    # Skip if we already re-exec'd (prevent infinite loop)
    if os.environ.get("_HFTOOL_GPU_REEXEC"):
        return

    # Parse --gpu from sys.argv (before click parses it)
    gpu_value = None
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--gpu" and i + 1 < len(args):
            gpu_value = args[i + 1]
            break
        elif arg.startswith("--gpu="):
            gpu_value = arg.split("=", 1)[1]
            break

    # No --gpu specified or auto mode - let normal flow handle it
    if not gpu_value or gpu_value == "auto":
        return

    # Parse GPU indices
    if gpu_value == "all":
        # "all" mode - don't restrict GPUs, but set multi-GPU flag
        os.environ["HFTOOL_MULTI_GPU"] = "1"
        os.environ["_HFTOOL_GPU_REEXEC"] = "1"
        # Re-exec using -m hftool to ensure correct module loading
        os.execv(sys.executable, [sys.executable, "-m", "hftool"] + sys.argv[1:])

    # Specific GPU index(es) like "1" or "0,2"
    try:
        gpu_indices = [int(x.strip()) for x in gpu_value.split(",")]
    except ValueError:
        return  # Invalid GPU spec, let click handle the error

    # Determine if ROCm or CUDA
    is_rocm = (
        os.path.exists("/opt/rocm") or
        os.environ.get("ROCM_PATH") or
        os.environ.get("HIP_PATH") or
        os.environ.get("HSA_OVERRIDE_GFX_VERSION")  # Common ROCm env var
    )

    # Set the appropriate environment variable
    visible_devices = ",".join(str(i) for i in gpu_indices)
    if is_rocm:
        os.environ["HIP_VISIBLE_DEVICES"] = visible_devices
        os.environ["ROCR_VISIBLE_DEVICES"] = visible_devices
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    # Set multi-GPU flag if multiple GPUs selected
    if len(gpu_indices) > 1:
        os.environ["HFTOOL_MULTI_GPU"] = "1"

    # Mark that we've done the re-exec and re-launch
    os.environ["_HFTOOL_GPU_REEXEC"] = "1"
    # Re-exec using -m hftool to ensure correct module loading
    os.execv(sys.executable, [sys.executable, "-m", "hftool"] + sys.argv[1:])

# Run the re-exec check immediately
_gpu_reexec_if_needed()

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
@click.option("--gpu", "-g", default=None, envvar="HFTOOL_GPU", help="GPU(s) to use: 'auto' (smart default), 'all', '0', '1', '0,1' (multi-GPU)")
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
    gpu: Optional[str],
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
    from hftool.cli.helpers import list_tasks_display
    from hftool.cli.commands.setup import ensure_pytorch_ready

    # Store options in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["open"] = open
    ctx.obj["seed"] = seed
    ctx.obj["gpu"] = gpu
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
        list_tasks_display()
        return

    # Handle --interactive-wizard / -I (full wizard mode)
    if wizard:
        from hftool.io.interactive_mode import run_interactive_mode, check_interactive_mode
        from hftool.cli.commands.task import run_task_command

        # Ensure PyTorch is installed before running wizard
        if not ensure_pytorch_ready():
            sys.exit(1)

        try:
            params = run_interactive_mode(quiet=quiet, output_json=output_json)

            # Run the task with wizard parameters
            run_task_command(
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
                gpu=params.get("gpu", gpu),
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
        from hftool.cli.commands.task import run_task_command

        # Handle interactive mode or missing input
        if input_data is None and not interactive:
            click.echo("Error: Missing option '--input' / '-i' (or use --interactive).", err=True)
            sys.exit(1)

        # Ensure PyTorch is installed before running tasks
        if not ensure_pytorch_ready():
            sys.exit(1)

        run_task_command(
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
            gpu=gpu,
        )


# =============================================================================
# Register all subcommands
# =============================================================================
# Import command objects from modules and add them to the main group.
# This avoids circular imports since command modules don't import main.

from hftool.cli.commands.setup import setup_command  # noqa: E402
from hftool.cli.commands.task import run_command  # noqa: E402
from hftool.cli.commands.history import history_command  # noqa: E402
from hftool.cli.commands.models import (  # noqa: E402
    models_command,
    download_command,
    status_command,
    info_command,
    clean_command,
)
from hftool.cli.commands.docker import docker_command  # noqa: E402
from hftool.cli.commands.tools import (  # noqa: E402
    benchmark_command,
    completion_command,
    doctor_command,
)
from hftool.cli.commands.voiceover import voiceover_command  # noqa: E402

main.add_command(setup_command)
main.add_command(run_command)
main.add_command(history_command)
main.add_command(models_command)
main.add_command(download_command)
main.add_command(status_command)
main.add_command(info_command)
main.add_command(clean_command)
main.add_command(docker_command)
main.add_command(benchmark_command)
main.add_command(completion_command)
main.add_command(doctor_command)
main.add_command(voiceover_command)


if __name__ == "__main__":
    main()
