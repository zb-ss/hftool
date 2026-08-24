"""TUI command — launch the interactive terminal user interface.

By default, the TUI runs inside Docker for deterministic dependencies
(correct PyTorch/ROCm version, textual, etc.). Use --native to bypass Docker.
"""

import os
import sys

import click


@click.command("tui")
@click.option("--native", is_flag=True, help="Run TUI natively (skip Docker, requires local dependencies)")
@click.option("--gpu", "-g", default=None, envvar="HFTOOL_GPU",
              help="GPU(s) to use: 'auto', 'all', '0', '1', '0,1'")
def tui_command(native: bool, gpu: str | None):
    """Launch interactive TUI (Terminal User Interface).

    Runs inside Docker by default for consistent dependencies (PyTorch, ROCm,
    textual). All GPU setup is handled automatically.

    \b
    Examples:
      hftool tui                       # Launch TUI (via Docker)
      hftool tui --native              # Launch TUI natively (dev mode)
      hftool tui --gpu 1               # Use specific GPU
    """
    # If already inside Docker, launch TUI directly
    if os.environ.get("HFTOOL_IN_DOCKER"):
        _run_native_tui()
        return

    # Native mode — bypass Docker
    if native:
        _run_native_tui()
        return

    # Default: route through Docker
    _run_docker_tui(gpu)


def _run_native_tui():
    """Launch TUI directly in current environment."""
    try:
        from hftool.tui.app import HFToolApp
    except ImportError:
        click.echo("Error: Textual is required for the TUI.", err=True)
        click.echo("Install with: pip install textual", err=True)
        sys.exit(1)

    app = HFToolApp()
    app.run()


def _run_docker_tui(gpu: str | None):
    """Launch TUI inside Docker container."""
    from hftool.utils.docker import (
        detect_hardware, run_in_docker, GPUPlatform,
        parse_gpu_arg,
    )

    hw = detect_hardware()

    if not hw.docker_available:
        click.echo("Error: Docker is not installed or not running.", err=True)
        click.echo("")
        click.echo("Options:", err=True)
        click.echo("  1. Install Docker: https://docs.docker.com/get-docker/", err=True)
        click.echo("  2. Run natively:   hftool tui --native", err=True)
        sys.exit(1)

    if hw.platform == GPUPlatform.MPS:
        click.echo("Note: Docker GPU passthrough is not supported on Apple Silicon.", err=True)
        click.echo("Launching TUI natively instead...", err=True)
        click.echo("")
        _run_native_tui()
        return

    # Parse GPU selection
    gpu_indices = None
    if gpu:
        gpu_indices = parse_gpu_arg(gpu, hw.platform)
    # With no explicit --gpu, expose all cards to the TUI. Each task then uses
    # live free VRAM and its catalog minimum to choose one physical GPU. This
    # does not enable multi-GPU execution; that still requires --gpu all.

    # Launch TUI inside Docker — pass "tui --native" so the container
    # doesn't try to nest another Docker layer
    # The TUI is a full-screen Textual app — it always needs a pseudo-TTY.
    exit_code, _ = run_in_docker(
        ["tui", "--native"],
        hw,
        gpu_indices=gpu_indices,
        tty=True,
        multi_gpu=gpu == "all",
    )
    raise SystemExit(exit_code)
