"""Docker commands — status, setup, build, run."""

import os
import sys
from typing import Optional

import click

from hftool.cli.helpers import get_console


# =============================================================================
# DOCKER COMMAND GROUP
# =============================================================================

@click.group("docker", invoke_without_command=True, context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.pass_context
def docker_command(ctx: click.Context):
    """Manage Docker-based execution for isolated GPU environments.

    Docker mode provides:
    - Pre-configured ROCm 7.1.1 or CUDA environment
    - All dependencies pre-installed (no pip conflicts)
    - Isolated from system libraries (safe for gaming systems)

    \b
    Examples:
      hftool docker status          # Check Docker setup
      hftool docker setup           # Interactive setup wizard
      hftool docker run -- -t t2i -i "A cat" -o cat.png
    """
    if ctx.invoked_subcommand is None:
        # Show status by default
        ctx.invoke(docker_status)


@docker_command.command("status")
def docker_status():
    """Show Docker and hardware status."""
    from hftool.utils.docker import detect_hardware, get_docker_preference

    console = get_console()
    hw = detect_hardware()
    pref = get_docker_preference()

    console.print("\n[bold]Hardware Detection[/bold]")
    console.print(f"  Platform: {hw.platform.value.upper()}")
    if hw.gpu_name:
        console.print(f"  GPU: {hw.gpu_name}")
    console.print(f"  GPU Available: {'Yes' if hw.gpu_available else 'No'}")

    console.print("\n[bold]Docker Status[/bold]")
    console.print(f"  Docker Installed: {'Yes' if hw.docker_available else 'No'}")
    console.print(f"  Docker Compose: {'Yes' if hw.docker_compose_available else 'No'}")

    if hw.docker_available and hw.recommended_image:
        console.print(f"  Recommended Image: {hw.recommended_image}")
        console.print(f"  Image Available: {'Yes' if hw.image_available else 'No (run: hftool docker build)'}")

    console.print("\n[bold]Mode Preference[/bold]")
    if pref:
        console.print(f"  Current: {pref}")
    else:
        console.print("  Current: Not set (run: hftool docker setup)")

    if hw.platform.value == "mps":
        console.print("\n[yellow]Note:[/yellow] Apple Silicon works best with native installation.")
        console.print("Docker GPU passthrough is not well supported on macOS.")


@docker_command.command("setup")
@click.option("--mode", type=click.Choice(["docker", "native"]), help="Set mode without wizard")
def docker_setup(mode: Optional[str]):
    """Interactive setup wizard for Docker mode.

    Detects your hardware and helps configure the optimal execution environment.
    """
    from hftool.utils.docker import (
        detect_hardware, set_docker_preference, build_image, GPUPlatform
    )

    console = get_console()
    hw = detect_hardware()

    if mode:
        # Direct mode setting
        set_docker_preference(mode)
        console.print(f"Mode set to: [bold]{mode}[/bold]")
        return

    console.print("\n[bold cyan]hftool Setup Wizard[/bold cyan]\n")

    # Hardware detection
    console.print("[bold]Detecting hardware...[/bold]")
    if hw.gpu_available:
        console.print(f"  Found: [green]{hw.gpu_name}[/green] ({hw.platform.value.upper()})")
    else:
        console.print("  [yellow]No GPU detected - will use CPU[/yellow]")

    # Docker check
    if not hw.docker_available:
        console.print("\n[yellow]Docker is not installed.[/yellow]")
        console.print("For the easiest experience, install Docker:")
        console.print("  https://docs.docker.com/get-docker/")
        console.print("\nAlternatively, use native mode (manual dependency management).")

        if click.confirm("\nContinue with native mode?", default=True):
            set_docker_preference("native")
            console.print("\nMode set to: [bold]native[/bold]")
            _show_native_install_instructions(hw)
        return

    # MPS special case
    if hw.platform == GPUPlatform.MPS:
        console.print("\n[yellow]Note:[/yellow] Apple Silicon (MPS) works best with native installation.")
        console.print("Docker GPU passthrough is limited on macOS.")
        set_docker_preference("native")
        console.print("\nMode set to: [bold]native[/bold]")
        _show_native_install_instructions(hw)
        return

    # Offer choices
    console.print("\n[bold]Choose execution mode:[/bold]\n")

    console.print("  [1] [green]Docker (Recommended)[/green]")
    console.print("      - All dependencies pre-configured")
    console.print("      - ROCm 7.1.1 isolated from system")
    console.print("      - Won't affect gaming/other apps")
    console.print(f"      - Image size: ~15GB for {hw.platform.value}")
    console.print("")
    console.print("  [2] Native")
    console.print("      - Uses system Python environment")
    console.print("      - Manual dependency management")
    console.print("      - May conflict with other packages")
    console.print("")

    choice = click.prompt("Select", type=click.Choice(["1", "2"]), default="1")

    if choice == "1":
        set_docker_preference("docker")
        console.print("\nMode set to: [bold]docker[/bold]")

        # Build image if needed
        if not hw.image_available:
            console.print(f"\nBuilding {hw.recommended_image}...")
            console.print("This may take 10-15 minutes on first run.\n")

            if build_image(hw.platform):
                console.print(f"\n[green]Successfully built {hw.recommended_image}[/green]")
            else:
                console.print("\n[red]Failed to build image.[/red]")
                console.print("Try building manually:")
                console.print("  cd <hftool-repo>")
                console.print(f"  docker build -f docker/Dockerfile.{hw.platform.value} -t {hw.recommended_image} .")
        else:
            console.print(f"\nImage {hw.recommended_image} is ready!")

        console.print("\n[bold]You're all set![/bold]")
        console.print("Run hftool commands normally - they'll execute in Docker automatically.")
        console.print("\nExample:")
        console.print('  hftool -t t2i -i "A sunset over mountains" -o sunset.png')
    else:
        set_docker_preference("native")
        console.print("\nMode set to: [bold]native[/bold]")
        _show_native_install_instructions(hw)


def _show_native_install_instructions(hw):
    """Show platform-specific native installation instructions."""
    from hftool.utils.docker import GPUPlatform

    console = get_console()
    console.print("\n[bold]Native Installation Instructions:[/bold]\n")

    if hw.platform == GPUPlatform.ROCM:
        console.print("For AMD ROCm:")
        console.print("  pip install hftool[all]")
        console.print("  pip uninstall torch torchvision torchaudio -y")
        console.print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2")
        console.print("\nNote: System ROCm 7.1+ requires building PyTorch from source or using Docker.")

    elif hw.platform == GPUPlatform.CUDA:
        console.print("For NVIDIA CUDA:")
        console.print("  pip install hftool[all]")
        console.print("  pip install torch torchvision torchaudio")

    elif hw.platform == GPUPlatform.MPS:
        console.print("For Apple Silicon (MPS):")
        console.print("  pip install hftool[all]")
        console.print("  pip install torch torchvision torchaudio")

    else:
        console.print("For CPU-only:")
        console.print("  pip install hftool[all]")
        console.print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")


@docker_command.command("build")
@click.option("--platform", "-p", type=click.Choice(["rocm", "cuda", "cpu"]), help="Platform to build")
@click.option("--force", "-f", is_flag=True, help="Force rebuild even if image exists")
@click.option("--no-cache", is_flag=True, help="Disable Docker cache (rebuilds from scratch)")
def docker_build(platform: Optional[str], force: bool, no_cache: bool):
    """Build the hftool Docker image for your platform."""
    from hftool.utils.docker import detect_hardware, build_image, check_image_available, GPUPlatform

    console = get_console()
    hw = detect_hardware()

    # Determine platform
    if platform:
        target_platform = GPUPlatform(platform)
    else:
        target_platform = hw.platform
        if target_platform == GPUPlatform.MPS:
            console.print("[yellow]MPS platform doesn't use Docker. Building CPU image instead.[/yellow]")
            target_platform = GPUPlatform.CPU

    image_name = f"hftool:{target_platform.value}"

    # Check if already exists
    if not force and check_image_available(image_name):
        console.print(f"Image {image_name} already exists.")
        if not click.confirm("Rebuild?"):
            return

    console.print(f"Building {image_name}...")
    if no_cache:
        console.print("[yellow]Building without cache (this will take longer)[/yellow]")
    console.print("This may take 10-15 minutes.\n")

    if build_image(target_platform, no_cache=no_cache):
        console.print(f"\n[green]Successfully built {image_name}[/green]")
    else:
        console.print(f"\n[red]Failed to build {image_name}[/red]")
        console.print("\nMake sure you're in the hftool source directory with docker/ folder.")
        raise SystemExit(1)


@docker_command.command("run", context_settings=dict(
    ignore_unknown_options=True,
))
@click.option("--gpu", "-g", default=None, envvar="HFTOOL_GPU",
              help="GPU(s) to use: 'auto', 'all', '0', '1', '0,1' (passed to container)")
@click.argument("hftool_args", nargs=-1, required=False)
@click.pass_context
def docker_run(ctx: click.Context, gpu: Optional[str], hftool_args: tuple):
    """Run hftool command inside Docker container.

    Pass hftool arguments after the run command (use -- to separate if needed).
    Uses device passthrough to pass only selected GPU(s) to the container.

    \b
    Examples:
      hftool docker run -t t2i -i "A cat" -o cat.png
      hftool docker run --gpu 1 -- -t t2v -i "A cat" -o cat.mp4
      hftool docker run --gpu auto -- -t t2i -i "A cat" -o cat.png
      hftool docker run --help
    """
    import subprocess
    import platform
    from hftool.utils.docker import (
        detect_hardware, run_in_docker, GPUPlatform,
        parse_gpu_arg, interactive_gpu_select, list_amd_gpus
    )

    hw = detect_hardware()

    if not hw.docker_available:
        click.echo("Error: Docker is not installed.", err=True)
        raise SystemExit(1)

    # Combine explicit args, context args, and any args after `--` (which the
    # top-level group strips out of sys.argv into ctx.obj["extra_args"] before
    # Click parses). Resolve this before GPU selection so auto mode can account
    # for the selected catalog model's minimum VRAM.
    extra_args = list(ctx.obj.get("extra_args", ())) if ctx.obj else []
    args = list(hftool_args) + list(ctx.args) + extra_args
    if not args:
        args = ["--help"]

    from hftool.utils.docker import get_required_vram_from_args

    required_vram_gb = get_required_vram_from_args(args)

    # Parse GPU selection
    gpu_indices = None
    if gpu:
        # Explicit GPU specified via --gpu (or HFTOOL_GPU env var)
        gpu_indices = parse_gpu_arg(gpu, hw.platform, required_vram_gb)
    elif hw.platform == GPUPlatform.ROCM:
        gpus = list_amd_gpus()
        if sys.stdin.isatty() and len(gpus) > 1:
            # Interactive terminal with multiple AMD GPUs: offer a picker
            gpu_indices = interactive_gpu_select(hw.platform)
            if gpu_indices is None:
                # User cancelled
                raise SystemExit(0)
        elif len(gpus) == 1:
            # Single GPU - use it
            gpu_indices = [gpus[0].index]
        elif len(gpus) > 1:
            # Headless/CI with multiple GPUs and no selection: never block on a
            # prompt — auto-pick the best non-display GPU instead.
            gpu_indices = parse_gpu_arg("auto", hw.platform, required_vram_gb)

    # Check for --no-open flag
    no_open = "--no-open" in args

    # run_in_docker handles path translation and returns the host output path
    exit_code, host_path = run_in_docker(
        list(args),
        hw,
        gpu_indices=gpu_indices,
        multi_gpu=gpu == "all",
    )

    # Handle output file (auto-open, fix permissions)
    if exit_code == 0 and host_path:

        if os.path.exists(host_path):
            # Fix ownership to current user (Docker creates as root)
            try:
                uid = os.getuid()
                gid = os.getgid()
                file_stat = os.stat(host_path)
                # Only chown if owned by root (uid 0)
                if file_stat.st_uid == 0:
                    # Try chown directly first (works if user has sudo NOPASSWD)
                    subprocess.run(
                        ["sudo", "-n", "chown", f"{uid}:{gid}", host_path],
                        capture_output=True,
                        timeout=5,
                    )
            except Exception:
                pass  # Silently fail - user can fix manually

            # Auto-open output file on host
            if not no_open:
                try:
                    system = platform.system()
                    if system == "Darwin":
                        subprocess.run(["open", host_path], check=False)
                    elif system == "Linux":
                        subprocess.run(["xdg-open", host_path], check=False,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    elif system == "Windows":
                        os.startfile(host_path)
                except Exception:
                    pass  # Silently fail if can't open

    raise SystemExit(exit_code)


# =============================================================================
# FIRST-RUN DETECTION
# =============================================================================

def check_first_run_setup():
    """Check if this is a first run and Docker setup is needed.

    Returns True if the user chose Docker mode and setup succeeded,
    False if native mode was chosen or setup failed.
    """
    from hftool.utils.docker import get_docker_preference

    pref = get_docker_preference()
    if pref is not None:
        # Already configured
        return pref == "docker"

    # First run - only prompt if running a task that needs GPU
    # For now, just return False to use native mode
    # The user can run 'hftool docker setup' to configure
    return False
