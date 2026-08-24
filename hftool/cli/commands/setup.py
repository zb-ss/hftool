"""Setup command and PyTorch detection helpers."""

import sys

import click


# =============================================================================
# PyTorch Setup Check
# =============================================================================

def check_pytorch_setup() -> dict:
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


def detect_system_gpu() -> dict:
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


def run_interactive_setup() -> bool:
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
    pytorch_status = check_pytorch_setup()
    system_gpu = detect_system_gpu()

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
        click.echo("Detected pipx environment.")

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

        subprocess.run(
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


def ensure_pytorch_ready() -> bool:
    """Ensure PyTorch is installed and ready. Run setup wizard if needed.

    Returns:
        True if PyTorch is ready, False otherwise
    """
    status = check_pytorch_setup()

    if status["installed"]:
        return True

    # PyTorch not installed - run interactive setup
    click.echo("")
    click.echo(click.style("PyTorch is required but not installed.", fg="yellow"))
    click.echo("")

    if click.confirm("Would you like to run the interactive setup wizard?", default=True):
        return run_interactive_setup()
    else:
        click.echo("")
        click.echo("Install PyTorch manually:")
        click.echo("  # NVIDIA: pip install torch torchvision torchaudio")
        click.echo("  # AMD ROCm: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2")
        click.echo("  # CPU: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
        click.echo("")
        return False


# =============================================================================
# SETUP COMMAND (click command - registered by main.py)
# =============================================================================

@click.command("setup")
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
    run_interactive_setup()
