"""Docker integration for hftool.

Provides seamless container-based execution with automatic hardware detection
and first-run setup experience.
"""

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple


class GPUPlatform(Enum):
    """Supported GPU platforms."""
    ROCM = "rocm"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon - native only, no Docker
    CPU = "cpu"


@dataclass
class HardwareInfo:
    """Detected hardware information."""
    platform: GPUPlatform
    gpu_name: Optional[str]
    gpu_available: bool
    docker_available: bool
    docker_compose_available: bool
    image_available: bool
    recommended_image: str

    @property
    def can_use_docker(self) -> bool:
        """Check if Docker execution is possible."""
        # MPS (Apple Silicon) doesn't work well with Docker for GPU
        return self.docker_available and self.platform != GPUPlatform.MPS


def detect_rocm_gpu() -> Tuple[bool, Optional[str]]:
    """Detect AMD ROCm GPU without importing torch.

    Returns:
        Tuple of (gpu_found, gpu_name)
    """
    # Check if /dev/kfd exists (ROCm compute interface)
    if not os.path.exists("/dev/kfd"):
        return False, None

    # Try to get GPU name from rocm-smi
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Parse output for GPU name
            for line in result.stdout.splitlines():
                if "Card series" in line or "GPU" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        return True, parts[-1].strip()
            # If we found /dev/kfd but can't parse name
            return True, "AMD GPU (unknown model)"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: check /sys for AMD GPU
    try:
        drm_path = Path("/sys/class/drm")
        if drm_path.exists():
            for card in drm_path.iterdir():
                device_path = card / "device" / "vendor"
                if device_path.exists():
                    vendor = device_path.read_text().strip()
                    if vendor == "0x1002":  # AMD vendor ID
                        # Try to get name from uevent
                        uevent_path = card / "device" / "uevent"
                        if uevent_path.exists():
                            uevent = uevent_path.read_text()
                            for line in uevent.splitlines():
                                if line.startswith("PCI_ID="):
                                    return True, f"AMD GPU ({line.split('=')[1]})"
                        return True, "AMD GPU"
    except Exception:
        pass

    # /dev/kfd exists but couldn't identify GPU
    return os.path.exists("/dev/kfd"), "AMD GPU (ROCm device detected)"


def detect_nvidia_gpu() -> Tuple[bool, Optional[str]]:
    """Detect NVIDIA GPU without importing torch.

    Returns:
        Tuple of (gpu_found, gpu_name)
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            gpu_name = result.stdout.strip().split("\n")[0]
            return True, gpu_name
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return False, None


def detect_apple_silicon() -> Tuple[bool, Optional[str]]:
    """Detect Apple Silicon Mac.

    Returns:
        Tuple of (is_apple_silicon, chip_name)
    """
    if sys.platform != "darwin":
        return False, None

    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            chip = result.stdout.strip()
            if "Apple" in chip:
                return True, chip
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return False, None


def check_docker() -> Tuple[bool, bool]:
    """Check if Docker and Docker Compose are available.

    Returns:
        Tuple of (docker_available, compose_available)
    """
    docker_ok = shutil.which("docker") is not None
    compose_ok = False

    if docker_ok:
        # Check if docker compose (v2) is available
        try:
            result = subprocess.run(
                ["docker", "compose", "version"],
                capture_output=True,
                timeout=5,
            )
            compose_ok = result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            pass

    return docker_ok, compose_ok


def check_image_available(image_name: str) -> bool:
    """Check if a Docker image is available locally.

    Args:
        image_name: Docker image name (e.g., "hftool:rocm")

    Returns:
        True if image exists locally
    """
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image_name],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def detect_hardware() -> HardwareInfo:
    """Detect available hardware and Docker setup.

    Returns:
        HardwareInfo with detected configuration
    """
    docker_ok, compose_ok = check_docker()

    # Check for ROCm first (AMD-first priority)
    rocm_found, rocm_name = detect_rocm_gpu()
    if rocm_found:
        image = "hftool:rocm"
        return HardwareInfo(
            platform=GPUPlatform.ROCM,
            gpu_name=rocm_name,
            gpu_available=True,
            docker_available=docker_ok,
            docker_compose_available=compose_ok,
            image_available=check_image_available(image) if docker_ok else False,
            recommended_image=image,
        )

    # Check for NVIDIA
    nvidia_found, nvidia_name = detect_nvidia_gpu()
    if nvidia_found:
        image = "hftool:cuda"
        return HardwareInfo(
            platform=GPUPlatform.CUDA,
            gpu_name=nvidia_name,
            gpu_available=True,
            docker_available=docker_ok,
            docker_compose_available=compose_ok,
            image_available=check_image_available(image) if docker_ok else False,
            recommended_image=image,
        )

    # Check for Apple Silicon
    apple_found, apple_name = detect_apple_silicon()
    if apple_found:
        return HardwareInfo(
            platform=GPUPlatform.MPS,
            gpu_name=apple_name,
            gpu_available=True,
            docker_available=docker_ok,
            docker_compose_available=compose_ok,
            image_available=False,  # No Docker for MPS
            recommended_image="",  # Native only
        )

    # CPU fallback
    image = "hftool:cpu"
    return HardwareInfo(
        platform=GPUPlatform.CPU,
        gpu_name=None,
        gpu_available=False,
        docker_available=docker_ok,
        docker_compose_available=compose_ok,
        image_available=check_image_available(image) if docker_ok else False,
        recommended_image=image,
    )


def get_docker_run_command(
    hardware: HardwareInfo,
    hftool_args: List[str],
    workdir: Optional[str] = None,
    hf_token: Optional[str] = None,
    extra_volumes: Optional[List[str]] = None,
    gpu_indices: Optional[List[int]] = None,
    mount_home: bool = True,
) -> List[str]:
    """Build the docker run command for the detected hardware.

    Args:
        hardware: Detected hardware info
        hftool_args: Arguments to pass to hftool inside container
        workdir: Working directory (default: current directory)
        hf_token: HuggingFace token (optional)
        extra_volumes: Additional volume mounts
        gpu_indices: Specific GPU indices to use (None = all GPUs)
        mount_home: Mount user's home directory for file browsing (default: True)

    Returns:
        List of command arguments for subprocess
    """
    workdir = workdir or os.getcwd()
    user_home = os.path.expanduser("~")
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hftool_config = os.environ.get("HFTOOL_CONFIG", os.path.expanduser("~/.hftool"))

    # Support custom model directory (e.g., HFTOOL_MODELS_DIR=/data3/.hftool/models/)
    models_dir = os.environ.get("HFTOOL_MODELS_DIR")
    if models_dir:
        models_dir = os.path.expanduser(models_dir)

    cmd = ["docker", "run", "--rm", "-it"]

    # Run as current user to avoid permission issues with created files
    # This ensures output files are owned by the host user, not root
    try:
        uid = os.getuid()
        gid = os.getgid()
        cmd.extend(["--user", f"{uid}:{gid}"])
    except (AttributeError, OSError):
        pass  # Windows doesn't have getuid/getgid

    # Platform-specific GPU flags with optional GPU selection
    if hardware.platform == GPUPlatform.ROCM:
        # For ROCm, /dev/kfd is always shared but GPUs are restricted via env vars
        cmd.extend([
            "--device=/dev/kfd",
            "--device=/dev/dri",
            "--security-opt", "seccomp=unconfined",
            "--group-add", "video",
            "--group-add", "render",
        ])
        # Pass through HSA_OVERRIDE_GFX_VERSION if set
        gfx_version = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
        if gfx_version:
            cmd.extend(["-e", f"HSA_OVERRIDE_GFX_VERSION={gfx_version}"])
        # GPU isolation via HIP_VISIBLE_DEVICES and ROCR_VISIBLE_DEVICES
        if gpu_indices:
            visible = ",".join(str(i) for i in gpu_indices)
            cmd.extend(["-e", f"HIP_VISIBLE_DEVICES={visible}"])
            cmd.extend(["-e", f"ROCR_VISIBLE_DEVICES={visible}"])

    elif hardware.platform == GPUPlatform.CUDA:
        # For NVIDIA, use --gpus flag with device selection
        if gpu_indices:
            device_str = ",".join(str(i) for i in gpu_indices)
            cmd.extend(["--gpus", f'"device={device_str}"'])
            cmd.extend(["-e", f"CUDA_VISIBLE_DEVICES={device_str}"])
        else:
            cmd.extend(["--gpus", "all"])

    # Shared memory for PyTorch DataLoader
    cmd.extend(["--shm-size", "16g"])

    # Volume mounts (use /data paths that work for any user)
    cmd.extend([
        "-v", f"{hf_home}:/data/huggingface",
        "-v", f"{hftool_config}:/data/hftool",
        "-v", f"{workdir}:/workspace",
    ])

    # Mount user's home directory for file browsing in interactive mode
    # This allows the file picker to access files outside the working directory
    if mount_home:
        cmd.extend(["-v", f"{user_home}:/home/host:ro"])  # Read-only for safety
        cmd.extend(["-e", f"HFTOOL_HOST_HOME=/home/host"])
        cmd.extend(["-e", f"HFTOOL_REAL_HOME={user_home}"])

    # Mark that we're running in Docker (for file picker path handling)
    cmd.extend(["-e", "HFTOOL_IN_DOCKER=1"])

    # Tell tools where to find their data (works for any user, including non-root)
    cmd.extend([
        "-e", "HF_HOME=/data/huggingface",
        "-e", "HFTOOL_CONFIG=/data/hftool",
    ])

    # Mount custom models directory if set
    if models_dir:
        # Mount to a predictable path inside container
        cmd.extend(["-v", f"{models_dir}:/models"])
        # Tell hftool inside container to use this path
        cmd.extend(["-e", "HFTOOL_MODELS_DIR=/models"])

    if extra_volumes:
        for vol in extra_volumes:
            cmd.extend(["-v", vol])

    # Environment variables
    cmd.extend(["-e", "HFTOOL_AUTO_DOWNLOAD=1"])

    if hf_token or os.environ.get("HF_TOKEN"):
        token = hf_token or os.environ.get("HF_TOKEN")
        cmd.extend(["-e", f"HF_TOKEN={token}"])

    # Pass through HuggingFace Hub settings (from ~/.hftool/.env or environment)
    hf_passthrough_vars = [
        "HF_HUB_ENABLE_HF_TRANSFER",  # Disable xet downloads if set to 0
        "HF_HUB_DISABLE_PROGRESS_BARS",
        "HF_HUB_DISABLE_SYMLINKS_WARNING",
        "HF_HUB_OFFLINE",
        "HUGGINGFACE_HUB_CACHE",
    ]
    for var in hf_passthrough_vars:
        value = os.environ.get(var)
        if value:
            cmd.extend(["-e", f"{var}={value}"])

    # Pass through debug and logging settings
    if os.environ.get("HFTOOL_DEBUG"):
        cmd.extend(["-e", f"HFTOOL_DEBUG={os.environ['HFTOOL_DEBUG']}"])

    if os.environ.get("HFTOOL_LOG_FILE"):
        # Log file needs special handling - mount the directory and map the path
        log_file = os.path.expanduser(os.environ["HFTOOL_LOG_FILE"])
        log_dir = os.path.dirname(log_file)
        log_name = os.path.basename(log_file)
        if log_dir:
            cmd.extend(["-v", f"{log_dir}:/var/log/hftool"])
            cmd.extend(["-e", f"HFTOOL_LOG_FILE=/var/log/hftool/{log_name}"])

    # Working directory
    cmd.extend(["-w", "/workspace"])

    # Image and command
    cmd.append(hardware.recommended_image)
    cmd.extend(hftool_args)

    return cmd


def pull_image(image_name: str, quiet: bool = False) -> bool:
    """Pull a Docker image.

    Args:
        image_name: Image to pull
        quiet: Suppress output

    Returns:
        True if successful
    """
    cmd = ["docker", "pull", image_name]

    try:
        if quiet:
            result = subprocess.run(cmd, capture_output=True)
            return result.returncode == 0
        else:
            result = subprocess.run(cmd)
            return result.returncode == 0
    except Exception:
        return False


def build_image(
    platform: GPUPlatform,
    project_root: Optional[str] = None,
    quiet: bool = False,
) -> bool:
    """Build the hftool Docker image for a platform.

    Args:
        platform: Target platform (rocm, cuda, cpu)
        project_root: Path to hftool project (for Dockerfile)
        quiet: Suppress output

    Returns:
        True if successful
    """
    if project_root is None:
        # Try to find docker directory relative to this file
        project_root = str(Path(__file__).parent.parent.parent)

    dockerfile = f"docker/Dockerfile.{platform.value}"
    dockerfile_path = Path(project_root) / dockerfile

    if not dockerfile_path.exists():
        return False

    # Get version from hftool package
    try:
        from hftool import __version__
        version = __version__
    except ImportError:
        version = "0.6.0"

    image_name = f"hftool:{platform.value}"
    cmd = [
        "docker", "build",
        "-f", dockerfile,
        "-t", image_name,
        "--build-arg", f"HFTOOL_VERSION={version}",
        ".",
    ]

    try:
        kwargs = {"cwd": project_root}
        if quiet:
            kwargs["capture_output"] = True

        result = subprocess.run(cmd, **kwargs)
        return result.returncode == 0
    except Exception:
        return False


def run_in_docker(
    hftool_args: List[str],
    hardware: Optional[HardwareInfo] = None,
    auto_setup: bool = True,
    gpu_indices: Optional[List[int]] = None,
) -> int:
    """Run hftool command in Docker container.

    Args:
        hftool_args: Arguments to pass to hftool
        hardware: Pre-detected hardware (will detect if None)
        auto_setup: Automatically build/pull image if missing
        gpu_indices: Specific GPU indices to use (None = all GPUs)

    Returns:
        Exit code from the container
    """
    if hardware is None:
        hardware = detect_hardware()

    if not hardware.docker_available:
        print("Error: Docker is not installed or not running.")
        print("Install Docker: https://docs.docker.com/get-docker/")
        return 1

    if hardware.platform == GPUPlatform.MPS:
        print("Note: Docker GPU passthrough is not supported on Apple Silicon.")
        print("Running natively is recommended for MPS devices.")
        return 1

    # Build or pull image if needed
    if not hardware.image_available and auto_setup:
        print(f"Building {hardware.recommended_image}...")
        # Try to find project root for building
        project_root = Path(__file__).parent.parent.parent
        if (project_root / "docker").exists():
            if not build_image(hardware.platform, str(project_root)):
                print(f"Failed to build {hardware.recommended_image}")
                return 1
        else:
            # Can't build locally, would need to pull from registry
            print(f"Image {hardware.recommended_image} not found.")
            print("Please build the image first:")
            print(f"  cd <hftool-repo> && docker build -f docker/Dockerfile.{hardware.platform.value} -t {hardware.recommended_image} .")
            return 1

    cmd = get_docker_run_command(hardware, hftool_args, gpu_indices=gpu_indices)

    try:
        result = subprocess.run(cmd)
        return result.returncode
    except KeyboardInterrupt:
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        print(f"Error running Docker: {e}")
        return 1


# Configuration file for Docker mode preference
def _get_config_dir() -> Path:
    """Get the hftool config directory, respecting HFTOOL_CONFIG env var."""
    config_dir = os.environ.get("HFTOOL_CONFIG")
    if config_dir:
        return Path(config_dir)
    return Path.home() / ".hftool"

def _get_docker_config_file() -> Path:
    """Get the Docker preference config file path."""
    return _get_config_dir() / "docker.conf"


def get_docker_preference() -> Optional[str]:
    """Get the user's Docker mode preference.

    Returns:
        "docker", "native", or None if not set
    """
    config_file = _get_docker_config_file()
    if config_file.exists():
        try:
            content = config_file.read_text().strip()
            if content in ("docker", "native"):
                return content
        except Exception:
            pass
    return None


def set_docker_preference(mode: str) -> None:
    """Set the user's Docker mode preference.

    Args:
        mode: "docker" or "native"
    """
    config_dir = _get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    _get_docker_config_file().write_text(mode)


def should_use_docker(hardware: Optional[HardwareInfo] = None) -> bool:
    """Determine if Docker should be used based on preference and hardware.

    Args:
        hardware: Pre-detected hardware (will detect if None)

    Returns:
        True if Docker should be used
    """
    pref = get_docker_preference()

    if pref == "native":
        return False

    if pref == "docker":
        if hardware is None:
            hardware = detect_hardware()
        return hardware.can_use_docker

    # No preference set - will trigger first-run wizard
    return False
