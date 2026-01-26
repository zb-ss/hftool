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


@dataclass
class GPUInfo:
    """Information about a detected GPU."""
    index: int
    name: str
    render_device: str  # e.g., /dev/dri/renderD128
    card_device: str  # e.g., /dev/dri/card0
    vram_gb: Optional[float] = None
    is_display_gpu: bool = False


def list_amd_gpus() -> List[GPUInfo]:
    """Detect all AMD GPUs and their render device paths.

    Uses /sys/class/drm to enumerate GPUs and map them to /dev/dri/renderD* devices.
    The mapping is done by matching PCI device paths between card* and renderD* entries.

    Returns:
        List of GPUInfo for each detected AMD GPU
    """
    gpus = []
    drm_path = Path("/sys/class/drm")

    if not drm_path.exists():
        return gpus

    # Build a map of PCI device path -> renderD* device
    render_devices_map = {}
    for entry in drm_path.iterdir():
        if entry.name.startswith("renderD"):
            try:
                pci_path = (entry / "device").resolve()
                render_devices_map[str(pci_path)] = f"/dev/dri/{entry.name}"
            except Exception:
                continue

    # Find all card* entries (not renderD* or card*-connectors)
    card_entries = sorted([
        d for d in drm_path.iterdir()
        if d.name.startswith("card") and d.name[4:].isdigit()
    ], key=lambda x: int(x.name[4:]))

    gpu_index = 0  # Assign sequential indices for user-facing selection
    for card_dir in card_entries:
        device_path = card_dir / "device"
        vendor_path = device_path / "vendor"

        # Check if this is an AMD device (vendor 0x1002)
        if not vendor_path.exists():
            continue

        try:
            vendor = vendor_path.read_text().strip()
            if vendor != "0x1002":
                continue
        except Exception:
            continue

        card_num = int(card_dir.name[4:])
        card_device = f"/dev/dri/{card_dir.name}"

        # Find the corresponding renderD* device by matching PCI device path
        try:
            pci_path = str(device_path.resolve())
            render_device = render_devices_map.get(pci_path)
        except Exception:
            render_device = None

        # Verify the render device exists
        if not render_device or not os.path.exists(render_device):
            continue

        # Try to get GPU name
        gpu_name = f"AMD GPU {gpu_index}"
        try:
            # Try rocm-smi first for better names (use gpu_index as rocm-smi device number)
            result = subprocess.run(
                ["rocm-smi", "-d", str(gpu_index), "--showproductname"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "Card series" in line:
                        parts = line.split(":")
                        if len(parts) >= 2:
                            gpu_name = parts[-1].strip()
                            break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # Fallback: try to read from uevent
            try:
                uevent_path = device_path / "uevent"
                if uevent_path.exists():
                    uevent = uevent_path.read_text()
                    for line in uevent.splitlines():
                        if line.startswith("PCI_ID="):
                            pci_id = line.split("=")[1]
                            gpu_name = f"AMD GPU ({pci_id})"
                            break
            except Exception:
                pass

        # Try to get VRAM size
        vram_gb = None
        try:
            # Check for VRAM via amdgpu driver sysfs
            vram_path = device_path / "mem_info_vram_total"
            if vram_path.exists():
                vram_bytes = int(vram_path.read_text().strip())
                vram_gb = round(vram_bytes / (1024**3), 1)
        except Exception:
            pass

        # Check if this GPU is being used for display
        # A GPU driving a display typically has active connectors
        is_display = False
        try:
            for entry in card_dir.iterdir():
                if entry.name.startswith(card_dir.name + "-"):
                    # This is a connector (e.g., card0-DP-1)
                    status_path = entry / "status"
                    if status_path.exists():
                        status = status_path.read_text().strip()
                        if status == "connected":
                            is_display = True
                            break
        except Exception:
            pass

        gpus.append(GPUInfo(
            index=gpu_index,
            name=gpu_name,
            render_device=render_device,
            card_device=card_device,
            vram_gb=vram_gb,
            is_display_gpu=is_display,
        ))
        gpu_index += 1

    return gpus


# Cache GPU list for repeated lookups
_cached_gpu_list: Optional[List[GPUInfo]] = None


def get_render_devices_for_gpus(gpu_indices: List[int]) -> List[str]:
    """Get the /dev/dri/renderD* paths for specific GPU indices.

    Args:
        gpu_indices: List of GPU indices (0, 1, 2, etc. as shown in interactive selection)

    Returns:
        List of render device paths (e.g., ["/dev/dri/renderD128", "/dev/dri/renderD129"])
    """
    global _cached_gpu_list
    if _cached_gpu_list is None:
        _cached_gpu_list = list_amd_gpus()

    # Build index-to-render-device map
    gpu_map = {gpu.index: gpu.render_device for gpu in _cached_gpu_list}

    devices = []
    for idx in gpu_indices:
        if idx in gpu_map:
            devices.append(gpu_map[idx])
    return devices


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


def translate_output_path_for_docker(
    output_path: str,
    user_home: str,
) -> Tuple[str, Optional[str], str]:
    """Translate a host output path to a container path, with mount info.

    Handles:
    - ~/path -> /output/path (mounts ~/path's parent)
    - /absolute/path -> /output/filename (mounts parent directory)
    - relative/path -> /workspace/relative/path (uses existing mount)

    Args:
        output_path: Path specified by user (may contain ~)
        user_home: User's home directory on host

    Returns:
        Tuple of (container_path, volume_mount_string_or_None, host_path)
    """
    # Expand ~ to actual home path
    expanded_path = os.path.expanduser(output_path)

    # Get parent directory and filename
    parent_dir = os.path.dirname(os.path.abspath(expanded_path))
    filename = os.path.basename(expanded_path)

    # Check if path is under home directory
    if expanded_path.startswith(user_home):
        # Path is under home - mount output directory as read-write
        # Container path: /output/<filename>
        container_path = f"/output/{filename}"
        volume_mount = f"{parent_dir}:/output"
        return container_path, volume_mount, expanded_path

    # Check if path is absolute (outside home and cwd)
    if os.path.isabs(expanded_path):
        # Absolute path - mount parent directory
        container_path = f"/output/{filename}"
        volume_mount = f"{parent_dir}:/output"
        return container_path, volume_mount, expanded_path

    # Relative path - will be relative to /workspace in container
    container_path = f"/workspace/{output_path}"
    return container_path, None, os.path.join(os.getcwd(), output_path)


def process_docker_args(
    args: List[str],
    user_home: str,
) -> Tuple[List[str], Optional[str], Optional[str]]:
    """Process hftool args to handle output paths for Docker.

    Finds -o/--output-file arguments, translates paths for container,
    and returns mount information.

    Args:
        args: Original hftool arguments
        user_home: User's home directory on host

    Returns:
        Tuple of (modified_args, volume_mount_or_None, host_output_path_or_None)
    """
    new_args = list(args)
    volume_mount = None
    host_output_path = None

    for i, arg in enumerate(new_args):
        if arg in ("-o", "--output-file") and i + 1 < len(new_args):
            output_path = new_args[i + 1]
            container_path, volume_mount, host_output_path = translate_output_path_for_docker(
                output_path, user_home
            )
            # Update the arg with container path
            new_args[i + 1] = container_path
            break

    return new_args, volume_mount, host_output_path


def get_docker_run_command(
    hardware: HardwareInfo,
    hftool_args: List[str],
    workdir: Optional[str] = None,
    hf_token: Optional[str] = None,
    extra_volumes: Optional[List[str]] = None,
    gpu_indices: Optional[List[int]] = None,
    mount_home: bool = True,
    output_volume: Optional[str] = None,
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
        output_volume: Volume mount for output directory (from process_docker_args)

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
        # /dev/kfd is always required for ROCm compute
        cmd.extend([
            "--device=/dev/kfd",
            "--security-opt", "seccomp=unconfined",
        ])

        # GPU device passthrough: pass only specific render devices if selected
        if gpu_indices:
            # Pass only the selected GPU render devices
            render_devices = get_render_devices_for_gpus(gpu_indices)
            for device in render_devices:
                cmd.extend(["--device", device])
            # Container sees GPUs as 0, 1, 2... regardless of host indices
            # So we set HIP_VISIBLE_DEVICES to container indices (0,1,2...)
            container_indices = ",".join(str(i) for i in range(len(gpu_indices)))
            cmd.extend(["-e", f"HIP_VISIBLE_DEVICES={container_indices}"])
            cmd.extend(["-e", f"ROCR_VISIBLE_DEVICES={container_indices}"])
        else:
            # No specific GPU selected - pass all of /dev/dri
            cmd.extend(["--device=/dev/dri"])

        # Use numeric GIDs instead of group names for --user compatibility
        # Group names inside container may not match host GIDs
        added_gids = set()
        try:
            kfd_gid = os.stat("/dev/kfd").st_gid
            cmd.extend(["--group-add", str(kfd_gid)])
            added_gids.add(kfd_gid)
        except OSError:
            cmd.extend(["--group-add", "render"])
        try:
            # Get video group from render device(s)
            devices_to_check = get_render_devices_for_gpus(gpu_indices) if gpu_indices else []
            if not devices_to_check:
                # Fallback to any render device
                for entry in os.listdir("/dev/dri"):
                    if entry.startswith("renderD"):
                        devices_to_check.append(f"/dev/dri/{entry}")
                        break
            for device in devices_to_check:
                try:
                    dri_gid = os.stat(device).st_gid
                    if dri_gid not in added_gids:
                        cmd.extend(["--group-add", str(dri_gid)])
                        added_gids.add(dri_gid)
                except OSError:
                    pass
        except OSError:
            cmd.extend(["--group-add", "video"])

        # Pass through HSA_OVERRIDE_GFX_VERSION if set
        gfx_version = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
        if gfx_version:
            cmd.extend(["-e", f"HSA_OVERRIDE_GFX_VERSION={gfx_version}"])

        # Suppress MIOpen workspace warnings (harmless but noisy)
        # Level 4 = errors only (0=all, 1=info, 2=warnings, 4=errors, 5=fatal)
        cmd.extend(["-e", "MIOPEN_LOG_LEVEL=4"])

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

    # Mount output directory if specified (for paths outside workspace)
    if output_volume:
        cmd.extend(["-v", output_volume])

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

    # Pass multi-GPU flag if multiple GPUs selected
    if gpu_indices and len(gpu_indices) > 1:
        cmd.extend(["-e", "HFTOOL_MULTI_GPU=1"])

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
    no_cache: bool = False,
) -> bool:
    """Build the hftool Docker image for a platform.

    Args:
        platform: Target platform (rocm, cuda, cpu)
        project_root: Path to hftool project (for Dockerfile)
        quiet: Suppress output
        no_cache: Disable Docker cache (rebuilds from scratch)

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
    ]
    if no_cache:
        cmd.append("--no-cache")
    cmd.append(".")

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
) -> Tuple[int, Optional[str]]:
    """Run hftool command in Docker container.

    Args:
        hftool_args: Arguments to pass to hftool
        hardware: Pre-detected hardware (will detect if None)
        auto_setup: Automatically build/pull image if missing
        gpu_indices: Specific GPU indices to use (None = all GPUs)

    Returns:
        Tuple of (exit_code, host_output_path_or_None)
    """
    user_home = os.path.expanduser("~")
    if hardware is None:
        hardware = detect_hardware()

    if not hardware.docker_available:
        print("Error: Docker is not installed or not running.")
        print("Install Docker: https://docs.docker.com/get-docker/")
        return 1, None

    if hardware.platform == GPUPlatform.MPS:
        print("Note: Docker GPU passthrough is not supported on Apple Silicon.")
        print("Running natively is recommended for MPS devices.")
        return 1, None

    # Build or pull image if needed
    if not hardware.image_available and auto_setup:
        print(f"Building {hardware.recommended_image}...")
        # Try to find project root for building
        project_root = Path(__file__).parent.parent.parent
        if (project_root / "docker").exists():
            if not build_image(hardware.platform, str(project_root)):
                print(f"Failed to build {hardware.recommended_image}")
                return 1, None
        else:
            # Can't build locally, would need to pull from registry
            print(f"Image {hardware.recommended_image} not found.")
            print("Please build the image first:")
            print(f"  cd <hftool-repo> && docker build -f docker/Dockerfile.{hardware.platform.value} -t {hardware.recommended_image} .")
            return 1, None

    # Process output paths - translate ~/path to container paths and get mount info
    processed_args, output_volume, host_output_path = process_docker_args(
        hftool_args, user_home
    )

    cmd = get_docker_run_command(
        hardware,
        processed_args,
        gpu_indices=gpu_indices,
        output_volume=output_volume,
    )

    try:
        result = subprocess.run(cmd)
        return result.returncode, host_output_path
    except KeyboardInterrupt:
        return 130, host_output_path  # Standard exit code for SIGINT
    except Exception as e:
        print(f"Error running Docker: {e}")
        return 1, None


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


def interactive_gpu_select(platform: GPUPlatform = GPUPlatform.ROCM) -> Optional[List[int]]:
    """Interactive GPU selection for Docker mode.

    Detects available GPUs and presents a menu for selection.

    Args:
        platform: GPU platform (currently only ROCM supported)

    Returns:
        List of selected GPU indices, or None if cancelled
    """
    if platform == GPUPlatform.ROCM:
        gpus = list_amd_gpus()
    else:
        # For other platforms, return None (use all GPUs)
        return None

    if not gpus:
        print("No AMD GPUs detected.")
        return None

    if len(gpus) == 1:
        # Only one GPU, select it automatically
        gpu = gpus[0]
        display_note = " (display)" if gpu.is_display_gpu else ""
        vram_str = f", {gpu.vram_gb}GB" if gpu.vram_gb else ""
        print(f"  Using GPU 0: {gpu.name}{vram_str}{display_note}")
        return [0]

    # Multiple GPUs - show selection menu
    print("\n  Available GPUs:")
    for gpu in gpus:
        display_note = " (display)" if gpu.is_display_gpu else ""
        vram_str = f", {gpu.vram_gb}GB" if gpu.vram_gb else ""
        print(f"    [{gpu.index}] {gpu.name}{vram_str}{display_note}")

    # Find non-display GPU as default recommendation
    non_display_gpus = [g for g in gpus if not g.is_display_gpu]
    if non_display_gpus:
        default = str(non_display_gpus[0].index)
        default_hint = f" [default: {default}]"
    else:
        default = "0"
        default_hint = " [default: 0]"

    print(f"\n  Enter GPU number(s), comma-separated for multi-GPU{default_hint}")
    print("  Press Enter for default, 'all' for all GPUs, 'q' to cancel")

    try:
        choice = input("  GPU> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.")
        return None

    if choice == "q":
        return None

    if choice == "" or choice == "auto":
        return [int(default)]

    if choice == "all":
        return [g.index for g in gpus]

    # Parse comma-separated indices
    try:
        indices = [int(x.strip()) for x in choice.split(",")]
        valid_indices = [g.index for g in gpus]
        for idx in indices:
            if idx not in valid_indices:
                print(f"  Invalid GPU index: {idx}")
                return None
        return indices
    except ValueError:
        print(f"  Invalid input: {choice}")
        return None


def parse_gpu_arg(gpu_arg: Optional[str], platform: GPUPlatform = GPUPlatform.ROCM) -> Optional[List[int]]:
    """Parse --gpu argument value into GPU indices.

    Args:
        gpu_arg: Value from --gpu argument (e.g., "1", "0,1", "all", "auto", None)
        platform: GPU platform

    Returns:
        List of GPU indices, or None to use all GPUs
    """
    if gpu_arg is None:
        return None

    gpu_arg = gpu_arg.strip().lower()

    if gpu_arg == "all":
        return None  # None means all GPUs

    if gpu_arg == "auto":
        # Auto-select best non-display GPU
        if platform == GPUPlatform.ROCM:
            gpus = list_amd_gpus()
            non_display = [g for g in gpus if not g.is_display_gpu]
            if non_display:
                return [non_display[0].index]
            elif gpus:
                return [gpus[0].index]
        return None

    # Parse comma-separated indices
    try:
        indices = [int(x.strip()) for x in gpu_arg.split(",")]
        return indices
    except ValueError:
        return None
