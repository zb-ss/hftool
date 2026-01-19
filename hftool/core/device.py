"""Device detection and configuration for hftool.

Supports ROCm (AMD), CUDA (NVIDIA), MPS (Apple Silicon), and CPU.
ROCm is the primary target for this project.

Multi-GPU support includes:
- Automatic detection of display GPU to avoid VRAM conflicts
- Explicit GPU selection via --gpu flag or HFTOOL_GPU env var
- Multi-GPU model parallelism for large models
"""

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Try to import torch, but allow the module to be imported without it
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


def configure_rocm_env() -> None:
    """Configure environment variables for optimal ROCm performance.
    
    This should be called early, before PyTorch operations.
    Sets up experimental features and memory optimizations for AMD GPUs.
    """
    # Enable experimental memory-efficient attention for RDNA3 (Navi31, etc.)
    # This enables AOTriton optimizations for scaled_dot_product_attention
    if "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL" not in os.environ:
        os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
    
    # Reduce memory fragmentation with expandable segments
    # Note: PYTORCH_HIP_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF
    if "PYTORCH_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    
    # Use hipBLAS instead of hipBLASLt for better compatibility on consumer GPUs
    # hipBLASLt is optimized for datacenter GPUs (MI250, MI300) but may not work well on RDNA3
    if "TORCH_BLAS_PREFER_HIPBLASLT" not in os.environ:
        os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"


@dataclass
class DeviceInfo:
    """Information about the detected compute device."""
    device: str  # "cuda", "mps", or "cpu"
    device_name: str  # Human-readable name
    is_rocm: bool  # True if AMD ROCm
    is_cuda: bool  # True if NVIDIA CUDA
    is_mps: bool  # True if Apple MPS
    device_count: int  # Number of devices
    total_memory_gb: Optional[float]  # Total VRAM in GB (if available)
    supports_bfloat16: bool  # Whether device supports bfloat16


@dataclass
class GPUInfo:
    """Detailed information about a specific GPU."""
    index: int  # PyTorch device index
    name: str  # GPU name (e.g., "AMD Radeon RX 7900 XTX")
    vram_gb: float  # Total VRAM in GB
    pci_bus: Optional[str]  # PCI bus address (e.g., "0000:03:00.0")
    has_display: bool  # True if display is connected to this GPU
    render_device: Optional[str]  # DRI render node (e.g., "/dev/dri/renderD128")
    is_rocm: bool  # True if AMD ROCm GPU


def detect_device() -> str:
    """Auto-detect the best available compute device.
    
    Returns:
        Device string: "cuda" (for both NVIDIA and ROCm), "mps", or "cpu"
    """
    if not _TORCH_AVAILABLE:
        return "cpu"
    
    # ROCm presents itself as CUDA to PyTorch
    if torch.cuda.is_available():
        return "cuda"
    
    # Apple Silicon MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    
    return "cpu"


def is_rocm() -> bool:
    """Check if the current CUDA device is actually AMD ROCm.
    
    Returns:
        True if running on ROCm, False otherwise
    """
    if not _TORCH_AVAILABLE or not torch.cuda.is_available():
        return False
    
    try:
        device_name = torch.cuda.get_device_name(0)
        # AMD GPUs typically have "AMD" or "Radeon" in the name
        rocm_detected = "AMD" in device_name or "Radeon" in device_name or "gfx" in device_name.lower()
        if rocm_detected:
            # Configure ROCm-specific optimizations
            configure_rocm_env()
        return rocm_detected
    except Exception:
        return False


def get_device_info() -> DeviceInfo:
    """Get detailed information about the compute device.
    
    Returns:
        DeviceInfo dataclass with device details
    """
    if not _TORCH_AVAILABLE:
        return DeviceInfo(
            device="cpu",
            device_name="CPU (torch not available)",
            is_rocm=False,
            is_cuda=False,
            is_mps=False,
            device_count=0,
            total_memory_gb=None,
            supports_bfloat16=False,
        )
    
    device = detect_device()
    
    if device == "cuda":
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        rocm = is_rocm()
        
        # Get total memory
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            total_memory_gb = total_memory / (1024 ** 3)
        except Exception:
            total_memory_gb = None
        
        # ROCm 6.x and modern NVIDIA cards support bfloat16
        # ROCm: RDNA3 (gfx1100+) and CDNA2+ support bfloat16
        # NVIDIA: Ampere+ (compute capability 8.0+) supports bfloat16
        supports_bf16 = True  # Modern GPUs generally support it
        if not rocm:
            try:
                props = torch.cuda.get_device_properties(0)
                supports_bf16 = props.major >= 8  # Ampere+
            except Exception:
                supports_bf16 = False
        
        return DeviceInfo(
            device="cuda",
            device_name=device_name,
            is_rocm=rocm,
            is_cuda=not rocm,
            is_mps=False,
            device_count=device_count,
            total_memory_gb=total_memory_gb,
            supports_bfloat16=supports_bf16,
        )
    
    elif device == "mps":
        return DeviceInfo(
            device="mps",
            device_name="Apple Silicon (MPS)",
            is_rocm=False,
            is_cuda=False,
            is_mps=True,
            device_count=1,
            total_memory_gb=None,  # MPS doesn't expose this easily
            supports_bfloat16=False,  # MPS has limited bfloat16 support
        )
    
    else:
        return DeviceInfo(
            device="cpu",
            device_name="CPU",
            is_rocm=False,
            is_cuda=False,
            is_mps=False,
            device_count=0,
            total_memory_gb=None,
            supports_bfloat16=False,
        )


def get_optimal_dtype(device: Optional[str] = None):
    """Get the optimal dtype for the given device.
    
    Args:
        device: Device string ("cuda", "mps", "cpu"). If None, auto-detect.
    
    Returns:
        torch.dtype: Optimal dtype (bfloat16 preferred for modern GPUs)
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for dtype selection")
    
    if device is None:
        device = detect_device()
    
    if device == "cuda":
        info = get_device_info()
        if info.supports_bfloat16:
            return torch.bfloat16
        return torch.float16
    
    elif device == "mps":
        # MPS works best with float16
        return torch.float16
    
    else:
        return torch.float32


def get_device_map(device: Optional[str] = None, multi_gpu: bool = True) -> str:
    """Get the device_map string for model loading.

    Args:
        device: Device string. If None, auto-detect.
        multi_gpu: Whether to use multiple GPUs if available.

    Returns:
        Device map string for from_pretrained()
    """
    if device is None:
        device = detect_device()

    if device == "cuda":
        if multi_gpu and _TORCH_AVAILABLE and torch.cuda.device_count() > 1:
            return "auto"  # Let accelerate handle multi-GPU
        return "cuda:0"

    return device


def _get_drm_card_for_gpu(gpu_index: int) -> Optional[int]:
    """Map PyTorch GPU index to DRM card number.

    This is tricky because PyTorch GPU indices may not match DRM card numbers.
    We try to match by PCI bus address.

    Args:
        gpu_index: PyTorch CUDA device index

    Returns:
        DRM card number or None if not found
    """
    if not _TORCH_AVAILABLE or not torch.cuda.is_available():
        return None

    try:
        # Get PCI bus ID from PyTorch (format: "0000:03:00.0")
        pci_bus = torch.cuda.get_device_properties(gpu_index).pci_bus_id
        if not pci_bus:
            return None

        # Search DRM cards for matching PCI address
        for card_path in glob.glob("/sys/class/drm/card[0-9]*"):
            card_num = int(Path(card_path).name.replace("card", ""))
            device_path = Path(card_path) / "device"

            if device_path.is_symlink():
                # The symlink points to the PCI device
                real_path = device_path.resolve()
                pci_addr = real_path.name  # e.g., "0000:03:00.0"
                if pci_addr == pci_bus:
                    return card_num

        return None
    except Exception:
        return None


def is_display_gpu(gpu_index: int) -> bool:
    """Check if a GPU has displays connected.

    This checks the DRM subsystem for connected display connectors.
    A GPU running the desktop compositor should be avoided for compute
    to prevent VRAM conflicts and crashes.

    Args:
        gpu_index: PyTorch CUDA device index

    Returns:
        True if display(s) connected to this GPU, False otherwise
    """
    card_num = _get_drm_card_for_gpu(gpu_index)
    if card_num is None:
        # Fallback: assume GPU 0 has the display if we can't determine
        return gpu_index == 0

    # Check all connectors for this card
    connector_pattern = f"/sys/class/drm/card{card_num}-*"
    for connector in glob.glob(connector_pattern):
        status_file = Path(connector) / "status"
        if status_file.exists():
            try:
                status = status_file.read_text().strip()
                if status == "connected":
                    return True
            except Exception:
                continue

    return False


def get_all_gpus() -> List[GPUInfo]:
    """Enumerate all available GPUs with detailed information.

    Returns:
        List of GPUInfo for each detected GPU, including display detection.
    """
    gpus = []

    if not _TORCH_AVAILABLE or not torch.cuda.is_available():
        return gpus

    device_count = torch.cuda.device_count()
    rocm = is_rocm()

    for i in range(device_count):
        try:
            props = torch.cuda.get_device_properties(i)
            name = props.name
            vram_gb = props.total_memory / (1024 ** 3)
            pci_bus = props.pci_bus_id if hasattr(props, "pci_bus_id") else None

            # Find render device
            card_num = _get_drm_card_for_gpu(i)
            render_device = None
            if card_num is not None:
                # renderD128 corresponds to card0, renderD129 to card1, etc.
                render_path = f"/dev/dri/renderD{128 + card_num}"
                if Path(render_path).exists():
                    render_device = render_path

            has_display = is_display_gpu(i)

            gpus.append(GPUInfo(
                index=i,
                name=name,
                vram_gb=vram_gb,
                pci_bus=pci_bus,
                has_display=has_display,
                render_device=render_device,
                is_rocm=rocm,
            ))
        except Exception:
            continue

    return gpus


def get_compute_gpu() -> int:
    """Get the best GPU index for compute workloads.

    This prefers GPUs without connected displays to avoid VRAM conflicts
    with the desktop compositor (sway, KDE, GNOME, etc.).

    If all GPUs have displays, or detection fails, returns the GPU
    with the most VRAM.

    Returns:
        GPU index (0-based) for compute workloads
    """
    gpus = get_all_gpus()

    if not gpus:
        return 0  # Fallback to first GPU

    # Prefer GPUs without displays
    non_display_gpus = [g for g in gpus if not g.has_display]

    if non_display_gpus:
        # Among non-display GPUs, prefer the one with most VRAM
        best = max(non_display_gpus, key=lambda g: g.vram_gb)
        return best.index

    # All GPUs have displays - pick the one with most VRAM
    best = max(gpus, key=lambda g: g.vram_gb)
    return best.index


def parse_gpu_selection(gpu_arg: str) -> List[int]:
    """Parse the --gpu argument into a list of GPU indices.

    Args:
        gpu_arg: One of:
            - "auto": Use get_compute_gpu() to select best GPU
            - "all": Use all available GPUs
            - "0", "1", etc.: Use specific GPU
            - "0,1", "1,2", etc.: Use multiple specific GPUs

    Returns:
        List of GPU indices to use
    """
    if not _TORCH_AVAILABLE or not torch.cuda.is_available():
        return []

    device_count = torch.cuda.device_count()

    if gpu_arg == "auto":
        return [get_compute_gpu()]

    if gpu_arg == "all":
        return list(range(device_count))

    # Parse comma-separated indices
    try:
        indices = [int(x.strip()) for x in gpu_arg.split(",")]
        # Validate indices
        valid = [i for i in indices if 0 <= i < device_count]
        return valid if valid else [0]
    except ValueError:
        return [get_compute_gpu()]


def get_cuda_visible_devices(gpu_indices: List[int]) -> str:
    """Get the CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES value.

    Args:
        gpu_indices: List of GPU indices to make visible

    Returns:
        Comma-separated string of GPU indices
    """
    return ",".join(str(i) for i in gpu_indices)
