"""Device detection and configuration for hftool.

Supports ROCm (AMD), CUDA (NVIDIA), MPS (Apple Silicon), and CPU.
ROCm is the primary target for this project.
"""

import os
from dataclasses import dataclass
from typing import Optional

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
