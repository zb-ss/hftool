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
from typing import List, Optional, Sequence

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
    # Native PyTorch SDPA is the default. Experimental attention backends are
    # opt-in and must not be silently enabled here.

    # Reduce memory fragmentation with expandable segments
    # Note: PYTORCH_HIP_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF
    if "PYTORCH_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    # Use hipBLAS instead of hipBLASLt for better compatibility on consumer GPUs
    # hipBLASLt is optimized for datacenter GPUs (MI250, MI300) but may not work well on RDNA3
    if "TORCH_BLAS_PREFER_HIPBLASLT" not in os.environ:
        os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"

    # Patch torch.repeat_interleave for ROCm HIP compatibility.
    # This op crashes with hipErrorIllegalState when both input and
    # repeats are CUDA tensors on RDNA3 GPUs (PyTorch 2.9 + ROCm 7.1).
    # Must be applied BEFORE any model loads — not just VLM, since TTS
    # and other models also trigger the same crash.
    _patch_repeat_interleave_for_rocm()


_REPEAT_INTERLEAVE_PATCHED = False


def _patch_repeat_interleave_for_rocm() -> None:
    """Globally patch ``torch.repeat_interleave`` for ROCm HIP.

    ``torch.repeat_interleave`` with CUDA tensors crashes on RDNA3 GPUs
    under ROCm HIP (``hipErrorIllegalState``).  The workaround moves
    tensors to CPU for that single op, then back to the original device.
    This is a no-op if PyTorch isn't available or isn't built for ROCm.
    """
    global _REPEAT_INTERLEAVE_PATCHED
    if _REPEAT_INTERLEAVE_PATCHED or not _TORCH_AVAILABLE:
        return
    if getattr(torch.version, "hip", None) is None:
        return

    _original = torch.repeat_interleave

    def _safe_repeat_interleave(input, repeats, *args, **kwargs):
        if isinstance(repeats, torch.Tensor) and repeats.is_cuda:
            device = input.device
            return _original(input.cpu(), repeats.cpu(), *args, **kwargs).to(device)
        return _original(input, repeats, *args, **kwargs)

    torch.repeat_interleave = _safe_repeat_interleave
    _REPEAT_INTERLEAVE_PATCHED = True


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
    """Detailed information about a visible GPU and its physical identity."""

    index: int  # PyTorch-visible index
    name: str
    vram_gb: float
    pci_bus: Optional[str]
    has_display: bool
    render_device: Optional[str]
    is_rocm: bool
    free_vram_gb: Optional[float] = None
    physical_index: Optional[int] = None

    @property
    def resolved_physical_index(self) -> int:
        """Return the physical index, falling back to the visible index."""
        return self.index if self.physical_index is None else self.physical_index


@dataclass(frozen=True)
class GPUSelection:
    """Result of a load-aware single-GPU selection decision."""

    gpu: Optional[GPUInfo]
    adequate: bool
    required_vram_gb: Optional[float]
    safety_reserve_gb: float
    reason: str

    @property
    def visible_index(self) -> Optional[int]:
        return self.gpu.index if self.gpu else None

    def format_message(self) -> str:
        """Explain physical/visible mapping, live memory, display, and reason."""
        if self.gpu is None:
            return "No compatible GPU is visible; use CPU or check Docker GPU passthrough."
        free_text = (
            f"{self.gpu.free_vram_gb:.1f}/{self.gpu.vram_gb:.1f} GB free"
            if self.gpu.free_vram_gb is not None
            else f"free VRAM unavailable ({self.gpu.vram_gb:.1f} GB total)"
        )
        display_text = "display attached" if self.gpu.has_display else "no display"
        return (
            f"Selected physical GPU {self.gpu.resolved_physical_index} as PyTorch cuda:{self.gpu.index}: "
            f"{free_text}, {display_text}. {self.reason}"
        )


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


def get_visible_physical_indices(device_count: Optional[int] = None) -> List[int]:
    """Map PyTorch-visible indices back to user-facing physical GPU indices."""
    if device_count is None:
        device_count = torch.cuda.device_count() if _TORCH_AVAILABLE and torch.cuda.is_available() else 0

    for variable in (
        "HFTOOL_PHYSICAL_GPU_INDICES",
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
    ):
        raw_value = os.environ.get(variable, "").strip()
        if not raw_value:
            continue
        try:
            indices = [int(value.strip()) for value in raw_value.split(",")]
        except ValueError:
            continue
        if len(indices) >= device_count:
            return indices[:device_count]

    return list(range(device_count))


def get_all_gpus() -> List[GPUInfo]:
    """Enumerate visible GPUs with live free memory and physical identity."""
    gpus: List[GPUInfo] = []
    if not _TORCH_AVAILABLE or not torch.cuda.is_available():
        return gpus

    device_count = torch.cuda.device_count()
    physical_indices = get_visible_physical_indices(device_count)
    rocm = is_rocm()

    for visible_index in range(device_count):
        try:
            props = torch.cuda.get_device_properties(visible_index)
            total_vram_gb = props.total_memory / (1024 ** 3)
            try:
                free_bytes, _ = torch.cuda.mem_get_info(visible_index)
                free_vram_gb: Optional[float] = free_bytes / (1024 ** 3)
            except Exception:
                free_vram_gb = None

            card_num = _get_drm_card_for_gpu(visible_index)
            render_device = None
            if card_num is not None:
                render_path = f"/dev/dri/renderD{128 + card_num}"
                if Path(render_path).exists():
                    render_device = render_path

            gpus.append(
                GPUInfo(
                    index=visible_index,
                    physical_index=physical_indices[visible_index],
                    name=props.name,
                    vram_gb=total_vram_gb,
                    free_vram_gb=free_vram_gb,
                    pci_bus=props.pci_bus_id if hasattr(props, "pci_bus_id") else None,
                    has_display=is_display_gpu(visible_index),
                    render_device=render_device,
                    is_rocm=rocm,
                )
            )
        except Exception:
            continue

    return gpus


def _selection_policy() -> tuple[float, float]:
    """Resolve configurable GPU reserve and display-penalty policy."""
    from hftool.core.config import Config
    from hftool.core.models import get_catalog_runtime_config

    defaults = get_catalog_runtime_config("gpu_selection")
    config = Config.get()
    reserve = float(
        config.get_value(
            "gpu_safety_reserve_gb",
            default=defaults.get("safety_reserve_gb", 0.0),
        )
    )
    display_penalty = float(
        config.get_value(
            "gpu_display_penalty_gb",
            default=defaults.get("display_penalty_gb", 0.0),
        )
    )
    if reserve < 0 or display_penalty < 0:
        raise ValueError("GPU reserve and display penalty must be non-negative")
    return reserve, display_penalty


def select_compute_gpu(
    required_vram_gb: Optional[float] = None,
    *,
    gpus: Optional[Sequence[GPUInfo]] = None,
    safety_reserve_gb: Optional[float] = None,
    display_penalty_gb: Optional[float] = None,
) -> GPUSelection:
    """Choose one GPU using live memory, display pressure, and model needs."""
    candidates = list(get_all_gpus() if gpus is None else gpus)
    policy_reserve, policy_display_penalty = _selection_policy()
    reserve = policy_reserve if safety_reserve_gb is None else safety_reserve_gb
    display_penalty = (
        policy_display_penalty if display_penalty_gb is None else display_penalty_gb
    )
    if reserve < 0 or display_penalty < 0:
        raise ValueError("GPU reserve and display penalty must be non-negative")
    if not candidates:
        return GPUSelection(None, False, required_vram_gb, reserve, "No GPU detected.")

    def free_vram(gpu: GPUInfo) -> float:
        return gpu.free_vram_gb if gpu.free_vram_gb is not None else gpu.vram_gb

    def is_adequate(gpu: GPUInfo) -> bool:
        if required_vram_gb is None:
            return True
        return free_vram(gpu) >= required_vram_gb + reserve

    def score(gpu: GPUInfo) -> float:
        return free_vram(gpu) - (display_penalty if gpu.has_display else 0.0)

    adequate_candidates = [gpu for gpu in candidates if is_adequate(gpu)]
    pool = adequate_candidates or candidates
    best = max(pool, key=lambda gpu: (score(gpu), free_vram(gpu), gpu.vram_gb))
    adequate = is_adequate(best)

    if required_vram_gb is None:
        reason = "Highest live headroom after applying the display-GPU penalty."
    elif adequate:
        reason = (
            f"Meets the {required_vram_gb:.1f} GB model minimum plus the "
            f"{reserve:.1f} GB safety reserve."
        )
    else:
        reason = (
            f"No GPU meets the {required_vram_gb:.1f} GB model minimum plus the "
            f"{reserve:.1f} GB safety reserve; CPU offload or freeing VRAM is required."
        )
    return GPUSelection(best, adequate, required_vram_gb, reserve, reason)


def get_compute_gpu(required_vram_gb: Optional[float] = None) -> int:
    """Return the best PyTorch-visible GPU index for compatibility."""
    selection = select_compute_gpu(required_vram_gb)
    return selection.visible_index if selection.visible_index is not None else 0


def parse_gpu_selection(gpu_arg: str, required_vram_gb: Optional[float] = None) -> List[int]:
    """Parse physical GPU intent into PyTorch-visible indices."""
    if not _TORCH_AVAILABLE or not torch.cuda.is_available():
        return []

    device_count = torch.cuda.device_count()
    if gpu_arg == "auto":
        return [get_compute_gpu(required_vram_gb)]
    if gpu_arg == "all":
        return list(range(device_count))

    try:
        requested_physical = [int(value.strip()) for value in gpu_arg.split(",")]
    except ValueError as error:
        raise ValueError(
            f"Invalid GPU selection '{gpu_arg}'. Use auto, all, or comma-separated indices."
        ) from error

    physical_to_visible = {
        physical: visible
        for visible, physical in enumerate(get_visible_physical_indices(device_count))
    }
    invalid = [index for index in requested_physical if index not in physical_to_visible]
    if invalid:
        visible_physical = ", ".join(str(index) for index in sorted(physical_to_visible))
        raise ValueError(
            f"Physical GPU index {invalid[0]} is not visible. Available physical GPUs: "
            f"{visible_physical or 'none'}."
        )
    return [physical_to_visible[index] for index in requested_physical]


def get_cuda_visible_devices(gpu_indices: List[int]) -> str:
    """Get the CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES value.

    Args:
        gpu_indices: List of GPU indices to make visible

    Returns:
        Comma-separated string of GPU indices
    """
    return ",".join(str(i) for i in gpu_indices)


def get_multi_gpu_kwargs(
    reserve_per_gpu_gb: float = 6.0,
    cpu_fallback_gb: float = 64.0,
    allow_cpu_offload: bool = False,
) -> dict:
    """Get kwargs for distributing a model across multiple GPUs.

    This is the centralized function for multi-GPU support. All task handlers
    should use this to get consistent multi-GPU behavior.

    The function checks HFTOOL_MULTI_GPU environment variable:
    - "1", "true", "yes", "balanced": Enable multi-GPU distribution
    - "0", "false", "no": Disable multi-GPU (single GPU or CPU offload)
    - unset: Keep single-GPU mode; multi-GPU requires explicit opt-in

    IMPORTANT: By default, CPU is NOT included in the device map to prevent
    critical components (like text_encoder) from being placed on CPU, which
    causes device mismatch errors during inference. Set allow_cpu_offload=True
    only if the model explicitly supports CPU offloading.

    Args:
        reserve_per_gpu_gb: Memory to reserve per GPU for VAE/intermediate tensors
        cpu_fallback_gb: CPU memory available for fallback offloading
        allow_cpu_offload: If True, include CPU in device map for memory-constrained
            situations. WARNING: This can cause device mismatch errors with some models.

    Returns:
        Dict with keys:
        - "use_multi_gpu": bool - Whether multi-GPU is enabled
        - "num_gpus": int - Number of available GPUs
        - "device_map": str or None - Device map for from_pretrained()
        - "max_memory": dict or None - Memory limits per device
        - "no_split_module_classes": list - Modules that should not be split across devices
        - "message": str - Status message to display to user

    Example:
        >>> from hftool.core.device import get_multi_gpu_kwargs
        >>> kwargs = get_multi_gpu_kwargs()
        >>> if kwargs["use_multi_gpu"]:
        ...     pipe = Pipeline.from_pretrained(
        ...         model,
        ...         device_map=kwargs["device_map"],
        ...         max_memory=kwargs["max_memory"],
        ...     )
    """
    result = {
        "use_multi_gpu": False,
        "num_gpus": 0,
        "device_map": None,
        "max_memory": None,
        "no_split_module_classes": None,
        "message": "",
    }

    if not _TORCH_AVAILABLE or not torch.cuda.is_available():
        result["message"] = "No GPU available"
        return result

    num_gpus = torch.cuda.device_count()
    result["num_gpus"] = num_gpus

    if num_gpus <= 1:
        result["message"] = "Single GPU mode"
        return result

    # Check environment variable
    # IMPORTANT: Multi-GPU is OFF by default. Only enable when explicitly requested
    # via HFTOOL_MULTI_GPU=1 or --gpu all (which sets the env var)
    multi_gpu_env = os.environ.get("HFTOOL_MULTI_GPU", "").lower()

    # Explicit enable (set by CLI when user selects --gpu all)
    if multi_gpu_env in ("1", "true", "yes", "balanced"):
        use_multi_gpu = True
    # Explicit disable OR not set (default is OFF)
    else:
        # Multi-GPU requires explicit opt-in to avoid device mismatch issues
        result["message"] = f"Single GPU mode ({num_gpus} GPUs available). Use --gpu all for multi-GPU."
        return result

    if not use_multi_gpu:
        return result

    # Calculate memory allocation - GPU only by default
    # Set max_memory to 0 for CPU to completely prevent CPU placement
    max_memory = {}
    gpu_allocs = []

    for i in range(num_gpus):
        try:
            mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            usable_gb = int(mem_gb - reserve_per_gpu_gb)
            if usable_gb > 0:
                max_memory[i] = f"{usable_gb}GB"
                gpu_allocs.append(f"GPU {i}: {usable_gb}GB")
        except Exception:
            pass

    # Only add CPU fallback if explicitly allowed
    # WARNING: Including CPU can cause device mismatch errors when text_encoder
    # or other critical components get placed on CPU
    if allow_cpu_offload:
        max_memory["cpu"] = f"{int(cpu_fallback_gb)}GB"
        cpu_msg = f", CPU fallback: {int(cpu_fallback_gb)}GB"
    else:
        # Explicitly set CPU to 0 to prevent any CPU placement
        max_memory["cpu"] = "0GB"
        cpu_msg = ""

    result["use_multi_gpu"] = True
    result["device_map"] = "balanced"
    result["max_memory"] = max_memory
    result["message"] = f"Multi-GPU mode: Distributing across {num_gpus} GPUs ({', '.join(gpu_allocs)}{cpu_msg})"

    return result
