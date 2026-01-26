# GPU Setup Guide

hftool is designed to work with AMD ROCm, NVIDIA CUDA, Apple MPS, and CPU. This guide covers setup for each platform.

## Quick Start

```bash
# Check your GPU status
hftool doctor

# See detected GPU info
hftool --version
```

---

## AMD ROCm (Primary Target)

hftool is optimized for AMD GPUs running ROCm. We recommend using Docker for the best experience.

### Option 1: Docker (Recommended)

Docker provides an isolated ROCm environment without affecting system drivers.

```bash
# Install hftool on host
pip install hftool

# Setup Docker for your GPU
hftool docker setup

# Build the ROCm image
hftool docker build

# Run commands in Docker
hftool docker run -- -t t2i -i "A cat" -o cat.png
```

**Benefits:**
- No system driver conflicts
- Safe for gaming systems (doesn't affect desktop)
- Consistent ROCm version
- Easy to update

### Option 2: Native ROCm

If you have ROCm installed system-wide:

```bash
# Install with ROCm PyTorch
pip install hftool[all]

# Replace CUDA PyTorch with ROCm version
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

### Supported AMD GPUs

| GPU Series | Architecture | Status |
|------------|--------------|--------|
| RX 7900 XTX/XT | RDNA3 (gfx1100) | Fully supported |
| RX 7800 XT | RDNA3 (gfx1101) | Fully supported |
| RX 7600 | RDNA3 (gfx1102) | Fully supported |
| RX 6900 XT | RDNA2 (gfx1030) | Supported* |
| RX 6800 XT | RDNA2 (gfx1030) | Supported* |
| RX 6700 XT | RDNA2 (gfx1031) | Supported* |

\* May need `HSA_OVERRIDE_GFX_VERSION=10.3.0`

### ROCm Environment Variables

```bash
# For unsupported GPUs (try without first)
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # RDNA2
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # RDNA3

# Suppress MIOpen warnings
export MIOPEN_LOG_LEVEL=4

# Memory optimization
export PYTORCH_ALLOC_CONF="expandable_segments:True"
```

### Using Ollama's ROCm

If you have Ollama installed, you can use its bundled ROCm:

```bash
export HFTOOL_ROCM_PATH=/usr/share/ollama/lib/rocm
```

---

## NVIDIA CUDA

### Installation

```bash
# Install with all dependencies
pip install hftool[all]

# PyTorch CUDA version installs automatically
# If you need a specific CUDA version:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Docker Option

```bash
hftool docker setup  # Detects NVIDIA GPU
hftool docker build
hftool docker run -- -t t2i -i "A cat" -o cat.png
```

### Supported NVIDIA GPUs

| GPU Series | Compute Capability | Status |
|------------|-------------------|--------|
| RTX 40 series | 8.9 | Fully supported |
| RTX 30 series | 8.6 | Fully supported |
| RTX 20 series | 7.5 | Supported |
| GTX 16 series | 7.5 | Supported |
| GTX 10 series | 6.1 | Limited (fp32 only) |

### CUDA Environment Variables

```bash
# GPU selection
export CUDA_VISIBLE_DEVICES=0      # First GPU
export CUDA_VISIBLE_DEVICES=0,1    # Multiple GPUs

# Memory allocation
export PYTORCH_ALLOC_CONF="expandable_segments:True"
```

---

## Apple Silicon (MPS)

### Installation

```bash
pip install hftool[all]
```

PyTorch automatically uses MPS on Apple Silicon Macs.

### Limitations

- MPS has limited bfloat16 support
- Some models may fall back to float16 or float32
- Memory is unified with system RAM

### Troubleshooting

If you encounter MPS errors:

```bash
# Fall back to CPU
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

---

## CPU Only

### Installation

```bash
pip install hftool[all]

# Install CPU-only PyTorch
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Performance Tips

- Use smaller models (SDXL vs HunyuanVideo)
- Expect significantly longer inference times
- Reduce inference steps (`--steps 10`)

---

## Multi-GPU Setup

### Automatic GPU Selection

hftool automatically selects the best GPU for compute:

```bash
# Auto-select (avoids display GPU)
hftool -t t2i -i "A cat" -o cat.png

# Use all GPUs for large models
hftool -t t2v -m hunyuan -i "A cat" -o cat.mp4 --gpu all
```

### Manual GPU Selection

```bash
# Use specific GPU
hftool -t t2i -i "A cat" -o cat.png --gpu 1

# Use multiple specific GPUs
hftool -t t2v -i "A cat" -o cat.mp4 --gpu 0,1

# Environment variable
export HFTOOL_GPU=1
```

### Display GPU Avoidance

hftool detects which GPU has displays connected and avoids it by default:

```
GPU 0: RX 7900 XTX (24GB) - Display connected ← Desktop/Gaming
GPU 1: RX 7900 XT (20GB) - No display ← Compute (selected)
```

This prevents VRAM conflicts with your desktop compositor.

---

## Memory Management

### Low VRAM Tips

For GPUs with 8-12GB VRAM:

```bash
# Enable CPU offload (slower but uses less VRAM)
export HFTOOL_CPU_OFFLOAD=1

# Or use sequential offload (slowest, minimum VRAM)
export HFTOOL_CPU_OFFLOAD=2

# Use models designed for low VRAM
hftool -t t2i -m sdxl-turbo -i "A cat" -o cat.png  # 8GB friendly
```

### VRAM Estimation

| Task | Model | Approximate VRAM |
|------|-------|------------------|
| text-to-image | Z-Image | 8GB |
| text-to-image | SDXL | 10GB |
| text-to-image | FLUX.1-schnell | 16GB |
| text-to-video | LTX-2 | 16GB |
| text-to-video | HunyuanVideo | 40GB+ (multi-GPU) |

---

## Troubleshooting

### ROCm Not Detected

```bash
# Check ROCm installation
rocminfo

# Check PyTorch sees ROCm
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU name (should show AMD)
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### CUDA Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch sees CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory Errors

```bash
# Enable CPU offload
export HFTOOL_CPU_OFFLOAD=1

# Use multi-GPU if available
hftool -t t2v -i "A cat" -o cat.mp4 --gpu all

# Reduce batch size / resolution
hftool -t t2i -i "A cat" -o cat.png --width 512 --height 512
```

### Device Mismatch Errors

This usually happens with multi-GPU setups:

```bash
# Disable multi-GPU
export HFTOOL_MULTI_GPU=0

# Or use single GPU explicitly
hftool -t t2i -i "A cat" -o cat.png --gpu 0
```

### MIOpen Warnings (AMD)

These warnings are harmless but noisy:

```bash
# Suppress MIOpen warnings
export MIOPEN_LOG_LEVEL=4
```

---

## Docker GPU Passthrough

### AMD ROCm

```bash
docker run --device /dev/kfd --device /dev/dri \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  hftool:rocm -t t2i -i "A cat" -o /output/cat.png
```

### NVIDIA CUDA

```bash
docker run --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  hftool:cuda -t t2i -i "A cat" -o /output/cat.png
```

### Using hftool docker

The `hftool docker` commands handle this automatically:

```bash
hftool docker run -- -t t2i -i "A cat" -o ~/Pictures/cat.png
```
