# hftool Docker Support

Docker provides the easiest way to run hftool with full GPU support, especially for AMD ROCm users who want to keep their system clean for gaming.

## Quick Start

```bash
# Interactive setup wizard (recommended)
hftool docker setup

# Or build manually
hftool docker build

# Run commands in Docker
hftool docker run -t t2i -i "A sunset over mountains" -o sunset.png
```

## Why Docker?

| Problem | Docker Solution |
|---------|----------------|
| ROCm version conflicts | Isolated ROCm 7.1.1 environment |
| PyTorch installation headaches | Pre-configured with correct wheels |
| System library conflicts | Won't affect gaming or other apps |
| Dependency hell | All packages pre-installed |

## Supported Platforms

| Platform | Image | GPU Access |
|----------|-------|------------|
| AMD ROCm | `hftool:rocm` | `/dev/kfd`, `/dev/dri` |
| NVIDIA CUDA | `hftool:cuda` | `--gpus all` |
| CPU Only | `hftool:cpu` | N/A |

**Note:** Apple Silicon (MPS) works best with native installation. Docker GPU passthrough is not well supported on macOS.

## Manual Docker Usage

### Building Images

```bash
# From the hftool repository root
cd /path/to/hftool

# Build for AMD ROCm
docker build -f docker/Dockerfile.rocm -t hftool:rocm .

# Build for NVIDIA CUDA
docker build -f docker/Dockerfile.cuda -t hftool:cuda .

# Build for CPU
docker build -f docker/Dockerfile.cpu -t hftool:cpu .
```

### Running Containers

#### AMD ROCm

```bash
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  --group-add render \
  --shm-size 16g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.hftool:/root/.hftool \
  -v $(pwd):/workspace \
  -w /workspace \
  -e HF_TOKEN=$HF_TOKEN \
  hftool:rocm \
  -t t2i -i "A cat" -o cat.png
```

#### NVIDIA CUDA

```bash
docker run --rm -it \
  --gpus all \
  --shm-size 16g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.hftool:/root/.hftool \
  -v $(pwd):/workspace \
  -w /workspace \
  -e HF_TOKEN=$HF_TOKEN \
  hftool:cuda \
  -t t2i -i "A cat" -o cat.png
```

### Docker Compose

```bash
cd /path/to/hftool/docker

# AMD ROCm
docker compose --profile rocm run --rm hftool-rocm -t t2i -i "A cat" -o /workspace/cat.png

# NVIDIA CUDA
docker compose --profile cuda run --rm hftool-cuda -t t2i -i "A cat" -o /workspace/cat.png

# Interactive mode
docker compose --profile rocm run --rm hftool-rocm -I
```

## Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `~/.cache/huggingface` | `/root/.cache/huggingface` | Model cache (shared with host) |
| `~/.hftool` | `/root/.hftool` | Config and history |
| `$(pwd)` | `/workspace` | Input/output files |

## Environment Variables

These are automatically passed through when using `hftool docker run`:

| Variable | Description |
|----------|-------------|
| `HFTOOL_MODELS_DIR` | Custom models directory (mounted to `/models` in container) |
| `HSA_OVERRIDE_GFX_VERSION` | AMD GPU architecture (e.g., `11.0.0` for RX 7900) |
| `HF_TOKEN` | HuggingFace token for gated models |
| `HFTOOL_DEBUG` | Enable debug output (`1` to enable) |
| `HFTOOL_LOG_FILE` | Log file path (directory is mounted) |
| `HFTOOL_AUTO_DOWNLOAD` | Auto-download models (default: `1` in Docker) |

## GPU-Specific Notes

### AMD ROCm

The ROCm container uses ROCm 7.1.1 with PyTorch 2.9.1. Key requirements:

- Host must have `amdgpu` kernel driver installed (standard for AMD GPUs)
- No system ROCm installation needed - container provides it
- Works with RX 7900 XTX/XT, RX 7800 XT, and other supported GPUs

For RDNA3 GPUs (RX 7000 series), you may need:
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # For gfx1100 (7900 XTX/XT)
```

### NVIDIA CUDA

Requires:
- NVIDIA driver installed on host
- nvidia-container-toolkit installed
- Docker configured for GPU access

## Troubleshooting

### "permission denied" for /dev/kfd

Add your user to the video and render groups:
```bash
sudo usermod -aG video,render $USER
# Log out and back in
```

### Out of memory errors

Increase shared memory:
```bash
docker run --shm-size 32g ...
```

### Slow first run

First run downloads model weights (~5-20GB depending on model). Subsequent runs use cached models from `~/.cache/huggingface`.

### Image build fails

Ensure you're building from the hftool repository root:
```bash
cd /path/to/hftool
docker build -f docker/Dockerfile.rocm -t hftool:rocm .
```
