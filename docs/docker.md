# Docker Guide

> [Back to README](../README.md) | [Installation](installation.md)

Docker provides the easiest way to use hftool with full GPU support, especially for AMD users who want to keep their system clean for gaming.

**Benefits:**
- ROCm 7.1.1 isolated from system (won't affect gaming drivers)
- All dependencies pre-installed (no pip conflicts)
- Works on any Linux with Docker

## Docker Install

**Option 1: One-liner install (recommended)**
```bash
curl -fsSL https://raw.githubusercontent.com/zb-ss/hftool/master/install.sh | bash

# With options:
curl -fsSL ... | bash -s -- --platform rocm      # Force platform
curl -fsSL ... | bash -s -- --install-dir /usr/local/bin  # Custom dir
```

**Option 2: Manual setup via pip**
```bash
# Install hftool (thin CLI wrapper)
pip install hftool

# Run the setup wizard
hftool docker setup
```

The setup wizard will detect your GPU, build a Docker image, and configure hftool to run commands in the container.

## Manual Docker Commands

```bash
# Check Docker status
hftool docker status

# Build the image manually
hftool docker build

# Run commands in Docker
hftool docker run -- -t t2i -i "A cat" -o cat.png

# GPU selection (AMD ROCm - uses device passthrough for reliable isolation)
hftool docker run --gpu 1 -- -t t2v -i "A cat" -o cat.mp4      # Use specific GPU
hftool docker run --gpu auto -- -t t2i -i "A cat" -o cat.png   # Auto-select non-display GPU
hftool docker run --gpu 0,1 -- -t t2v -m ltx2 -i "A cat" -o cat.mp4  # Multi-GPU

# Output files auto-open on host after generation completes
# Use --no-open to disable
hftool docker run -- -t t2i -i "A cat" -o cat.png --no-open
```

## Docker GPU Selection (AMD ROCm)

For multi-GPU AMD systems, hftool uses device passthrough to pass only selected GPU(s) to the container. This is more reliable than environment variable isolation:

```bash
# Interactive GPU selection (shown when multiple GPUs detected)
hftool docker run -- -t t2i -i "A cat" -o cat.png
#   Available GPUs:
#     [0] AMD Radeon RX 7900 XTX, 24.0GB (display)
#     [1] AMD Radeon RX 7900 XTX, 24.0GB
#   GPU> 1

# Explicit selection
hftool docker run --gpu 1 -- -t t2v -i "A cat" -o cat.mp4
```

| Option | Description |
|--------|-------------|
| `--gpu auto` | Select best non-display GPU |
| `--gpu 0` | Use specific GPU by index |
| `--gpu 0,1` | Use multiple GPUs (multi-GPU mode) |
| (no option) | Interactive selection if multiple GPUs |

## Environment Variables Passed to Docker

These environment variables are automatically passed through to the container:

| Variable | Description |
|----------|-------------|
| `HFTOOL_MODELS_DIR` | Custom models directory (mounted to `/models`) |
| `HSA_OVERRIDE_GFX_VERSION` | AMD GPU architecture (e.g., `11.0.0` for RX 7900) |
| `HF_TOKEN` | HuggingFace token for gated models |
| `HFTOOL_DEBUG` | Enable debug output |
| `HFTOOL_LOG_FILE` | Log file path (directory is mounted) |

## Docker Files

| File | Purpose |
|------|---------|
| `docker/Dockerfile.rocm` | AMD ROCm 7.1.1 + PyTorch 2.9.1 |
| `docker/Dockerfile.cuda` | NVIDIA CUDA 12.4 |
| `docker/Dockerfile.cpu` | CPU-only fallback |
| `docker/docker-compose.yml` | Easy GPU passthrough |

See [docker/README.md](../docker/README.md) for detailed Docker documentation.
