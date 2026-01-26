# Environment Variables

Complete reference for all environment variables supported by hftool.

## Quick Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `HFTOOL_MODELS_DIR` | `~/.cache/huggingface` | Model download directory |
| `HFTOOL_CONFIG` | `~/.hftool` | Config directory path |
| `HFTOOL_AUTO_DOWNLOAD` | `0` | Auto-download models |
| `HFTOOL_DEBUG` | `0` | Enable debug output |
| `HFTOOL_LOG_FILE` | (none) | Log file path |
| `HFTOOL_QUIET` | `0` | Suppress output |
| `HFTOOL_GPU` | `auto` | GPU selection |
| `HFTOOL_MULTI_GPU` | `0` | Multi-GPU mode |
| `HFTOOL_CPU_OFFLOAD` | `0` | CPU offload level |
| `HF_TOKEN` | (none) | HuggingFace API token |

---

## Core Settings

### HFTOOL_MODELS_DIR

Directory where models are downloaded and cached.

```bash
export HFTOOL_MODELS_DIR=~/.cache/huggingface  # default
export HFTOOL_MODELS_DIR=/mnt/ssd/models       # custom location
```

**Shared with:** HuggingFace Hub, transformers, diffusers

### HFTOOL_CONFIG

Path to the hftool configuration directory.

```bash
export HFTOOL_CONFIG=~/.hftool  # default
```

**Contains:** `config.toml`, `history.json`, `benchmarks.json`

### HFTOOL_AUTO_DOWNLOAD

Automatically download models without confirmation prompts.

```bash
export HFTOOL_AUTO_DOWNLOAD=0  # prompt before download (default)
export HFTOOL_AUTO_DOWNLOAD=1  # auto-download
```

**Useful for:** Scripts, CI/CD pipelines, batch processing

### HFTOOL_DEBUG

Enable debug mode with verbose output and all warnings.

```bash
export HFTOOL_DEBUG=0  # normal output (default)
export HFTOOL_DEBUG=1  # verbose debug output
```

### HFTOOL_LOG_FILE

Write logs to a file in addition to console output.

```bash
export HFTOOL_LOG_FILE=~/.hftool/hftool.log
```

### HFTOOL_QUIET

Suppress informational output. Errors are still shown.

```bash
export HFTOOL_QUIET=0  # normal output (default)
export HFTOOL_QUIET=1  # minimal output
```

### HFTOOL_AUTO_OPEN

Automatically open generated files after creation.

```bash
export HFTOOL_AUTO_OPEN=0  # don't auto-open (default)
export HFTOOL_AUTO_OPEN=1  # auto-open in default app
```

### HFTOOL_INTERACTIVE

Start in interactive mode by default.

```bash
export HFTOOL_INTERACTIVE=0  # CLI mode (default)
export HFTOOL_INTERACTIVE=1  # interactive wizard mode
```

---

## GPU Configuration

### HFTOOL_GPU

GPU selection strategy.

```bash
export HFTOOL_GPU=auto   # auto-select best GPU (default)
export HFTOOL_GPU=all    # use all GPUs
export HFTOOL_GPU=0      # use first GPU
export HFTOOL_GPU=1      # use second GPU
export HFTOOL_GPU=0,1    # use specific GPUs
```

**Note:** `auto` avoids GPUs with displays connected to prevent VRAM conflicts.

### HFTOOL_MULTI_GPU

Enable multi-GPU model distribution for large models.

```bash
export HFTOOL_MULTI_GPU=0         # single GPU (default)
export HFTOOL_MULTI_GPU=1         # distribute across GPUs
export HFTOOL_MULTI_GPU=balanced  # same as 1
```

**Automatic:** Set when using `--gpu all`

**Use case:** Large models like HunyuanVideo that exceed single GPU VRAM

### HFTOOL_CPU_OFFLOAD

CPU offload level for memory-constrained systems.

```bash
export HFTOOL_CPU_OFFLOAD=0  # disabled (default)
export HFTOOL_CPU_OFFLOAD=1  # model CPU offload
export HFTOOL_CPU_OFFLOAD=2  # sequential CPU offload (slower, less VRAM)
```

**Level 1 (model offload):** Moves entire model components to CPU when not in use. Good balance of speed and memory.

**Level 2 (sequential offload):** Moves individual layers to CPU. Slowest but uses least VRAM.

---

## AMD ROCm Settings

### HFTOOL_ROCM_PATH

Custom ROCm installation path.

```bash
export HFTOOL_ROCM_PATH=/opt/rocm  # default
export HFTOOL_ROCM_PATH=/usr/share/ollama/lib/rocm  # Ollama's ROCm
```

**Use case:** Systems with multiple ROCm versions, Ollama integration

### HSA_OVERRIDE_GFX_VERSION

Override GPU architecture for unsupported AMD GPUs.

```bash
# RX 7900 XTX (officially supported, no override needed)
# export HSA_OVERRIDE_GFX_VERSION=11.0.0

# RX 6000 series (may need override)
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

**Reference:** [ROCm GPU Support](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)

### HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES

Select which AMD GPUs are visible to applications.

```bash
export HIP_VISIBLE_DEVICES=0      # first GPU only
export HIP_VISIBLE_DEVICES=1      # second GPU only
export HIP_VISIBLE_DEVICES=0,1    # both GPUs
```

**Note:** Both variables work; `HIP_VISIBLE_DEVICES` is more common.

### MIOPEN_LOG_LEVEL

Control MIOpen (ROCm deep learning library) logging verbosity.

```bash
export MIOPEN_LOG_LEVEL=1  # errors only
export MIOPEN_LOG_LEVEL=2  # warnings
export MIOPEN_LOG_LEVEL=3  # info
export MIOPEN_LOG_LEVEL=4  # silent (recommended)
```

**Note:** hftool sets this to 4 in Docker to suppress noisy warnings.

### PYTORCH_ALLOC_CONF

PyTorch memory allocator configuration.

```bash
export PYTORCH_ALLOC_CONF="expandable_segments:True"  # reduces fragmentation
```

**Note:** `PYTORCH_CUDA_ALLOC_CONF` is deprecated; use `PYTORCH_ALLOC_CONF`.

### TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL

Enable experimental AOTriton optimizations for RDNA3 GPUs.

```bash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1  # enable (default for RDNA3)
```

### TORCH_BLAS_PREFER_HIPBLASLT

BLAS library preference.

```bash
export TORCH_BLAS_PREFER_HIPBLASLT=0  # use hipBLAS (better for consumer GPUs)
export TORCH_BLAS_PREFER_HIPBLASLT=1  # use hipBLASLt (datacenter GPUs)
```

---

## NVIDIA CUDA Settings

### CUDA_VISIBLE_DEVICES

Select which NVIDIA GPUs are visible to applications.

```bash
export CUDA_VISIBLE_DEVICES=0      # first GPU only
export CUDA_VISIBLE_DEVICES=1      # second GPU only
export CUDA_VISIBLE_DEVICES=0,1    # both GPUs
```

---

## HuggingFace Settings

### HF_TOKEN / HUGGINGFACE_TOKEN

HuggingFace API token for accessing gated models.

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Required for:** FLUX.1, Llama, Qwen, and other gated models

**Get your token:** https://huggingface.co/settings/tokens

### HF_HOME

HuggingFace cache directory (shared by all HF libraries).

```bash
export HF_HOME=~/.cache/huggingface  # default
```

### HF_HUB_ENABLE_HF_TRANSFER

Enable fast downloads using hf_transfer.

```bash
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
```

**Note:** Requires `hf_transfer` package installed.

### HF_HUB_DISABLE_PROGRESS_BARS

Disable download progress bars.

```bash
export HF_HUB_DISABLE_PROGRESS_BARS=1  # useful for CI/CD
```

---

## Docker Settings

These are set automatically when running `hftool docker run`.

### HFTOOL_IN_DOCKER

Indicates running inside a Docker container.

```bash
# Set automatically by hftool docker run
export HFTOOL_IN_DOCKER=1
```

### HFTOOL_HOST_HOME

Path to host home directory (for path translation).

```bash
export HFTOOL_HOST_HOME=/home/host
```

### HFTOOL_REAL_HOME

Actual home directory path outside container.

```bash
export HFTOOL_REAL_HOME=/home/username
```

---

## Task-Specific Settings

### GLMTTS_PATH

Path to GLM-TTS model (Chinese TTS).

```bash
export GLMTTS_PATH=./GLM-TTS
```

---

## Example Configurations

### AMD ROCm (RX 7900 XTX)

```bash
# ~/.bashrc or ~/.zshrc
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
export HFTOOL_AUTO_DOWNLOAD=1
export MIOPEN_LOG_LEVEL=4
```

### NVIDIA (RTX 4090)

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
export HFTOOL_AUTO_DOWNLOAD=1
export CUDA_VISIBLE_DEVICES=0
```

### Low-VRAM System (8GB)

```bash
export HFTOOL_CPU_OFFLOAD=1
export HFTOOL_AUTO_DOWNLOAD=1
```

### Multi-GPU Workstation

```bash
export HFTOOL_MULTI_GPU=1
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

### CI/CD Pipeline

```bash
export HFTOOL_AUTO_DOWNLOAD=1
export HFTOOL_QUIET=1
export HFTOOL_AUTO_OPEN=0
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_TOKEN=$CI_HF_TOKEN
```
