# Hardware & GPU Guide

> [Back to README](../README.md) | [GPU Setup](gpu-setup.md)

## Table of Contents
- [Hardware Recommendations](#hardware-recommendations)
- [AMD ROCm](#amd-rocm-primary-target)
- [ROCm Setup Without System-Wide Installation](#rocm-setup-without-system-wide-installation)
- [NVIDIA CUDA](#nvidia-cuda)
- [Apple Silicon](#apple-silicon-mps)
- [CPU](#cpu)
- [Multi-GPU Support](#multi-gpu-support)

---

## Hardware Recommendations

### AMD ROCm (Primary Target)

hftool is optimized for AMD GPUs with ROCm 6.x:

| Task | Model | VRAM Required | Notes |
|------|-------|---------------|-------|
| Text-to-Image | Z-Image-Turbo | ~10-12 GB | Comfortable on RX 7900 XTX |
| Image-to-Image | Qwen Image Edit | ~20-24 GB | Use CPU offload on 24GB cards |
| Image-to-Image | FLUX.2 Klein | ~29 GB | RTX 4090+, non-commercial |
| Image-to-Image | SDXL Refiner | ~8-10 GB | Fast, lower VRAM |
| Text-to-Video | LTX-2 | ~40 GB | Use `--gpu all` for multi-GPU |
| Text-to-Video | HunyuanVideo 480p | ~20-24 GB | Use CPU offload |
| Text-to-Video | HunyuanVideo 720p | ~30-40 GB | Requires multi-GPU |
| Text-to-Speech | Kokoro (default) | <1 GB | Runs on CPU |
| Text-to-Speech | Chatterbox | ~4-6 GB | Voice cloning |
| Text-to-Speech | Bark | ~2-4 GB | Legacy |
| Voiceover | Kokoro + FFmpeg | <1 GB | Script → audio → video |
| Voiceover | Chatterbox + FFmpeg | ~4-6 GB | With voice cloning |
| Speech-to-Text | Whisper-large-v3 | ~4-6 GB | Easy |

---

## ROCm Setup Without System-Wide Installation

If you have [Ollama](https://ollama.com) installed, you can use its bundled ROCm libraries instead of installing ROCm system-wide (which can interfere with gaming GPU drivers).

**Step 1:** Install PyTorch ROCm in your hftool environment:

```bash
# If using pipx:
pipx runpip hftool uninstall torch torchvision torchaudio -y
pipx runpip hftool install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# If using pip:
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

**Step 2:** Add ROCm configuration to your `.env` file (`~/.hftool/.env` or project directory):

```bash
# Use Ollama's bundled ROCm libraries
HFTOOL_ROCM_PATH=/usr/local/lib/ollama/rocm

# Set your GPU architecture (required for AMD GPUs)
# RDNA3: gfx1100 (RX 7900 XTX/XT), gfx1101 (RX 7800/7700), gfx1102 (RX 7600)
# RDNA2: gfx1030 (RX 6900/6800), gfx1031 (RX 6700), gfx1032 (RX 6600)
HSA_OVERRIDE_GFX_VERSION=11.0.0
```

**Step 3:** Verify GPU detection:

```bash
hftool -t t2i -i "test" -o test.png -v
# Should show "Using device: cuda" or similar
```

---

## NVIDIA CUDA

Works with CUDA 11.8+ and modern NVIDIA GPUs.

---

## Apple Silicon (MPS)

Basic support for M1/M2/M3 Macs. Some models may require `--dtype float32`.

---

## CPU

Works but slow. Use smaller models:
- `openai/whisper-small` for ASR
- `hexgrad/Kokoro-82M` for TTS (or `ResembleAI/chatterbox` for voice cloning)

---

## Multi-GPU Support

For systems with multiple GPUs (e.g., dual RX 7900 XTX), hftool can automatically detect which GPU has your display connected and route compute workloads to the other GPU. This prevents VRAM conflicts that can crash your desktop compositor.

**Check your GPUs:**

```bash
hftool doctor
# Shows:
#   GPU 0: AMD Radeon RX 7900 XTX [DISPLAY]
#   GPU 1: AMD Radeon RX 7900 XTX <- recommended
```

**GPU Selection Options:**

| Option | Description |
|--------|-------------|
| `--gpu auto` | Smart selection - uses GPU without display (default behavior) |
| `--gpu 0` | Use specific GPU by index |
| `--gpu 1` | Use specific GPU by index |
| `--gpu 0,1` | Use multiple specific GPUs |
| `--gpu all` | Use all GPUs with model parallelism (distributes model across GPUs) |

**How `--gpu all` works:**

When you select `--gpu all`, hftool uses `device_map="balanced"` to automatically distribute model layers across all available GPUs. This is essential for large models that don't fit in a single GPU's VRAM. The centralized multi-GPU logic in `hftool/core/device.py` ensures consistent behavior across all task types (text-to-image, text-to-video, image-to-image, etc.).

**Examples:**

```bash
# Auto-select compute GPU (avoids display GPU)
hftool -t t2v -i "A cat running" -o cat.mp4 --gpu auto

# Use specific GPU
hftool -t t2v -i "A cat running" -o cat.mp4 --gpu 1

# Use all GPUs for large models like LTX-2 or HunyuanVideo
hftool -t t2v -m ltx2 -i "A cat running" -o cat.mp4 --gpu all

# Docker with specific GPU
hftool docker run --gpu 1 -- -t t2v -i "A cat" -o cat.mp4

# Environment variable (useful in .env file)
HFTOOL_GPU=1 hftool -t t2v -i "A cat" -o cat.mp4

# Force multi-GPU mode via environment variable
HFTOOL_MULTI_GPU=1 hftool -t t2v -m ltx2 -i "A cat" -o cat.mp4
```

**Interactive Mode:** When using `hftool -I`, the wizard will show GPU selection with display detection for multi-GPU systems.
