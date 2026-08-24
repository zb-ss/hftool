# Installation

> [Back to README](../README.md)

## Quick Install (Docker - Recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/zb-ss/hftool/master/install.sh | bash
```

This auto-detects your GPU, builds a Docker image, and creates a wrapper at `~/.local/bin/hftool`. See [Docker Install](docker.md) for details and options.

---

## pip Install

```bash
pip install hftool
```

On first run, hftool will detect if PyTorch is missing or misconfigured and offer to install it for you:

```
============================================================
  hftool - First Time Setup
============================================================

Detected hardware:
  [✓] AMD GPU detected: Radeon RX 7900 XTX

Select PyTorch version to install:

  [1] NVIDIA GPU (CUDA)
  [2] AMD GPU (ROCm 6.2) (recommended)
  [3] Apple Silicon (MPS)
  [4] CPU only
  [5] Skip (install manually later)

Your choice [2]:
```

You can also run the setup wizard manually at any time:

```bash
hftool setup
```

## Install with Specific Features

```bash
# Text-to-Image (Z-Image, SDXL, FLUX)
pip install "hftool[with_t2i]"

# Text-to-Video (HunyuanVideo, CogVideoX, Wan2.2)
pip install "hftool[with_t2v]"

# Text-to-Speech — Kokoro (lightweight default)
pip install "hftool[with_tts_kokoro]"

# Text-to-Speech — Chatterbox (voice cloning + emotion control)
pip install "hftool[with_tts_chatterbox]"

# Text-to-Speech — Bark, MMS-TTS (legacy)
pip install "hftool[with_tts]"

# Voiceover Pipeline (script parsing + audio merging)
pip install "hftool[with_voiceover]"

# Speech-to-Text (Whisper)
pip install "hftool[with_stt]"

# All features
pip install "hftool[all]"
```

## Optional Dependencies

For enhanced user experience features:

```bash
# Interactive file picker and JSON builder
pip install InquirerPy

# Or for pipx:
pipx runpip hftool install InquirerPy
```

**Note:** Without InquirerPy, the `@` file picker and `--interactive` mode will not work, but all other features remain functional.

## System Requirements

- **Python**: >= 3.10
- **PyTorch**: >= 2.0 with CUDA/ROCm support
- **ffmpeg**: Required for video output and MP3 audio conversion
  ```bash
  # Ubuntu/Debian
  sudo apt install ffmpeg

  # macOS
  brew install ffmpeg

  # Arch Linux
  sudo pacman -S ffmpeg
  ```

## Development Install

```bash
git clone https://github.com/zb-ss/hftool
cd hftool

# Install PyTorch first (see Quick Install above for your platform)
pip install torch torchvision torchaudio  # or with ROCm/CPU index

# Then install hftool in dev mode
pip install -e ".[dev]"  # Includes pytest
```

## pipx Install (Isolated Environment)

```bash
# Install hftool
pipx install hftool[all]
```

**Important for AMD GPU users:** The install above pulls in CUDA PyTorch by default. Replace it with ROCm PyTorch:

```bash
# AMD ROCm - uninstall CUDA version and install ROCm version:
pipx runpip hftool uninstall torch torchvision torchaudio -y
pipx runpip hftool install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

For other platforms:
```bash
# NVIDIA (already installed by default, but to reinstall):
pipx runpip hftool install torch torchvision torchaudio

# CPU only:
pipx runpip hftool uninstall torch torchvision torchaudio -y
pipx runpip hftool install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Docker Install (Recommended for AMD ROCm)

See the full [Docker Guide](docker.md) for complete Docker installation and usage instructions.

Docker provides the easiest way to use hftool with full GPU support, especially for AMD users who want to keep their system clean for gaming.

**Benefits:**
- ROCm 7.1.1 isolated from system (won't affect gaming drivers)
- All dependencies pre-installed (no pip conflicts)
- Works on any Linux with Docker

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
