# hftool Documentation

> [Back to README](../README.md)

## Getting Started

| Document | Description |
|----------|-------------|
| [Installation Guide](installation.md) | pip, pipx, Docker, and development setup |
| [Docker Guide](docker.md) | Docker setup, GPU passthrough, multi-GPU |
| [Hardware Guide](hardware.md) | VRAM requirements, multi-GPU, platform notes |
| [GPU Setup](gpu-setup.md) | AMD ROCm, NVIDIA CUDA, Apple MPS configuration |

## Using hftool

| Document | Description |
|----------|-------------|
| [CLI Reference](cli-reference.md) | Full command reference, aliases, config files |
| [Features & UX](features.md) | File picker, history, dry-run, completions, diagnostics |
| [Model Management](models.md) | Download, clean, gated models, storage, debugging |
| [Configuration](configuration.md) | Config file format, locations, and examples |
| [Environment Variables](environment.md) | Complete environment variable reference |

## Task Guides

| Document | Description |
|----------|-------------|
| [Text-to-Image](text-to-image.md) | Z-Image, SDXL, FLUX — examples and parameters |
| [Image-to-Image](image-to-image.md) | Qwen Edit, FLUX.2 Klein, SDXL — multi-image, params |
| [Text-to-Video](text-to-video.md) | LTX-2, HunyuanVideo, I2V — examples and parameters |
| [Text-to-Speech](text-to-speech.md) | Kokoro, Chatterbox, Bark — voice cloning guide |
| [Voiceover Pipeline](voiceover.md) | Auto-voiceover, re-voice, manual script, Docker workflow |
| [Speech-to-Text](speech-to-text.md) | Whisper transcription with timestamps and SRT |
| [Other Tasks](other-tasks.md) | LLMs, classification, detection, translation |

## Quick Links

- [.env.example](../.env.example) — Copy this to `.env` for your settings
- `hftool doctor` — Check your system setup
- `hftool -I` — Full interactive wizard
- `hftool --help` — CLI usage

## Need Help?

- Run `hftool doctor` for system diagnostics
- See [GitHub Issues](https://github.com/zb-ss/hftool/issues) for bug reports
