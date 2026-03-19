# hftool

A CLI for running HuggingFace models, optimized for AMD ROCm.

> **What this is:** A convenient wrapper for common AI tasks (image/video/speech generation, transcription). Not a replacement for transformers or diffusers, but a simpler interface when you just want to run a model without writing Python.
>
> **Who it's for:** AMD GPU owners frustrated with CUDA-first tooling, and anyone who wants a unified CLI for multiple AI modalities.

## Demo

[![Watch hftool interactive mode demo](https://img.youtube.com/vi/oYANVmglEhs/maxresdefault.jpg)](https://www.youtube.com/watch?v=oYANVmglEhs)

## Supported Tasks

| Task | Alias | Models | Docs |
|------|-------|--------|------|
| Text-to-Image | `t2i` | Z-Image-Turbo, SDXL, FLUX | [Guide](docs/text-to-image.md) |
| Image-to-Image | `i2i` | Qwen Image Edit, FLUX.2 Klein, SDXL | [Guide](docs/image-to-image.md) |
| Text-to-Video | `t2v` | LTX-2, HunyuanVideo, CogVideoX, Wan2.2 | [Guide](docs/text-to-video.md) |
| Image-to-Video | `i2v` | LTX-2 I2V, HunyuanVideo I2V, Wan2.2 I2V | [Guide](docs/text-to-video.md) |
| Voiceover | `vo` | Auto-voiceover (VLM), re-voice (ASR), manual script | [Guide](docs/voiceover.md) |
| Text-to-Speech | `tts` | Kokoro, Chatterbox (voice cloning), Bark | [Guide](docs/text-to-speech.md) |
| Speech-to-Text | `asr` | Whisper (with timestamps, SRT export) | [Guide](docs/speech-to-text.md) |
| Vision-Language | `vlm` | Qwen 3.5 (9B, 4B, 27B), InternVL 3.5 | [Guide](docs/voiceover.md#vlm-model-options) |
| Text Generation | `llm` | Llama, Mistral, Qwen | [Guide](docs/other-tasks.md) |
| + more | | Classification, detection, translation, summarization | [Guide](docs/other-tasks.md) |

Works on **AMD ROCm**, NVIDIA CUDA, Apple MPS, and CPU.

## Quick Install

```bash
# Docker (recommended for AMD ROCm)
curl -fsSL https://raw.githubusercontent.com/zb-ss/hftool/master/install.sh | bash

# pip
pip install hftool

# All features
pip install "hftool[all]"
```

On first run, hftool detects your hardware and helps install the right PyTorch version. See the full [Installation Guide](docs/installation.md) for pip, pipx, Docker, and development setup.

## Quick Start

```bash
# Full interactive wizard
hftool -I

# Generate an image
hftool -t t2i -i "A cat in space" -o cat.png

# Generate a video
hftool -t t2v -i "A cat playing with a ball" -o cat.mp4

# Transcribe audio
hftool -t asr -i recording.wav -o transcript.txt

# Interactive TUI (runs in Docker — no setup needed)
hftool tui

# Auto-voiceover a video
hftool voiceover --auto --video demo.mp4 --output final.mp4

# Re-voice existing narration with a different voice
hftool voiceover --revoice --video tutorial.mp4 --output revoiced.mp4

# Voiceover from a script
hftool voiceover --script timing.srt --video input.mp4 --output final.mp4

# Preview without running
hftool -t t2i -i "A cat" --dry-run

# Interactive file selection
hftool -t asr -i @ -o transcript.txt
```

## Features

- **TUI** (`hftool tui`) — Full-screen terminal UI, runs in Docker with zero setup
- **Interactive Wizard** (`-I`) — Guided task, model, and input selection
- **File Picker** (`@` syntax) — Interactive file selection with fuzzy search
- **Auto-Voiceover** — VLM-powered video analysis + narration generation
- **Voice Cloning** — Clone any voice with Chatterbox TTS
- **Command History** — View and re-run previous commands
- **Dry-Run Mode** — Preview operations before executing
- **Shell Completions** — Tab completion for bash, zsh, fish
- **Model Management** — Download, list, and clean up models
- **Docker Support** — Isolated GPU environments for AMD ROCm
- **Multi-GPU** — Automatic display GPU detection and model parallelism

## Documentation

### Getting Started

| Document | Description |
|----------|-------------|
| [Installation Guide](docs/installation.md) | pip, pipx, Docker, and development setup |
| [Docker Guide](docs/docker.md) | Docker setup, GPU passthrough, multi-GPU |
| [Hardware Guide](docs/hardware.md) | VRAM requirements, multi-GPU, platform notes |
| [GPU Setup](docs/gpu-setup.md) | AMD ROCm, NVIDIA CUDA, Apple MPS configuration |

### Using hftool

| Document | Description |
|----------|-------------|
| [CLI Reference](docs/cli-reference.md) | Full command reference, aliases, config files |
| [Features & UX](docs/features.md) | File picker, history, dry-run, completions, diagnostics |
| [Model Management](docs/models.md) | Download, clean, gated models, storage, debugging |
| [Configuration](docs/configuration.md) | Config file format, locations, and examples |
| [Environment Variables](docs/environment.md) | Complete environment variable reference |

### Task Guides

| Document | Description |
|----------|-------------|
| [Text-to-Image](docs/text-to-image.md) | Z-Image, SDXL, FLUX — examples and parameters |
| [Image-to-Image](docs/image-to-image.md) | Qwen Edit, FLUX.2 Klein, SDXL — multi-image, params |
| [Text-to-Video](docs/text-to-video.md) | LTX-2, HunyuanVideo, I2V — examples and parameters |
| [Text-to-Speech](docs/text-to-speech.md) | Kokoro, Chatterbox, Bark — voice cloning guide |
| [Voiceover Pipeline](docs/voiceover.md) | Auto-voiceover, re-voice, manual script, Docker workflow |
| [Speech-to-Text](docs/speech-to-text.md) | Whisper transcription with timestamps and SRT |
| [Other Tasks](docs/other-tasks.md) | LLMs, classification, detection, translation |

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT License

## Links

- [GitHub Repository](https://github.com/zb-ss/hftool)
- [Report a Bug](https://github.com/zb-ss/hftool/issues/new?template=bug_report.md)
- [Request a Model](https://github.com/zb-ss/hftool/issues/new?template=model_request.md)
