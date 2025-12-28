# hftool

A CLI for running HuggingFace models, optimized for AMD ROCm.

> **What this is:** A convenient wrapper for common AI tasks (image/video/speech generation, transcription). Not a replacement for transformers or diffusers, but a simpler interface when you just want to run a model without writing Python.
>
> **Who it's for:** AMD GPU owners frustrated with CUDA-first tooling, and anyone who wants a unified CLI for multiple AI modalities.

## Features

- **Text-to-Image**: Z-Image-Turbo, Stable Diffusion XL, FLUX
- **Image-to-Image**: Qwen Image Edit (advanced editing with multi-image support), SDXL Refiner
- **Text-to-Video**: HunyuanVideo-1.5, CogVideoX, Wan2.2
- **Text-to-Speech**: Bark, MMS-TTS, GLM-TTS
- **Speech-to-Text**: Whisper (with timestamps and SRT export)
- **Plus**: Text generation, classification, translation, and more via transformers pipelines
- **Model Management**: Download, list, and clean up models with simple commands
- **Auto-Setup**: Detects your hardware and helps install the right PyTorch version

Works on **AMD ROCm**, NVIDIA CUDA, Apple MPS, and CPU.

## Installation

### Quick Install

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

### Install with Specific Features

```bash
# Text-to-Image (Z-Image, SDXL, FLUX)
pip install "hftool[with_t2i]"

# Text-to-Video (HunyuanVideo, CogVideoX, Wan2.2)
pip install "hftool[with_t2v]"

# Text-to-Speech (Bark, MMS-TTS)
pip install "hftool[with_tts]"

# Speech-to-Text (Whisper)
pip install "hftool[with_stt]"

# All features
pip install "hftool[all]"
```

### System Requirements

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

### Development Install

```bash
git clone https://github.com/zb-ss/hftool
cd hftool

# Install PyTorch first (see Quick Install above for your platform)
pip install torch torchvision torchaudio  # or with ROCm/CPU index

# Then install hftool in dev mode
pip install -e ".[dev]"  # Includes pytest
```

### pipx Install (Isolated Environment)

```bash
# Install hftool
pipx install hftool[all]

# Then inject the correct PyTorch for your platform:
# NVIDIA:
pipx runpip hftool install torch torchvision torchaudio

# AMD ROCm:
pipx runpip hftool install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# CPU only:
pipx runpip hftool install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Quick Start

```bash
# Generate an image (auto-opens when done!)
hftool -t t2i -i "A cat in space" -o cat.png

# Generate speech
hftool -t tts -i "Hello world" -o hello.wav

# Transcribe audio
hftool -t asr -i recording.wav -o transcript.txt
```

**Auto-open feature**: By default, generated images, audio, and video files automatically open in your system's default application when complete!

When you run a task for the first time, hftool will prompt you to download the required model:

```
============================================================
Model not found: Z-Image Turbo
============================================================

  Task:     text-to-image
  Model:    Z-Image Turbo
  Repo:     Tongyi-MAI/Z-Image-Turbo
  Size:     ~6.0 GB
  Location: /home/user/.hftool/models/Tongyi-MAI--Z-Image-Turbo

Download this model now? [Y/n]:
```

---

## Model Management

### List Available Models

```bash
# List all models
hftool models

# List models for a specific task
hftool models -t text-to-image
hftool models -t t2i  # (using alias)

# Show only downloaded models
hftool models --downloaded

# Output as JSON
hftool models --json
```

### Download Models

```bash
# Download default model for a task
hftool download -t text-to-image
hftool download -t t2i  # (using alias)

# Download specific model by short name
hftool download -t t2i -m sdxl

# Download by HuggingFace repo_id
hftool download -m openai/whisper-large-v3

# Download all default models for all tasks
hftool download --all

# Re-download (force)
hftool download -t t2i -f
```

### Check Status

```bash
# Show downloaded models and disk usage
hftool status
```

### Clean Up

```bash
# Interactive selection (default) - shows numbered list to choose from
hftool clean

# Delete specific model by name
hftool clean -m whisper-large-v3

# Delete multiple models at once
hftool clean -m whisper-large-v3 -m z-image-turbo

# Delete all downloaded models
hftool clean --all

# Skip confirmation prompts
hftool clean --all -y
```

**Interactive selection example:**
```
Downloaded models:
------------------------------------------------------------
  [ 1] Whisper Large v3 (automatic-speech-recognition)
       openai/whisper-large-v3 - 3.1 GB
  [ 2] Z-Image Turbo (text-to-image)
       Tongyi-MAI/Z-Image-Turbo - 6.0 GB
------------------------------------------------------------

Enter model numbers to delete (comma-separated, ranges with -, or 'all'):
Examples: 1,3,5  or  1-3  or  1,3-5,7  or  all

Selection []: 1,2
```

### Custom Storage Location

By default, models are stored in `~/.hftool/models/`. You can customize this:

```bash
# Set custom location via environment variable
export HFTOOL_MODELS_DIR=/path/to/models

# Or use one-time
HFTOOL_MODELS_DIR=/mnt/storage hftool -t t2i -i "A cat" -o cat.png
```

**Using a `.env` file** (recommended):

Create a `.env` file in your project directory or `~/.hftool/.env`:

```bash
# .env
HFTOOL_MODELS_DIR=/data/models
HFTOOL_AUTO_DOWNLOAD=1
HFTOOL_AUTO_OPEN=0
HFTOOL_DEBUG=0          # Set to 1 to show all warnings
```

hftool automatically loads `.env` files on startup.

### Debug Mode and Logging

By default, hftool suppresses noisy warnings from dependencies (torch, diffusers, transformers). To see all warnings for debugging:

```bash
# Via environment variable
HFTOOL_DEBUG=1 hftool -t i2i -i '{"image": "photo.jpg", "prompt": "..."}'

# Or in .env file
HFTOOL_DEBUG=1
```

**File Logging**: Save all warnings and debug info to a log file:

```bash
# Via environment variable
HFTOOL_LOG_FILE=~/.hftool/hftool.log hftool -t i2i ...

# Or in .env file (recommended)
HFTOOL_LOG_FILE=~/.hftool/hftool.log
```

The log file captures all warnings, errors, and debug info even when `HFTOOL_DEBUG=0`. Useful for troubleshooting issues without cluttering the terminal.

### Auto-Download Mode

To skip interactive prompts and auto-download models:

```bash
export HFTOOL_AUTO_DOWNLOAD=1
```

### Auto-Open Output Files

By default, generated images, audio, and video files automatically open in your system's default application when complete. Control this with:

```bash
# Always open (even text files)
hftool -t t2i -i "A cat" -o cat.png --open

# Never open
hftool -t t2i -i "A cat" -o cat.png --no-open

# Or set via environment variable
export HFTOOL_AUTO_OPEN=1    # Always open
export HFTOOL_AUTO_OPEN=0    # Never open
```

**Default behavior**: Auto-opens image, audio, and video files. Text output is printed to console.

---

## Usage

### Basic Syntax

```bash
hftool -t <task> -i <input> [-m <model>] [-o <output>] [-- extra_args]
```

### List Available Tasks

```bash
hftool --list-tasks
```

### Task Aliases

| Alias | Full Name |
|-------|-----------|
| `t2i` | text-to-image |
| `i2i`, `img2img` | image-to-image |
| `t2v` | text-to-video |
| `i2v` | image-to-video |
| `tts` | text-to-speech |
| `asr`, `stt` | automatic-speech-recognition |
| `llm` | text-generation |

---

## Examples

### Text-to-Image

Generate images with Z-Image-Turbo (state-of-the-art open-source model):

```bash
# Basic usage (uses default model)
hftool -t t2i -i "A cat wearing a space helmet" -o cat_space.png

# With specific model
hftool -t t2i -m Tongyi-MAI/Z-Image-Turbo \
       -i "A photorealistic sunset over mountains" \
       -o sunset.png

# With custom parameters (Z-Image-Turbo uses 9 steps, guidance_scale=0)
hftool -t t2i -m Tongyi-MAI/Z-Image-Turbo \
       -i "A renaissance painting of a robot" \
       -o robot.png \
       -- --num_inference_steps 9 --guidance_scale 0.0 --height 1024 --width 1024
```

**Other supported models:**
- `stabilityai/stable-diffusion-xl-base-1.0`
- `black-forest-labs/FLUX.1-schnell`

---

### Image-to-Image

Transform existing images with Qwen Image Edit (default) or SDXL:

```bash
# Basic image editing with Qwen Image Edit (default)
hftool -t i2i \
       -i '{"image": "photo.jpg", "prompt": "turn this into a watercolor painting"}' \
       -o watercolor.png

# Multi-image editing - combine multiple images (Qwen feature)
hftool -t i2i \
       -i '{"image": ["person1.jpg", "person2.jpg"], "prompt": "Both people standing together in a park"}' \
       -o combined.png

# With custom parameters
hftool -t i2i \
       -i '{"image": "portrait.jpg", "prompt": "as a Renaissance painting"}' \
       -o renaissance.png \
       -- --seed 42 --true_cfg_scale 4.0 --num_inference_steps 50

# Style transfer with SDXL Refiner (smaller model, faster)
hftool -t i2i -m sdxl-refiner \
       -i '{"image": "landscape.jpg", "prompt": "professional photography, enhanced colors"}' \
       -o enhanced.png \
       -- --strength 0.3
```

**Supported models:**
- `Qwen/Qwen-Image-Edit-2511` (default, 25 GB) - Advanced editing with character consistency, multi-image support
- `stabilityai/stable-diffusion-xl-refiner-1.0` (6.2 GB) - Fast refinement and subtle changes
- `stabilityai/stable-diffusion-xl-base-1.0` (6.5 GB) - Stronger style transfer

**Input format:** JSON with `image` (path or list of paths) and `prompt` (edit description)

**Qwen Image Edit parameters** (pass after `--`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--seed` | random | Random seed for reproducibility |
| `--true_cfg_scale` | 4.0 | True CFG scale (higher = stronger prompt adherence) |
| `--num_inference_steps` | 40 | Number of denoising steps |
| `--guidance_scale` | 1.0 | Standard CFG guidance scale |
| `--negative_prompt` | " " | What to avoid in generation |

**SDXL Refiner/Base parameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--seed` | random | Random seed for reproducibility |
| `--strength` | 0.3-0.7 | How much to change the image (0.0-1.0) |
| `--num_inference_steps` | 30 | Number of denoising steps |
| `--guidance_scale` | 7.5 | CFG guidance scale |

**Qwen Image Edit features:**
- Character consistency: Preserves identity in imaginative edits
- Multi-image input: Combine multiple images into one scene
- Industrial design: Batch product design and material replacement
- Geometric reasoning: Generate auxiliary construction lines

**Memory requirements:** Qwen Image Edit requires ~25GB VRAM. For GPUs with less memory:

```bash
# Use multi-GPU (distributes across available GPUs)
HFTOOL_MULTI_GPU=1 hftool -t i2i -i '{"image": "photo.jpg", "prompt": "..."}' -o out.png

# Use CPU offload (slower but works on 16-24GB GPUs)
HFTOOL_CPU_OFFLOAD=1 hftool -t i2i -i '{"image": "photo.jpg", "prompt": "..."}' -o out.png

# Use sequential CPU offload (most memory efficient, slowest)
HFTOOL_CPU_OFFLOAD=2 hftool -t i2i -i '{"image": "photo.jpg", "prompt": "..."}' -o out.png
```

**Note:** Qwen Image Edit requires diffusers >= 0.36.0. Upgrade with:
```bash
pip install --upgrade diffusers>=0.36.0
# Or for pipx:
pipx runpip hftool install --upgrade diffusers>=0.36.0
```

---

### Text-to-Video

Generate videos with HunyuanVideo-1.5:

```bash
# Basic usage (480p, ~2.5 second video)
hftool -t t2v -i "A person walking on a beach at sunset" -o beach.mp4

# With specific model and parameters
hftool -t t2v -m hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \
       -i "A timelapse of clouds moving over a city" \
       -o clouds.mp4 \
       -- --num_frames 61 --num_inference_steps 30

# Image-to-Video (animate an image)
hftool -t i2v -m hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v \
       -i '{"image": "photo.jpg", "prompt": "The person waves hello"}' \
       -o animated.mp4
```

**Other supported models:**
- `THUDM/CogVideoX-5b`
- `Wan-AI/Wan2.1-T2V-1.3B`

**Note:** Requires system `ffmpeg` for video encoding.

---

### Text-to-Speech

Generate speech with Bark:

```bash
# Basic usage (uses bark-small by default)
hftool -t tts -i "Hello, this is a test of the text to speech system." -o hello.wav

# With full Bark model (higher quality, larger)
hftool -t tts -m suno/bark \
       -i "Welcome to hftool, your command-line AI assistant." \
       -o welcome.wav

# Output as MP3 (requires ffmpeg)
hftool -t tts -i "This will be saved as MP3." -o output.mp3
```

**Supported models:**
- `suno/bark-small` (default, 1.5 GB, fast)
- `suno/bark` (5 GB, full quality, multi-language, sound effects)
- `facebook/mms-tts-eng` (0.3 GB, lightweight)

#### GLM-TTS Setup (Advanced)

GLM-TTS requires manual installation:

```bash
# Clone the repository
git clone https://github.com/zai-org/GLM-TTS.git
cd GLM-TTS && pip install -r requirements.txt

# Set environment variable
export GLMTTS_PATH=/path/to/GLM-TTS

# Run
hftool -t tts -m zai-org/GLM-TTS -i "你好世界" -o hello_chinese.wav
```

---

### Speech-to-Text (ASR)

Transcribe audio with Whisper:

```bash
# Basic transcription
hftool -t asr -i recording.wav -o transcript.txt

# With specific model
hftool -t asr -m openai/whisper-large-v3 -i podcast.mp3 -o transcript.txt

# With timestamps (outputs JSON)
hftool -t asr -i interview.wav -o transcript.json \
       -- --return_timestamps true

# Generate SRT subtitles
hftool -t asr -i video_audio.wav -o subtitles.srt \
       -- --return_timestamps true --format srt
```

**Supported models:**
- `openai/whisper-large-v3` (best quality)
- `openai/whisper-medium`
- `openai/whisper-small` (fastest)

---

### Text Generation (LLMs)

Run language models:

```bash
# Basic generation
hftool -t llm -m meta-llama/Llama-3.2-1B-Instruct \
       -i "Explain quantum computing in simple terms:" \
       -o response.txt \
       -- --max_new_tokens 200
```

---

### Other Tasks

```bash
# Image Classification
hftool -t image-classification -m google/vit-base-patch16-224 \
       -i photo.jpg -o result.json

# Object Detection
hftool -t object-detection -m facebook/detr-resnet-50 \
       -i street.jpg -o detections.json

# Summarization
hftool -t summarization -m facebook/bart-large-cnn \
       -i article.txt -o summary.txt

# Translation
hftool -t translation -m Helsinki-NLP/opus-mt-en-de \
       -i "Hello, how are you?" -o translation.txt
```

---

## CLI Reference

### Main Command

```
Usage: hftool [OPTIONS] COMMAND [ARGS]...

Options:
  -t, --task TEXT         Task to perform
  -m, --model TEXT        Model name/path (uses task default if omitted)
  -i, --input TEXT        Input data: text, file path, or URL
  -o, --output-file TEXT  Output file path (auto-generated if omitted)
  -d, --device TEXT       Device: auto, cuda, mps, cpu (default: auto)
  --dtype TEXT            Data type: bfloat16, float16, float32
  --open / --no-open      Open output with default app (auto for media files)
  --list-tasks            List all available tasks and aliases
  -v, --verbose           Show detailed progress
  --help                  Show this message and exit

Commands:
  setup     Run interactive PyTorch setup wizard
  models    List available models for tasks
  download  Download models from HuggingFace Hub
  status    Show download status and disk usage
  clean     Delete downloaded models
  run       Run a task (alternative to -t flag)
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HFTOOL_MODELS_DIR` | Custom models storage directory | `~/.hftool/models/` |
| `HFTOOL_AUTO_DOWNLOAD` | Auto-download models without prompting | `0` (disabled) |
| `HFTOOL_AUTO_OPEN` | Auto-open output files | `auto` (media files only) |
| `HFTOOL_ROCM_PATH` | Path to ROCm libraries (e.g., Ollama's bundled ROCm) | (none) |
| `HSA_OVERRIDE_GFX_VERSION` | AMD GPU architecture override (e.g., `11.0.0` for RX 7900) | (none) |

### Passing Model-Specific Arguments

Use `--` to pass additional arguments to the underlying model:

```bash
hftool -t t2i -i "A cat" -o cat.png \
       -- --num_inference_steps 20 --guidance_scale 7.5 --seed 42
```

---

## Hardware Recommendations

### AMD ROCm (Primary Target)

hftool is optimized for AMD GPUs with ROCm 6.x:

| Task | Model | VRAM Required | Notes |
|------|-------|---------------|-------|
| Text-to-Image | Z-Image-Turbo | ~10-12 GB | Comfortable on RX 7900 XTX |
| Image-to-Image | Qwen Image Edit | ~20-24 GB | Use CPU offload on 24GB cards |
| Image-to-Image | SDXL Refiner | ~8-10 GB | Fast, lower VRAM |
| Text-to-Video | HunyuanVideo 480p | ~20-24 GB | Use CPU offload |
| Text-to-Video | HunyuanVideo 720p | ~30-40 GB | Requires multi-GPU |
| Text-to-Speech | Bark | ~2-4 GB | Easy |
| Speech-to-Text | Whisper-large-v3 | ~4-6 GB | Easy |

#### ROCm Setup (Without System-Wide Installation)

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

### NVIDIA CUDA

Works with CUDA 11.8+ and modern NVIDIA GPUs.

### Apple Silicon (MPS)

Basic support for M1/M2/M3 Macs. Some models may require `--dtype float32`.

### CPU

Works but slow. Use smaller models:
- `openai/whisper-small` for ASR
- `suno/bark-small` for TTS

---

## Project Structure

```
hftool/
├── cli.py              # CLI entry point with subcommands
├── core/
│   ├── device.py       # ROCm/CUDA/MPS/CPU detection
│   ├── registry.py     # Task registry and configuration
│   ├── models.py       # Model registry with download metadata
│   └── download.py     # Model download manager
├── tasks/
│   ├── base.py         # Abstract base task class
│   ├── text_to_image.py
│   ├── image_to_image.py
│   ├── text_to_video.py
│   ├── text_to_speech.py
│   ├── speech_to_text.py
│   └── transformers_generic.py
├── io/
│   ├── input_loader.py # Input handling
│   └── output_handler.py # Output handling (ffmpeg)
└── utils/
    └── deps.py         # Dependency checking
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

MIT License

---

## Links

- [GitHub Repository](https://github.com/zb-ss/hftool)
- [Contributing Guide](CONTRIBUTING.md)
- [Hugging Face](https://huggingface.co)

### Model References

- [Z-Image](https://github.com/Tongyi-MAI/Z-Image) - State-of-the-art text-to-image
- [Qwen Image Edit](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) - Advanced image editing with character consistency
- [HunyuanVideo-1.5](https://huggingface.co/tencent/HunyuanVideo-1.5) - High-quality video generation
- [Bark](https://huggingface.co/suno/bark) - High-quality TTS with sound effects
- [Whisper](https://huggingface.co/openai/whisper-large-v3) - Speech recognition
