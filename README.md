# hftool

A CLI for running HuggingFace models, optimized for AMD ROCm.

> **What this is:** A convenient wrapper for common AI tasks (image/video/speech generation, transcription). Not a replacement for transformers or diffusers, but a simpler interface when you just want to run a model without writing Python.
>
> **Who it's for:** AMD GPU owners frustrated with CUDA-first tooling, and anyone who wants a unified CLI for multiple AI modalities.

## Demo

[![Watch hftool interactive mode demo](https://img.youtube.com/vi/oYANVmglEhs/maxresdefault.jpg)](https://www.youtube.com/watch?v=oYANVmglEhs)

## Features

### AI Tasks
- **Text-to-Image**: Z-Image-Turbo, Stable Diffusion XL, FLUX
- **Image-to-Image**: Qwen Image Edit (advanced editing with multi-image support), SDXL Refiner
- **Text-to-Video**: HunyuanVideo-1.5, CogVideoX, Wan2.2
- **Text-to-Speech**: Bark, MMS-TTS, GLM-TTS
- **Speech-to-Text**: Whisper (with timestamps and SRT export)
- **Plus**: Text generation, classification, translation, and more via transformers pipelines

### User Experience
- **Interactive Wizard** (`-I`): Full guided experience - select task, model, input, output, and all options
- **File Picker** (`@` syntax): Interactive file selection with multiple modes (@, @?, @., @~, @*.ext, @@)
- **Interactive Input**: Guided JSON builder for complex inputs (image-to-image, etc.)
- **History Tracking**: View and re-run previous commands with `hftool history`
- **Dry-Run Mode**: Preview operations without executing (--dry-run)
- **Configuration Files**: Save preferences in TOML config files
- **Shell Completions**: Tab completion for bash, zsh, and fish
- **Better Error Messages**: Actionable suggestions when things go wrong
- **Progress Bars**: Visual feedback during model loading and generation

### Management
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

### Optional Dependencies

For enhanced user experience features:

```bash
# Interactive file picker and JSON builder
pip install InquirerPy

# Or for pipx:
pipx runpip hftool install InquirerPy
```

**Note:** Without InquirerPy, the `@` file picker and `--interactive` mode will not work, but all other features remain functional.

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
# Full interactive wizard - guided experience for beginners
hftool -I

# Or specify everything on command line
hftool -t t2i -i "A cat in space" -o cat.png

# Interactive file selection
hftool -t asr -i @ -o transcript.txt

# Preview before running
hftool -t t2i -i "A cat" --dry-run

# Reproducible generation with seed
hftool -t t2i -i "A cat" -o cat.png --seed 42

# Re-run previous command
hftool history --rerun 5

# Install shell completions for tab completion
hftool completion --install
```

**New features**:
- **Auto-open**: Generated images, audio, and video files automatically open when complete!
- **File picker**: Use `@` to interactively select input files
- **History**: View and re-run previous commands with `hftool history`
- **Dry-run**: Preview operations without executing with `--dry-run`
- **Config files**: Save preferences in `~/.hftool/config.toml`

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

## Configuration File

hftool supports persistent configuration via TOML files for convenience.

### Creating a Config File

```bash
# Create default config with helpful comments
hftool config init

# Or manually create ~/.hftool/config.toml
```

### Config File Structure

```toml
# ~/.hftool/config.toml

[defaults]
device = "cuda"          # Device to use: auto, cuda, mps, cpu
dtype = "bfloat16"       # Data type: bfloat16, float16, float32
auto_open = true         # Auto-open output files
verbose = false          # Verbose output

[text-to-image]
model = "z-image-turbo"  # Default model for this task
num_inference_steps = 9
guidance_scale = 0.0
width = 1024
height = 1024

[text-to-speech]
model = "bark-small"
sample_rate = 24000

[automatic-speech-recognition]
model = "whisper-large-v3"
return_timestamps = true

[aliases]
# Custom model aliases for convenience
fast-image = "Tongyi-MAI/Z-Image-Turbo"
quality-image = "black-forest-labs/FLUX.1-dev"
my-whisper = "openai/whisper-large-v3"

[paths]
models_dir = "~/.hftool/models"
output_dir = "~/ai-outputs"
history_file = "~/.hftool/history.json"
```

### Config Priority

Settings are applied in this order (highest to lowest):
1. **CLI arguments** - `hftool -t t2i --device cuda`
2. **Environment variables** - `HFTOOL_DEVICE=cuda`
3. **Project config** - `./.hftool/config.toml` (current directory)
4. **User config** - `~/.hftool/config.toml` (home directory)
5. **Built-in defaults**

### Config Commands

```bash
# View current configuration
hftool config show

# Create default config file
hftool config init

# Edit config in your $EDITOR
hftool config edit
```

### Example Usage

```bash
# With config file setting device=cuda and model=z-image-turbo
hftool -t t2i -i "A cat in space" -o cat.png
# Uses cuda device and z-image-turbo from config

# Override config with CLI args
hftool -t t2i -i "A cat" -o cat.png --device cpu -m sdxl
# Uses cpu device and sdxl model (CLI overrides config)
```

---

## File Picker

hftool includes a powerful file picker that makes it easy to select input files without typing full paths.

### @ Syntax

Use `@` in the `-i` / `--input` parameter to trigger the file picker:

| Syntax | Description | Example |
|--------|-------------|---------|
| `@` | Interactive file picker (current directory) | `hftool -t asr -i @ -o transcript.txt` |
| `@?` | Interactive with fuzzy search (shows all files) | `hftool -t t2i -i @? -o output.png` |
| `@.` | Pick from current directory | `hftool -t asr -i @. -o transcript.txt` |
| `@~` | Pick from home directory | `hftool -t t2i -i @~ -o output.png` |
| `@/path/` | Pick from specific directory | `hftool -t asr -i @/recordings/ -o transcript.txt` |
| `@*.ext` | Files matching glob pattern | `hftool -t asr -i @*.wav -o transcript.txt` |
| `@@` | Recent files from history | `hftool -t t2i -i @@ -o output.png` |

### Interactive Mode

When `@?` is used or no matching files are found, hftool enters interactive mode:

```
? Select a file: 
  recording1.wav
  recording2.wav
> recording3.wav
  music.mp3
  podcast.wav
```

Use arrow keys to select, Enter to confirm, Ctrl+C to cancel.

### Examples

```bash
# Pick a WAV file interactively
hftool -t asr -i @ -o transcript.txt

# Select from all files with fuzzy search
hftool -t t2i -i @? -o output.png

# Pick from a specific directory
hftool -t asr -i @/home/user/recordings/ -o transcript.txt

# Use glob pattern to filter
hftool -t asr -i @*.wav -o transcript.txt

# Recent files from history
hftool -t t2i -i @@ -o output.png
```

**Note:** The file picker requires the optional `InquirerPy` dependency:
```bash
pip install InquirerPy
# Or for pipx:
pipx runpip hftool install InquirerPy
```

---

## Interactive JSON Builder

For tasks that require complex JSON input (like image-to-image), use `--interactive` or `-i @?` to launch an interactive builder:

```bash
# Interactive mode for image-to-image
hftool -t i2i --interactive -o output.png

# Or trigger with @?
hftool -t i2i -i @? -o output.png
```

The interactive builder guides you through entering parameters:

```
? image: photo.jpg
? prompt: turn this into a watercolor painting
? seed (optional): 42
? true_cfg_scale (optional): 4.0
? num_inference_steps (optional): 50
```

Supports:
- Image file selection with file picker
- Multi-image inputs (enter comma-separated paths)
- Optional parameter skipping (press Enter to use defaults)
- Parameter validation and type conversion

---

## Command History

hftool tracks all commands you run and allows you to view and re-run them:

### View History

```bash
# Show recent commands
hftool history

# Show last 20 commands
hftool history -n 20

# Output as JSON
hftool history --json
```

**Example output:**
```
Recent command history:
================================================================================

[5] ✓ 2024-01-15 14:32:15 - text-to-image
    Model: z-image-turbo
    Input: A cat in space
    Output: cat.png
    Seed: 42
    Rerun: hftool history --rerun 5

[4] ✗ 2024-01-15 14:28:10 - automatic-speech-recognition
    Model: whisper-large-v3
    Input: recording.wav
    Output: transcript.txt
    Error: Model not downloaded
    Rerun: hftool history --rerun 4
```

### Re-run Commands

```bash
# Re-run command #5
hftool history --rerun 5

# With confirmation prompt
hftool history --rerun 5
# Shows: Re-running command #5 from 2024-01-15 14:32:15:
#   hftool -t text-to-image -i "A cat in space" -o cat.png --seed 42
# Continue? [Y/n]:
```

### Clear History

```bash
# Clear all history
hftool history --clear
```

### History Storage

History is stored in `~/.hftool/history.json` by default. Customize with:

```toml
# ~/.hftool/config.toml
[paths]
history_file = "~/custom/path/history.json"
```

Or via environment variable:
```bash
export HFTOOL_HISTORY_FILE=~/custom/path/history.json
```

---

## Dry-Run Mode

Preview operations without executing them. Useful for:
- Checking model requirements before downloading
- Estimating VRAM usage
- Validating parameters

```bash
# Preview text-to-image generation
hftool -t t2i -i "A cat in space" -o cat.png --dry-run
```

**Example output:**
```
============================================================
Dry-Run Mode: text-to-image
============================================================

Task:     text-to-image
Model:    Z-Image Turbo (Tongyi-MAI/Z-Image-Turbo)
Size:     ~6.0 GB
Device:   cuda
Dtype:    bfloat16
VRAM:     ~10-12 GB estimated

Input:    "A cat in space"
Output:   cat.png

Parameters:
  num_inference_steps: 9
  guidance_scale: 0.0
  width: 1024
  height: 1024
  seed: 42

Dependencies:
  ✓ torch
  ✓ diffusers
  ✓ transformers

Status:   Model downloaded

Would run: hftool -t text-to-image -i "A cat in space" -o cat.png --seed 42
```

Use dry-run to:
- **Verify dependencies** before attempting generation
- **Check disk space** requirements
- **Estimate VRAM** usage for your GPU
- **Preview parameters** from config file

---

## Shell Completions

Enable tab completion for faster CLI usage:

```bash
# Auto-detect shell and install
hftool completion --install

# Show completion script for bash
hftool completion bash

# Install for specific shell
hftool completion zsh --install
```

After installation, restart your shell or run:
- bash: `source ~/.bashrc`
- zsh: `source ~/.zshrc`
- fish: Completions load automatically

**Completions include**:
- Task names and aliases (t2i, text-to-image, etc.)
- Model names (z-image-turbo, whisper-large-v3, etc.)
- Device options (auto, cuda, mps, cpu)
- File picker syntax (@, @?, @~, etc.)

---

## System Diagnostics

Check your system setup and troubleshoot issues:

```bash
# Run all diagnostic checks
hftool doctor

# Output as JSON
hftool doctor --json
```

**Checks performed**:
- Python version (requires 3.10+)
- PyTorch installation and GPU detection
- ffmpeg availability (for video/audio tasks)
- Network connectivity to HuggingFace Hub
- Optional feature dependencies
- Configuration file status

Exit codes: 0=OK, 1=warnings, 2=errors

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

# Resume interrupted download (default)
hftool download -t t2i
# Disable resume
hftool download -t t2i --no-resume
```

**Note**: Downloads automatically resume if interrupted. Use `hftool status` to see partial downloads.

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
  -i, --input TEXT        Input data: text, file path, @ reference, @? for interactive
  -o, --output-file TEXT  Output file path (auto-generated if omitted)
  -d, --device TEXT       Device: auto, cuda, mps, cpu (default: auto)
  --dtype TEXT            Data type: bfloat16, float16, float32
  --seed INTEGER          Random seed for reproducible generation
  --interactive           Interactive mode for complex inputs (JSON builder)
  --dry-run               Preview operation without executing
  --open / --no-open      Open output with default app (auto for media files)
  --list-tasks            List all available tasks and aliases
  -v, --verbose           Show detailed progress
  --help                  Show this message and exit

Commands:
  setup     Run interactive PyTorch setup wizard
  config    View and manage configuration (show, init, edit)
  models    List available models for tasks
  download  Download models from HuggingFace Hub
  status    Show download status and disk usage
  clean     Delete downloaded models
  history   View and manage command history (--rerun, --clear)
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
- [Report a Bug](https://github.com/zb-ss/hftool/issues/new?template=bug_report.md)
- [Request a Model](https://github.com/zb-ss/hftool/issues/new?template=model_request.md)
- [Hugging Face](https://huggingface.co)

### Model References

- [Z-Image](https://github.com/Tongyi-MAI/Z-Image) - State-of-the-art text-to-image
- [Qwen Image Edit](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) - Advanced image editing with character consistency
- [HunyuanVideo-1.5](https://huggingface.co/tencent/HunyuanVideo-1.5) - High-quality video generation
- [Bark](https://huggingface.co/suno/bark) - High-quality TTS with sound effects
- [Whisper](https://huggingface.co/openai/whisper-large-v3) - Speech recognition
