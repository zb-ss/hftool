# hftool

A powerful CLI for running Hugging Face models: text-to-image, text-to-video, text-to-speech, speech-to-text, and more.

## Features

- **Text-to-Image**: Z-Image-Turbo, Stable Diffusion XL, FLUX
- **Text-to-Video**: HunyuanVideo-1.5, CogVideoX, Wan2.2
- **Text-to-Speech**: VibeVoice, Bark, MMS-TTS, GLM-TTS
- **Speech-to-Text**: Whisper (with timestamps and SRT export)
- **Plus**: Text generation, classification, translation, and more via transformers pipelines

**Optimized for AMD ROCm** (also supports NVIDIA CUDA, Apple MPS, and CPU).

## Installation

### Quick Install

```bash
pip install hftool
```

### Install with Specific Features

```bash
# Text-to-Image (Z-Image, SDXL, FLUX)
pip install "hftool[with_t2i]"

# Text-to-Video (HunyuanVideo, CogVideoX, Wan2.2)
pip install "hftool[with_t2v]"

# Text-to-Speech (VibeVoice, Bark, MMS-TTS)
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
git clone https://github.com/zashboy-websites/hftool
cd hftool
pip install -e ".[dev]"  # Includes pytest
```

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
| `t2v` | text-to-video |
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

Generate speech with VibeVoice:

```bash
# Basic usage
hftool -t tts -i "Hello, this is a test of the text to speech system." -o hello.wav

# With specific model
hftool -t tts -m microsoft/VibeVoice-Realtime-0.5B \
       -i "Welcome to hftool, your command-line AI assistant." \
       -o welcome.wav

# Output as MP3 (requires ffmpeg)
hftool -t tts -i "This will be saved as MP3." -o output.mp3
```

**Other supported models:**
- `suno/bark-small` (multi-language, sound effects)
- `facebook/mms-tts-eng` (lightweight)

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

```
Usage: hftool [OPTIONS]

Options:
  -t, --task TEXT         Task to perform (required unless --list-tasks)
  -m, --model TEXT        Model name/path (uses task default if omitted)
  -i, --input TEXT        Input data: text, file path, or URL (required)
  -o, --output-file TEXT  Output file path (auto-generated if omitted)
  -d, --device TEXT       Device: auto, cuda, mps, cpu (default: auto)
  --dtype TEXT            Data type: bfloat16, float16, float32
  --list-tasks            List all available tasks and aliases
  -v, --verbose           Show detailed progress
  --help                  Show this message and exit
```

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
| Text-to-Video | HunyuanVideo 480p | ~20-24 GB | Use CPU offload |
| Text-to-Video | HunyuanVideo 720p | ~30-40 GB | Requires multi-GPU |
| Text-to-Speech | VibeVoice | ~2-4 GB | Easy |
| Speech-to-Text | Whisper-large-v3 | ~4-6 GB | Easy |

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
├── cli.py              # CLI entry point
├── core/
│   ├── device.py       # ROCm/CUDA/MPS/CPU detection
│   └── registry.py     # Task registry and configuration
├── tasks/
│   ├── base.py         # Abstract base task class
│   ├── text_to_image.py
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

- [GitHub Repository](https://github.com/zashboy-websites/hftool)
- [Hugging Face](https://huggingface.co)

### Model References

- [Z-Image](https://github.com/Tongyi-MAI/Z-Image) - State-of-the-art text-to-image
- [HunyuanVideo-1.5](https://huggingface.co/tencent/HunyuanVideo-1.5) - High-quality video generation
- [VibeVoice](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) - Real-time TTS
- [Whisper](https://huggingface.co/openai/whisper-large-v3) - Speech recognition
