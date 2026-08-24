# CLI Reference

> [Back to README](../README.md)

## Table of Contents
- [Main Command](#main-command)
- [Task Aliases](#task-aliases)
- [Environment Variables](#environment-variables)
- [Passing Model-Specific Arguments](#passing-model-specific-arguments)
- [Configuration File](#configuration-file)

---

## Main Command

```
Usage: hftool [OPTIONS] COMMAND [ARGS]...

Options:
  -t, --task TEXT         Task to perform
  -m, --model TEXT        Model name/path (uses task default if omitted)
  -i, --input TEXT        Input data: text, file path, @ reference, @? for interactive
  -o, --output-file TEXT  Output file path (auto-generated if omitted)
  -d, --device TEXT       Device: auto, cuda, mps, cpu (default: auto)
  -g, --gpu TEXT          GPU selection: auto, all, 0, 1, 0,1 (multi-GPU)
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
  docker    Manage Docker-based execution (setup, status, build, run)
  models    List available models for tasks
  download  Download models from HuggingFace Hub
  status    Show download status and disk usage
  clean     Delete downloaded models
  history   View and manage command history (--rerun, --clear)
  run       Run a task (alternative to -t flag)
```

### Basic Syntax

```bash
hftool -t <task> -i <input> [-m <model>] [-o <output>] [-- extra_args]
```

---

## Task Aliases

| Alias | Full Name |
|-------|-----------|
| `t2i` | text-to-image |
| `i2i`, `img2img` | image-to-image |
| `t2v` | text-to-video |
| `i2v` | image-to-video |
| `vo` | voiceover |
| `tts` | text-to-speech |
| `asr`, `stt` | automatic-speech-recognition |
| `llm` | text-generation |

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HFTOOL_MODELS_DIR` | Custom models storage directory | `~/.hftool/models/` |
| `HFTOOL_AUTO_DOWNLOAD` | Auto-download models without prompting | `0` (disabled) |
| `HFTOOL_AUTO_OPEN` | Auto-open output files | `auto` (media files only) |
| `HFTOOL_GPU` | GPU selection: `auto`, `all`, `0`, `1`, `0,1` | (none) |
| `HFTOOL_MULTI_GPU` | Multi-GPU mode: `1`/`balanced` enables; unset stays single-GPU | disabled |
| `HFTOOL_CPU_OFFLOAD` | CPU offload: `1` model offload, `2` sequential offload | disabled |
| `HFTOOL_ATTENTION_BACKEND` | Optional Diffusers attention backend; native SDPA is default | (none) |
| `HFTOOL_TORCH_COMPILE` | Opt in to `torch.compile`; cold/warm timing differs | `0` |
| `HFTOOL_CPU_OFFLOAD` | CPU offload level: `0` disabled, `1` model, `2` sequential | (none) |
| `HFTOOL_ROCM_PATH` | Path to ROCm libraries (e.g., Ollama's bundled ROCm) | (none) |
| `HSA_OVERRIDE_GFX_VERSION` | AMD GPU architecture override (e.g., `11.0.0` for RX 7900) | (none) |
| `HF_TOKEN` | HuggingFace token for gated models | (none) |

---

## Passing Model-Specific Arguments

Use `--` to pass additional arguments to the underlying model:

```bash
hftool -t t2i -i "A cat" -o cat.png \
       -- --num_inference_steps 20 --guidance_scale 7.5 --seed 42
```

---

## Configuration File

hftool supports persistent configuration via TOML files for convenience.

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
model = "kokoro"           # Fast lightweight default (was bark-small)
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

### Example Usage

```bash
# With config file setting device=cuda and model=z-image-turbo
hftool -t t2i -i "A cat in space" -o cat.png
# Uses cuda device and z-image-turbo from config

# Override config with CLI args
hftool -t t2i -i "A cat" -o cat.png --device cpu -m sdxl
# Uses cpu device and sdxl model (CLI overrides config)
```
