# Configuration Guide

hftool supports flexible configuration through multiple sources, allowing you to set defaults and override them as needed.

## Configuration Priority

Configuration is resolved in this order (highest to lowest priority):

1. **CLI arguments** - Command line flags always win
2. **Environment variables** - `HFTOOL_*` variables
3. **Project config** - `./.hftool/config.toml` in current directory
4. **User config** - `~/.hftool/config.toml` in home directory
5. **Built-in defaults** - Sensible defaults for all options

## Config File Locations

| Location | Purpose |
|----------|---------|
| `~/.hftool/config.toml` | User-level defaults (applies everywhere) |
| `./.hftool/config.toml` | Project-specific overrides |
| `~/.hftool/models/` | Default model cache directory |
| `~/.hftool/history.json` | Command history for interactive mode |
| `~/.hftool/benchmarks.json` | Performance benchmark results |

## Config File Format (TOML)

```toml
# ~/.hftool/config.toml

# Default settings for all tasks
[defaults]
# Default output directory for generated files
output_dir = "~/AI/outputs"

# Default device selection: auto, cuda, cpu, mps
device = "auto"

# Auto-download models without prompting
auto_download = false

# Open generated files automatically
auto_open = false

# Default inference settings
steps = 30
guidance = 7.5
seed = -1  # -1 = random

# Task-specific overrides
[text-to-image]
model = "zimage"
width = 1024
height = 1024
steps = 25

[text-to-video]
model = "ltx2"
fps = 24
duration = 5.0

[text-to-speech]
model = "bark"
voice = "v2/en_speaker_6"

[automatic-speech-recognition]
model = "whisper-large-v3"
language = "auto"

# Model aliases (shortcuts)
[aliases]
sdxl = "stabilityai/stable-diffusion-xl-base-1.0"
flux = "black-forest-labs/FLUX.1-schnell"
```

## Per-Model Configuration

You can configure settings for specific models:

```toml
# Model-specific settings
[models."stabilityai/stable-diffusion-xl-base-1.0"]
steps = 30
guidance = 7.5
scheduler = "euler"

[models."Lightricks/LTX-Video"]
steps = 50
guidance = 3.5
fps = 24

[models."suno/bark"]
voice = "v2/en_speaker_6"
```

## Environment Variables

All settings can also be set via environment variables. See [environment.md](environment.md) for the complete list.

Common pattern:
- Config key `auto_download` → Environment variable `HFTOOL_AUTO_DOWNLOAD`
- Config key `models_dir` → Environment variable `HFTOOL_MODELS_DIR`

## Example Configurations

### Minimal Setup (AMD GPU)

```toml
# ~/.hftool/config.toml
[defaults]
auto_download = true
output_dir = "~/AI"
```

### Multi-GPU Workstation

```toml
# ~/.hftool/config.toml
[defaults]
auto_download = true
multi_gpu = true

[text-to-video]
# Large video models benefit from multi-GPU
model = "hunyuan"
```

### Low-VRAM Setup (8GB GPU)

```toml
# ~/.hftool/config.toml
[defaults]
cpu_offload = 1

[text-to-image]
model = "sdxl"  # Works well on 8GB
steps = 20
```

### CI/CD Pipeline

```bash
# Environment variables for headless operation
export HFTOOL_AUTO_DOWNLOAD=1
export HFTOOL_QUIET=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HFTOOL_AUTO_OPEN=0
```

## Validating Configuration

Check your current configuration:

```bash
# Show loaded config files
hftool doctor

# Test with dry-run
hftool -t t2i -i "test" -o test.png --dry-run
```

## Creating Config Directory

```bash
# Create config directory structure
mkdir -p ~/.hftool

# Copy example config
cat > ~/.hftool/config.toml << 'EOF'
[defaults]
auto_download = false
output_dir = "~/AI"

[text-to-image]
model = "zimage"
steps = 25
EOF
```
