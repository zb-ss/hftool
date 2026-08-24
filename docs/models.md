# Model Management

> [Back to README](../README.md)

Model definitions come from the versioned catalog packaged with hftool. The TUI
and CLI read the same repository, revision/profile, pipeline class, dtype,
defaults, license, status, and VRAM fields; there is no separate frontend model
list to drift out of sync.

Use `hftool tui` for the primary model-browsing experience. It separates catalog
status (`default`, `recommended`, `legacy`) from local download status and shows
the intended use, license, and runtime requirements before generation.

## List Available Models

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

# Detailed base/profile assets and authoritative defaults
hftool info qwen-image-2512-lightning

# No download, dependency installation, or pipeline load
hftool -t t2i -m qwen-image-2512-lightning -i "preview" --dry-run --json
```

## Download Models

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

Adapter-backed profiles are complete only when both the base model and exact
declared adapter file are present. `hftool download -t t2i -m
qwen-image-2512-lightning` downloads both assets; it does not fetch every weight
file in the adapter repository.

## Reproducible image benchmark gate

```bash
hftool benchmark -t t2i -m flux2-klein-4b --json
```

The image gate uses five named cases with fixed seeds and aspect ratios. It
records exact catalog/profile settings, cold load time, first-generation time,
the median of four warm generations, peak allocated VRAM, live free VRAM before
and after, and whether CPU offload was active. Benchmarking performs real model
inference and therefore requires the model and a suitable runtime; dry-run is
the safe inspection command.

## Check Status

```bash
# Show downloaded models and disk usage
hftool status
```

## Clean Up

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

## Custom Storage Location

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

## Gated Models (Authentication Required)

Some models like FLUX.2-klein-9B require accepting a license agreement and HuggingFace authentication:

```bash
# Option 1: Login with huggingface-cli (recommended)
pip install huggingface_hub
huggingface-cli login
# Follow prompts to enter your token

# Option 2: Set environment variable
export HF_TOKEN=your_token_here

# Option 3: Add to .env file
echo "HF_TOKEN=your_token_here" >> ~/.hftool/.env
```

**Steps for gated models:**
1. Visit the model page (e.g., https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
2. Accept the license agreement
3. Create an access token at https://huggingface.co/settings/tokens
4. Login with `huggingface-cli login` or set `HF_TOKEN`

hftool will automatically detect your token and show a warning if authentication is missing for gated models.

**Important: Token permissions for gated repos**

If you get errors like "cannot find the requested files" or "check your internet connection" when downloading gated models, your token may lack the required permissions.

When creating your token at https://huggingface.co/settings/tokens:
- **Recommended**: Use a **"Read"** token (classic type) - works with all repos
- **Fine-grained tokens**: Must have "Access to public gated repos" enabled

To check/fix your token:
1. Go to https://huggingface.co/settings/tokens
2. Click on your token to view its permissions
3. Ensure it has access to gated repositories

## Debug Mode and Logging

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

## Auto-Download Mode

To skip interactive prompts and auto-download models:

```bash
export HFTOOL_AUTO_DOWNLOAD=1
```

## Auto-Open Output Files

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
