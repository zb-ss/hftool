# Features & UX

> [Back to README](../README.md)

## Table of Contents
- [Terminal UI](#terminal-ui-primary-interface)
- [File Picker](#file-picker)
- [Interactive JSON Builder](#interactive-json-builder)
- [Command History](#command-history)
- [Dry-Run Mode](#dry-run-mode)
- [Shell Completions](#shell-completions)
- [System Diagnostics](#system-diagnostics)

---

## Terminal UI (Primary Interface)

```bash
hftool tui
```

The TUI runs in Docker by default and is the main interactive interface. Model
tables show recommended/legacy status, local asset state, intended use, minimum
VRAM, and license. The model detail panel shows complete download size, exact
inference defaults, dtype, and commercial-use terms. System diagnostics show
live VRAM and physical-to-visible GPU mapping, and generation runs a no-download
headroom preflight before loading a pipeline.

The task, model-management, settings, and voiceover screens share the same
container runtime and catalog/executor paths. Use `hftool tui --native` only for
development or a deliberately managed native environment.

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
