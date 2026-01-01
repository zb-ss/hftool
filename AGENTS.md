# AGENTS.md

## Build & Run Commands
- Install: `pip install -e ".[all]"` or `pip install -e ".[dev]"` for testing
- Run CLI: `hftool -t <task> -i <input> [-m <model>] [-o <output>]`
- List tasks: `hftool --list-tasks`
- List models: `hftool models` or `hftool models -t <task>`
- Download model: `hftool download -t <task>` or `hftool download -m <model>`
- View config: `hftool config show`
- View history: `hftool history` or `hftool history --rerun <id>`
- Preview operations: `hftool -t <task> -i <input> --dry-run`
- System diagnostics: `hftool doctor`
- Shell completions: `hftool completion --install`
- Run tests: `pytest tests/` or `pytest tests/test_core.py::TestTaskRegistry::test_name`

## New Features (v0.3.0+)

### Phase 1 Features

#### Configuration File
- Config file: `~/.hftool/config.toml` or `./.hftool/config.toml`
- View config: `hftool config show`
- Create config: `hftool config init`
- Edit config: `hftool config edit`
- Priority: CLI args > env vars > project config > user config > defaults
- Supports task-specific defaults, model aliases, custom paths

#### Dry-Run Mode
Preview operations without executing:
```bash
hftool -t t2i -i "prompt" -o output.png --dry-run
```
Shows: model info, device, VRAM estimate, parameters, dependencies

#### Better Error Messages
Errors now include actionable suggestions:
- CUDA OOM → suggests `--dtype float16`, smaller model
- Missing deps → suggests `pip install` commands
- Network errors → suggests `--offline` mode
- File not found → suggests using file picker with `@`

#### Progress Bars
Visual feedback for:
- Model downloading
- Model loading
- Generation progress

### Phase 2 Features

#### History Tracking
```bash
hftool history                 # View recent commands
hftool history -n 20           # Show last 20 commands
hftool history --rerun 5       # Re-run command #5
hftool history --clear         # Clear all history
```
- Stored in `~/.hftool/history.json`
- Tracks success/failure, timestamps, parameters
- Re-run with confirmation prompt

#### File Picker (@ syntax)
```bash
hftool -t asr -i @              # Interactive file picker
hftool -t asr -i @?             # Fuzzy search mode
hftool -t asr -i @.             # Pick from current directory
hftool -t asr -i @~             # Pick from home directory
hftool -t asr -i @/path/        # Pick from specific directory
hftool -t asr -i @*.wav         # Glob pattern matching
hftool -t asr -i @@             # Recent files from history
```
- Requires `InquirerPy` (optional dependency)
- Interactive selection with arrow keys
- Supports glob patterns and directory navigation

#### Interactive JSON Builder
```bash
hftool -t i2i --interactive     # Guided JSON builder
hftool -t i2i -i @?             # Trigger interactive mode
```
- Guided parameter entry for complex inputs
- File picker integration for image paths
- Type validation and conversion
- Optional parameter skipping

#### Parameter Schemas
- Structured input validation for tasks
- Type hints and default values
- Auto-completion in interactive mode
- Documentation strings for parameters

#### Seed Control
```bash
hftool -t t2i -i "A cat" --seed 42
```
- Top-level `--seed` option for reproducibility
- Automatically applied to model generation
- Shown in history and dry-run output

### Phase 3 Features (Power User)

#### Quiet/JSON Output Modes
```bash
hftool -t t2i -i "A cat" -o cat.png --quiet   # Only output file path
hftool -t t2i -i "A cat" -o cat.png --json    # JSON output
```
- `--quiet` / `-q`: Minimal output for scripting
- `--json`: Structured JSON output for automation
- Exit codes: 0 = success, 1 = failure
- Works with all tasks and commands

#### Model Information
```bash
hftool info whisper-large-v3
hftool info openai/whisper-large-v3 --json
```
- Show detailed model information
- Display recommended settings from metadata
- VRAM estimates for different resolutions
- Download status and local path
- HuggingFace URL and dependencies

#### Metadata Embedding
```bash
hftool -t t2i -i "A cat" -o cat.png              # Default: enabled
hftool -t t2i -i "A cat" -o cat.png --no-embed-metadata
```
- Embeds generation parameters in output files
- **PNG**: PIL tEXt chunks (lossless)
- **JPEG**: EXIF UserComment (requires `piexif`)
- **Audio/Video**: Sidecar .json files
- Stores: task, model, prompt, seed, steps, guidance, timestamp
- Readable with exiftool and standard tools

#### Benchmarking
```bash
hftool benchmark -t text-to-image -m z-image-turbo
hftool benchmark --all                    # All downloaded models
hftool benchmark --all --skip-large       # Skip models >15GB
```
- Measure load time, inference time, VRAM usage
- Uses standardized test prompts
- Results cached in `~/.hftool/benchmarks.json`
- Supports `--json` output
- Keeps last 100 results per model

#### Batch Processing
```bash
hftool -t asr --batch ./audio_files/ --batch-output-dir ./transcripts/
hftool -t t2i --batch inputs.txt --batch-output-dir ./outputs/
hftool -t t2i --batch-json batch.json
```
- Process multiple inputs from file or directory
- `--batch <source>`: File list or directory
- `--batch-json <file>`: JSON array with per-entry params
- `--batch-output-dir`: Output directory
- Auto-generates numbered filenames
- Shows progress, continues on error
- Summary at end (success/failure counts)

**Batch JSON format**:
```json
[
  {"input": "A cat", "output": "cat.png", "params": {"seed": 42}},
  {"input": "A dog", "output": "dog.png", "params": {"seed": 123}}
]
```

### Phase 4 Features (Quality of Life)

#### Shell Completions
```bash
hftool completion --install           # Auto-detect and install
hftool completion bash                # Show bash completion script
hftool completion zsh                 # Show zsh completion script
hftool completion fish                # Show fish completion script
hftool completion bash --install      # Install for bash
```
- Tab completion for tasks, models, devices, dtypes
- File picker syntax completion (@, @?, @~, etc.)
- Auto-detection and installation
- Works with bash, zsh, and fish shells
- Restart shell after installation

#### Doctor Command
```bash
hftool doctor                         # Run system diagnostics
hftool doctor --json                  # JSON output for automation
```
- Checks Python version (requires 3.10+)
- Checks PyTorch installation and GPU
- Checks ffmpeg availability
- Checks network connectivity to HuggingFace
- Shows installed optional features
- Shows configuration status
- Provides actionable suggestions
- Exit codes: 0=OK, 1=warnings, 2=errors

#### Resume Downloads
```bash
hftool download -t t2i --resume       # Resume interrupted download (default)
hftool download -t t2i --no-resume    # Force fresh download
hftool status                         # Shows resumable downloads
```
- Automatic resume for interrupted downloads
- Detection of partial downloads
- Status command shows resumable downloads
- Enabled by default for reliability

#### Full Interactive Wizard
```bash
hftool -I                             # Full interactive wizard
hftool --interactive-wizard           # Same as above
```
- Complete guided experience for all tasks
- Step-by-step: task → model → input → output → options
- Shows model download status and sizes
- Supports text prompts, file selection, JSON builder
- Device auto-detection with GPU info
- Optional advanced parameters
- Can be set as default via config or env:
  ```toml
  # ~/.hftool/config.toml
  [defaults]
  interactive = true
  ```
  Or: `HFTOOL_INTERACTIVE=1`

## Supported Tasks
- **text-to-image** (t2i): Z-Image, SDXL, FLUX - requires `[with_t2i]`
- **image-to-image** (i2i): Qwen Image Edit, SDXL Refiner - requires `[with_t2i]`
- **text-to-video** (t2v): HunyuanVideo, CogVideoX, Wan2.2 - requires `[with_t2v]` + system ffmpeg
- **text-to-speech** (tts): Bark, MMS-TTS - requires `[with_tts]`
- **automatic-speech-recognition** (asr/stt): Whisper - requires `[with_stt]`

## Model Management
- Models stored in `~/.hftool/models/` (configurable via `HFTOOL_MODELS_DIR`)
- Interactive download prompt on first run (can be disabled with `HFTOOL_AUTO_DOWNLOAD=1`)
- Use `hftool status` to see downloaded models and disk usage
- Use `hftool clean` to remove downloaded models

## Code Style
- **Python >=3.10** with type hints (`Optional`, `List`, `Dict` from `typing`)
- **Naming**: `snake_case` for variables/functions; `PascalCase` for classes
- **Imports**: stdlib -> third-party -> local; use try/except for optional deps
- **Optional deps**: Use `hftool.utils.deps.is_available()` / `check_dependency()`
- **Error handling**: `click.echo(..., err=True)` for errors, `sys.exit(1)` on failure
- **Formatting**: ~120 char line limit, 4-space indentation

## Project Structure
```
hftool/
  cli.py              # CLI entry point with subcommands (models, download, status, clean, config, history)
  core/
    device.py         # ROCm/CUDA/MPS/CPU detection
    registry.py       # Task registry and configuration
    models.py         # Model registry with download metadata (ModelInfo, MODEL_REGISTRY)
    download.py       # Model download manager (huggingface_hub integration)
    config.py         # Configuration file manager (TOML support)
    history.py        # Command history tracking and re-run
    parameters.py     # Parameter schemas for tasks
  tasks/              # Task handlers (text_to_image, text_to_video, etc.)
  io/
    input_loader.py   # Input handling
    output_handler.py # Output handling (ffmpeg)
    file_picker.py    # @ file reference resolver
    interactive_input.py  # Interactive JSON builder
  utils/
    deps.py           # Dependency checking
    errors.py         # Custom error classes
    progress.py       # Progress bar utilities
tests/                # pytest tests (test_core.py, test_io.py, test_utils.py, test_config.py, test_history.py, etc.)
```

## Key APIs
- `hftool.core.models.get_models_for_task(task)` - Get available models for a task
- `hftool.core.models.get_default_model_info(task)` - Get default model info
- `hftool.core.download.ensure_model_available(repo_id, size_gb, task, name)` - Ensure model is downloaded
- `hftool.core.download.get_models_dir()` - Get models storage directory
