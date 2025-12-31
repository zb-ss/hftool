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
