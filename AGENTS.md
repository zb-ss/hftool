# AGENTS.md

## Build & Run Commands
- Install: `pip install -e ".[all]"` or `pip install -e ".[dev]"` for testing
- Run CLI: `hftool -t <task> -i <input> [-m <model>] [-o <output>]`
- List tasks: `hftool --list-tasks`
- List models: `hftool models` or `hftool models -t <task>`
- Download model: `hftool download -t <task>` or `hftool download -m <model>`
- Run tests: `pytest tests/` or `pytest tests/test_core.py::TestTaskRegistry::test_name`

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
  cli.py              # CLI entry point with subcommands (models, download, status, clean)
  core/
    device.py         # ROCm/CUDA/MPS/CPU detection
    registry.py       # Task registry and configuration
    models.py         # Model registry with download metadata (ModelInfo, MODEL_REGISTRY)
    download.py       # Model download manager (huggingface_hub integration)
  tasks/              # Task handlers (text_to_image, text_to_video, etc.)
  io/                 # Input/output handling (loaders, savers)
  utils/              # Dependency checking, helpers
tests/                # pytest tests (test_core.py, test_io.py, test_utils.py)
```

## Key APIs
- `hftool.core.models.get_models_for_task(task)` - Get available models for a task
- `hftool.core.models.get_default_model_info(task)` - Get default model info
- `hftool.core.download.ensure_model_available(repo_id, size_gb, task, name)` - Ensure model is downloaded
- `hftool.core.download.get_models_dir()` - Get models storage directory
