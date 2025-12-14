# AGENTS.md

## Build & Run Commands
- Install: `pip install -e ".[all]"` or `pip install -e ".[dev]"` for testing
- Run CLI: `hftool -t <task> -i <input> [-m <model>] [-o <output>]`
- List tasks: `hftool --list-tasks`
- Run tests: `pytest tests/` or `pytest tests/test_core.py::TestTaskRegistry::test_name`

## Supported Tasks
- **text-to-image** (t2i): Z-Image, SDXL, FLUX - requires `[with_t2i]`
- **text-to-video** (t2v): HunyuanVideo, CogVideoX, Wan2.2 - requires `[with_t2v]` + system ffmpeg
- **text-to-speech** (tts): VibeVoice, Bark, MMS-TTS - requires `[with_tts]`
- **automatic-speech-recognition** (asr/stt): Whisper - requires `[with_stt]`

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
  cli.py              # CLI entry point (thin wrapper)
  core/               # Device detection, task registry
  tasks/              # Task handlers (text_to_image, text_to_video, etc.)
  io/                 # Input/output handling (loaders, savers)
  utils/              # Dependency checking, helpers
tests/                # pytest tests (test_core.py, test_io.py, test_utils.py)
```
