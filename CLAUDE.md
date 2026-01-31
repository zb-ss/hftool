# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

hftool is a CLI for running HuggingFace models, optimized for AMD ROCm but supporting NVIDIA CUDA, Apple MPS, and CPU. It wraps common AI tasks (image/video/speech generation, transcription) without requiring users to write Python.

**Primary target**: AMD GPU owners frustrated with CUDA-first tooling.

## Build & Test Commands

```bash
# Development install
pip install -e ".[dev,all]"

# For AMD ROCm (replace auto-installed CUDA PyTorch):
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# For pipx installs on AMD:
pipx runpip hftool uninstall torch torchvision torchaudio -y
pipx runpip hftool install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_core.py::TestTaskRegistry::test_name

# List tasks and models
hftool --list-tasks
hftool models -t text-to-image

# Test a task (dry-run)
hftool -t t2i -i "A cat" -o cat.png --dry-run

# System diagnostics
hftool doctor
```

## Docker Support (Recommended for AMD ROCm)

Docker provides isolated ROCm 7.1.1 environment without affecting system drivers (safe for gaming systems).

```bash
# Interactive setup wizard
hftool docker setup

# Check hardware and Docker status
hftool docker status

# Build Docker image for your platform
hftool docker build

# Run commands in Docker container
hftool docker run -- -t t2i -i "A cat" -o cat.png

# GPU selection for multi-GPU systems (AMD only)
hftool docker run --gpu 1 -- -t t2v -i "A cat" -o cat.mp4
hftool docker run --gpu auto -- -t t2i -i "A cat" -o cat.png
```

### Docker GPU Selection (AMD ROCm)

For multi-GPU AMD systems, `hftool docker run` uses **device passthrough** instead of environment variables for reliable GPU isolation. Key functions in `hftool/utils/docker.py`:

```python
# Detect GPUs and their render devices
list_amd_gpus() -> List[GPUInfo]  # Returns [GPUInfo(index=0, render_device="/dev/dri/renderD128", ...), ...]

# Map user indices to render devices
get_render_devices_for_gpus([1]) -> ["/dev/dri/renderD129"]

# Interactive picker (shown when multiple GPUs detected)
interactive_gpu_select() -> Optional[List[int]]

# Parse --gpu argument
parse_gpu_arg("auto") -> [1]  # Returns non-display GPU
parse_gpu_arg("0,1") -> [0, 1]
```

When `--gpu 1` is specified, the Docker command passes only that GPU's render device:
```bash
docker run ... --device=/dev/kfd --device=/dev/dri/renderD129 ...
```

This is more reliable than `HIP_VISIBLE_DEVICES` because the container only sees the selected GPU(s).

**Note:** NVIDIA users are unaffected - they still use the standard `--gpus` flag.

### Docker Files

| File | Purpose |
|------|---------|
| `docker/Dockerfile.rocm` | AMD ROCm 7.1.1 + PyTorch 2.9.1 |
| `docker/Dockerfile.cuda` | NVIDIA CUDA 12.4 |
| `docker/Dockerfile.cpu` | CPU-only fallback |
| `docker/docker-compose.yml` | Easy GPU passthrough |
| `hftool/utils/docker.py` | Hardware detection & Docker utilities |

### Building Docker Images

```bash
# Build from project root
docker build -f docker/Dockerfile.rocm -t hftool:rocm --build-arg HFTOOL_VERSION=0.5.0 .

# Or use hftool (auto-passes version)
hftool docker build
```

**Note:** Dockerfiles use `SETUPTOOLS_SCM_PRETEND_VERSION` since `.git` is not in Docker context. The version is passed via `--build-arg` from `hftool.__version__`.

## Architecture

### Plugin Architecture (Tasks)

Tasks are Python modules in `hftool/tasks/` with a `create_task()` factory function that returns a `BaseTask` subclass.

```python
class MyTask(TextInputMixin, BaseTask):
    def load_pipeline(self, model: str, **kwargs) -> Any: ...
    def run_inference(self, pipeline: Any, input_data: Any, **kwargs) -> Any: ...
    def save_output(self, result: Any, output_path: str, **kwargs) -> str: ...
```

Tasks are registered in `hftool/core/registry.py` with `TaskConfig` dataclass.

### Model Registry

Models are defined in `hftool/core/models.py` using `ModelInfo` dataclass:
- `repo_id`: HuggingFace repo ID
- `model_type`: DIFFUSERS, TRANSFORMERS, or CUSTOM
- `size_gb`: Download size for prompts
- `pip_dependencies`: Auto-installed packages

### Configuration Priority (highest to lowest)

1. CLI arguments
2. Environment variables (`HFTOOL_*`)
3. Project config (`./.hftool/config.toml`)
4. User config (`~/.hftool/config.toml`)
5. Built-in defaults

### Device Detection

`hftool/core/device.py` auto-detects: ROCm → CUDA → MPS → CPU. ROCm-first with `HFTOOL_ROCM_PATH` support for using Ollama's bundled ROCm.

### Multi-GPU Architecture

Multi-GPU support is centralized in `hftool/core/device.py` via `get_multi_gpu_kwargs()`:

```python
from hftool.core.device import get_multi_gpu_kwargs

gpu_config = get_multi_gpu_kwargs(reserve_per_gpu_gb=6.0)
if gpu_config["use_multi_gpu"]:
    load_kwargs["device_map"] = gpu_config["device_map"]  # "balanced"
    load_kwargs["max_memory"] = gpu_config["max_memory"]  # per-GPU limits + CPU fallback
```

This ensures consistent multi-GPU behavior across all task handlers (text_to_image.py, text_to_video.py, image_to_image.py). When users select `--gpu all`, the CLI sets `HFTOOL_MULTI_GPU=1` which triggers model distribution via `device_map="balanced"`.

### Dependency System

Optional dependencies are split into extras: `with_t2i`, `with_t2v`, `with_tts`, `with_stt`, `with_interactive`. Use `check_dependency()` from `hftool/utils/deps.py` with try/except for graceful degradation.

## Key Files

| File | Purpose |
|------|---------|
| `cli.py` | Main CLI entry point with all subcommands |
| `core/models.py` | Model registry (MODEL_REGISTRY dict) |
| `core/registry.py` | Task registry (TASK_REGISTRY dict) |
| `core/device.py` | Device detection, multi-GPU via `get_multi_gpu_kwargs()` |
| `tasks/base.py` | Abstract BaseTask class and mixins |
| `io/interactive_mode.py` | Full interactive wizard (-I flag) |
| `utils/errors.py` | Error handling with pattern matching |
| `utils/docker.py` | Docker utilities, hardware detection, GPU device passthrough |
| `docker/` | Dockerfiles for ROCm, CUDA, CPU |

## Adding New Models

1. Edit `hftool/core/models.py`, add entry to `MODEL_REGISTRY[task_name]`:
   ```python
   "short-name": ModelInfo(
       repo_id="org/model-name",
       name="Display Name",
       model_type=ModelType.DIFFUSERS,
       size_gb=5.0,
       is_default=False,
       pip_dependencies=["diffusers>=0.36.0"],  # Required packages
       gated=False,  # Set True if requires HF license acceptance
   ),
   ```
2. Test: `hftool models -t <task>` then `hftool -t <task> -m short-name -i "test" -o test.png`

**Important considerations when adding models:**
- **Check diffusers version**: New pipelines may require unreleased diffusers. Use `pip_dependencies=["git+https://github.com/huggingface/diffusers"]` for main branch
- **Gated models**: Set `gated=True` for models requiring HF license acceptance
- **Custom pipelines**: If model uses a new pipeline class, add loading logic to the task handler (e.g., `image_to_image.py`)
- **Auto-install fallback**: Add try/except with `install_pip_dependencies()` call for new pipeline imports

## Adding New Tasks

1. Create `hftool/tasks/your_task.py` with `BaseTask` subclass and `create_task()` factory
2. Register in `hftool/core/registry.py` TASK_REGISTRY
3. Add models to `hftool/core/models.py`
4. Add tests in `tests/`

## Version Bumping

**Always bump the version when adding features or fixing bugs.**

### Files to Update (ALL of these must match):

| File | Location |
|------|----------|
| `hftool/__init__.py` | `__version__ = "X.Y.Z"` |
| `pyproject.toml` | `fallback_version = "X.Y.Z"` (in `[tool.setuptools_scm]`) |
| `docker/Dockerfile.rocm` | `ARG HFTOOL_VERSION=X.Y.Z` |
| `docker/Dockerfile.cuda` | `ARG HFTOOL_VERSION=X.Y.Z` |
| `docker/Dockerfile.cpu` | `ARG HFTOOL_VERSION=X.Y.Z` |

### Pre-1.0 Versioning (Current)

While the project is at `0.x.y`, we use conservative versioning to avoid rushing toward 1.0:

| Change Type | Bump | Examples |
|-------------|------|----------|
| Breaking CLI/API changes | MINOR (0.X.0) | Removing a flag, changing output format |
| New task types | MINOR | Adding image-to-3d, video-to-video |
| New models | PATCH (0.0.X) | Adding another SDXL variant, new TTS voice |
| New CLI options | PATCH | Adding --quiet, --json flags |
| Bug fixes | PATCH | Docker path issues, warning suppression |
| UX improvements | PATCH | Better progress bars, cleaner output |
| Documentation | PATCH | README updates, docstrings |
| Internal refactors | PATCH | Code cleanup without behavior changes |

**1.0 criteria:** Stable API, comprehensive test coverage, complete documentation, battle-tested in real-world use.

### Post-1.0 Versioning (Future)

Once stable at 1.0+, follow standard [Semantic Versioning](https://semver.org/):
- **MAJOR** (X.0.0): Breaking changes to CLI interface or API
- **MINOR** (0.X.0): New features (backward-compatible)
- **PATCH** (0.0.X): Bug fixes

**Bump version in the same commit as the feature/fix, not separately.**

## Code Style

- Python 3.10+ with type hints required
- `snake_case` for functions/variables, `PascalCase` for classes
- ~120 char line limit
- Imports: stdlib → third-party → local, with try/except for optional deps
- Use `HFToolError` for user-facing errors (from `hftool/utils/errors.py`)
- Use `check_dependency()` for optional package checks

## Error Handling Pattern

```python
from hftool.utils.errors import HFToolError
from hftool.utils.deps import check_dependency

# Check optional dependency
check_dependency("diffusers", extra="with_t2i")

# Raise user-friendly error
raise HFToolError("Message", suggestion="How to fix it")
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `HFTOOL_MODELS_DIR` | Model storage (default: `~/.hftool/models/`) |
| `HFTOOL_AUTO_DOWNLOAD` | Skip download prompts (1=enabled) |
| `HFTOOL_DEBUG` | Show all warnings (1=enabled) |
| `HFTOOL_LOG_FILE` | Log to file |
| `HFTOOL_MULTI_GPU` | Multi-GPU mode: `1`/`balanced` enables, `0` disables |
| `HFTOOL_CPU_OFFLOAD` | CPU offload level: `0` disabled, `1` model, `2` sequential |
| `HFTOOL_ROCM_PATH` | AMD ROCm library path |
| `HSA_OVERRIDE_GFX_VERSION` | AMD GPU architecture |
| `HF_TOKEN` | HuggingFace token for gated models |

## Supported Tasks

- **text-to-image** (t2i): Z-Image, SDXL, FLUX
- **image-to-image** (i2i): Qwen Image Edit, FLUX.2 Klein (non-commercial), SDXL
- **text-to-video** (t2v): LTX-2, HunyuanVideo, CogVideoX, Wan2.2
- **image-to-video** (i2v): LTX-2 I2V, HunyuanVideo I2V
- **text-to-speech** (tts): Bark, MMS-TTS
- **automatic-speech-recognition** (asr/stt): Whisper
- Various transformers pipeline tasks (text-generation, summarization, etc.)

## External Dependencies

- **ffmpeg**: Required for video/audio processing (system package)
- **torch**: Users install for their platform (ROCm/CUDA/MPS/CPU)
- **Docker**: Optional, recommended for AMD ROCm (isolated environment)
