# Contributing to hftool

Thank you for your interest in contributing to hftool! This guide will help you get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/zb-ss/hftool.git
cd hftool

# Install PyTorch for your platform first:
# NVIDIA: pip install torch torchvision torchaudio
# AMD ROCm: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
# CPU only: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install hftool in development mode
pip install -e ".[dev,all]"

# Run tests
pytest tests/ -v
```

## Adding a New Model

Adding a new model to hftool is straightforward. Here's how:

### Step 1: Add to Model Registry

Edit `hftool/core/models.py` and add your model to the appropriate task in `MODEL_REGISTRY`:

```python
MODEL_REGISTRY: Dict[str, Dict[str, ModelInfo]] = {
    "text-to-image": {
        # Add your model here
        "your-model-name": ModelInfo(
            repo_id="organization/model-name",  # HuggingFace repo ID
            name="Your Model Name",              # Human-readable name
            model_type=ModelType.DIFFUSERS,      # DIFFUSERS, TRANSFORMERS, or CUSTOM
            size_gb=5.0,                         # Approximate download size
            is_default=False,                    # Set True if this should be the default
            description="Brief description",     # Shown in `hftool models`
            # Optional fields:
            pip_dependencies=["some-package"],   # Auto-installed when model is used
            ignore_patterns=["*.bin"],           # Files to skip during download
            metadata={"key": "value"},           # Model-specific settings
        ),
    },
}
```

### Step 2: Test Your Model

```bash
# List models to verify it appears
hftool models -t text-to-image

# Download the model
hftool download -t text-to-image -m your-model-name

# Test inference
hftool -t t2i -m your-model-name -i "A test prompt" -o test.png
```

### Step 3: Custom Task Handler (if needed)

Most models work with the existing task handlers. If your model needs special handling:

1. Check if it works with the existing handler first
2. If not, add model-specific logic to the appropriate task handler in `hftool/tasks/`

Example for a model needing custom loading in `text_to_image.py`:

```python
def load_pipeline(self, model: str, **kwargs) -> Any:
    model_lower = model.lower()
    
    # Add custom handling for your model
    if "your-model" in model_lower:
        return self._load_your_model(model, **kwargs)
    
    # ... existing code
```

## Adding a New Task

To add an entirely new task type:

### Step 1: Create Task Handler

Create a new file `hftool/tasks/your_task.py`:

```python
"""Your task description."""

from typing import Any, Dict, Optional
from hftool.tasks.base import BaseTask, TextInputMixin  # or other mixins

class YourTask(TextInputMixin, BaseTask):
    """Handler for your task.
    
    Supported models:
    - model-a (org/model-a)
    - model-b (org/model-b)
    """
    
    def load_pipeline(self, model: str, **kwargs) -> Any:
        """Load model/pipeline."""
        # Your loading logic
        pass
    
    def get_default_kwargs(self) -> Dict[str, Any]:
        """Default inference parameters."""
        return {}
    
    def run_inference(self, pipeline: Any, input_data: Any, **kwargs) -> Any:
        """Run inference."""
        # Your inference logic
        pass
    
    def save_output(self, result: Any, output_path: str, **kwargs) -> str:
        """Save results to file."""
        # Your save logic
        pass


def create_task(device: str = "auto", dtype: Optional[str] = None) -> YourTask:
    """Factory function."""
    return YourTask(device=device, dtype=dtype)
```

### Step 2: Register the Task

Edit `hftool/core/registry.py`:

```python
TASK_REGISTRY = {
    # ... existing tasks
    "your-task": {
        "module": "hftool.tasks.your_task",
        "description": "Your task description",
        "input_type": "text",  # or "image", "audio", etc.
        "output_type": "image",  # or "text", "audio", "video"
    },
}

TASK_ALIASES = {
    # ... existing aliases
    "yt": "your-task",  # Short alias
}
```

### Step 3: Add Models

Add models to `MODEL_REGISTRY` as shown above, using your new task name as the key.

## Model Selection Guidelines

When adding models, consider:

1. **Quality vs Size**: Include both lightweight and full-quality options
2. **Licensing**: Prefer models with permissive licenses (MIT, Apache 2.0)
3. **Popularity**: Well-maintained models with active communities
4. **Hardware**: Test on both NVIDIA and AMD if possible

## Code Style

- **Python 3.10+** with type hints
- **snake_case** for variables/functions, **PascalCase** for classes
- Run tests before submitting: `pytest tests/ -v`
- Keep functions focused and under 50 lines when possible

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/add-new-model`
3. Make your changes
4. Add tests if applicable
5. Run tests: `pytest tests/ -v`
6. Commit with a descriptive message
7. Push and create a Pull Request

## Requesting New Models

If you'd like a model added but can't contribute code:

1. Open an issue with the "model request" label
2. Include:
   - HuggingFace model URL
   - What task it performs
   - Why it would be valuable (quality, speed, unique capabilities)

## Questions?

Open an issue or start a discussion. We're happy to help!
