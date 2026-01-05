"""Full interactive mode for hftool.

Provides a complete guided wizard experience for running any task.
Users can select task, model, input, output, and all options interactively.

Usage:
    hftool --interactive
    hftool -I
    
Or set in config/env:
    # ~/.hftool/config.toml
    [defaults]
    interactive = true
    
    # Environment
    HFTOOL_INTERACTIVE=1
"""

import os
import sys
import random
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import click


def is_inquirer_available() -> bool:
    """Check if InquirerPy is available."""
    try:
        from InquirerPy import inquirer
        return True
    except ImportError:
        return False


def run_interactive_mode(
    quiet: bool = False,
    output_json: bool = False,
) -> Dict[str, Any]:
    """Run the full interactive wizard.
    
    Guides the user through:
    1. Task selection
    2. Model selection (with download status)
    3. Input (text, file, or JSON builder)
    4. Output file
    5. Device and dtype
    6. Seed
    7. Extra parameters
    
    Args:
        quiet: Suppress non-essential output
        output_json: Output result as JSON
    
    Returns:
        Dictionary with all parameters needed to run the task
    
    Raises:
        click.Abort: If user cancels
        ValueError: If InquirerPy not available and fallback fails
    """
    # Check for InquirerPy
    if not is_inquirer_available():
        click.echo("Interactive mode requires InquirerPy.", err=True)
        click.echo("Install with: pip install InquirerPy", err=True)
        click.echo("", err=True)
        click.echo("Alternatively, use command-line options:", err=True)
        click.echo("  hftool -t <task> -i <input> -o <output>", err=True)
        sys.exit(1)
    
    from InquirerPy import inquirer
    from InquirerPy.base.control import Choice
    from InquirerPy.separator import Separator
    
    from hftool.core.registry import TASK_REGISTRY, TASK_ALIASES
    from hftool.core.models import MODEL_REGISTRY, get_models_for_task
    from hftool.core.download import get_download_status
    
    click.echo("")
    click.echo(click.style("╔══════════════════════════════════════════════════════════╗", fg="cyan"))
    click.echo(click.style("║           hftool - Interactive Mode                      ║", fg="cyan"))
    click.echo(click.style("╚══════════════════════════════════════════════════════════╝", fg="cyan"))
    click.echo("")
    
    try:
        # Step 1: Select task
        task = _select_task(inquirer, Choice, Separator)
        
        # Step 2: Select model
        model = _select_model(inquirer, Choice, Separator, task)
        
        # Step 3: Get input
        input_data = _get_input(inquirer, task)
        
        # Step 4: Output file
        output_file = _get_output(inquirer, task)
        
        # Step 5: Device
        device = _select_device(inquirer, Choice)
        
        # Step 6: Dtype (optional)
        dtype = _select_dtype(inquirer, Choice)
        
        # Step 7: Seed
        seed = _get_seed(inquirer)
        
        # Step 8: Extra parameters (optional)
        extra_kwargs = _get_extra_params(inquirer, task)
        
        # Show summary
        click.echo("")
        click.echo(click.style("═══ Summary ═══", fg="cyan", bold=True))
        click.echo(f"  Task:   {task}")
        click.echo(f"  Model:  {model or '(default)'}")
        
        # Truncate long input for display
        input_display = input_data
        if len(input_display) > 50:
            input_display = input_display[:47] + "..."
        click.echo(f"  Input:  {input_display}")
        click.echo(f"  Output: {output_file or '(auto)'}")
        click.echo(f"  Device: {device}")
        click.echo(f"  Dtype:  {dtype or '(auto)'}")
        click.echo(f"  Seed:   {seed}")
        if extra_kwargs:
            click.echo(f"  Params: {extra_kwargs}")
        click.echo("")
        
        # Confirm
        if not inquirer.confirm(message="Run with these settings?", default=True).execute():
            raise click.Abort()
        
        return {
            "task": task,
            "model": model,
            "input_data": input_data,
            "output_file": output_file,
            "device": device,
            "dtype": dtype,
            "seed": seed,
            "extra_kwargs": extra_kwargs,
            "quiet": quiet,
            "output_json": output_json,
        }
    
    except KeyboardInterrupt:
        click.echo("")
        click.echo("Cancelled.")
        raise click.Abort()


def _select_task(inquirer, Choice, Separator) -> str:
    """Select a task interactively.
    
    Returns:
        Selected task name
    """
    from hftool.core.registry import TASK_REGISTRY
    
    # Group tasks by category
    categories = {
        "Image Generation": ["text-to-image", "image-to-image"],
        "Video Generation": ["text-to-video", "image-to-video"],
        "Audio/Speech": ["text-to-speech", "automatic-speech-recognition"],
        "Text/NLP": ["text-generation", "text-classification", "question-answering", 
                     "summarization", "translation"],
        "Vision": ["image-classification", "object-detection", "image-to-text"],
    }
    
    choices = []
    for category, tasks in categories.items():
        choices.append(Separator(f"─── {category} ───"))
        for task_name in tasks:
            if task_name in TASK_REGISTRY:
                config = TASK_REGISTRY[task_name]
                choices.append(Choice(
                    value=task_name,
                    name=f"{task_name}: {config.description}",
                ))
    
    return inquirer.select(
        message="Select a task:",
        choices=choices,
        default="text-to-image",
    ).execute()


def _select_model(inquirer, Choice, Separator, task: str) -> Optional[str]:
    """Select a model for the task.
    
    Args:
        task: Task name
    
    Returns:
        Model short name or None for default
    """
    from hftool.core.models import MODEL_REGISTRY, get_models_for_task
    from hftool.core.download import get_download_status
    from hftool.core.registry import TASK_ALIASES
    
    # Resolve task alias
    resolved_task = TASK_ALIASES.get(task, task)
    
    if resolved_task not in MODEL_REGISTRY:
        # No models registered - use default
        return None
    
    models = get_models_for_task(resolved_task)
    
    if not models:
        return None
    
    choices = []
    default_model = None
    
    for short_name, info in models.items():
        status = get_download_status(info.repo_id)
        
        # Status indicator
        if status == "downloaded":
            status_icon = click.style("✓", fg="green")
        elif status == "partial":
            status_icon = click.style("~", fg="yellow")
        else:
            status_icon = " "
        
        # Default marker
        default_mark = ""
        if info.is_default:
            default_mark = click.style(" (default)", fg="cyan")
            default_model = short_name
        
        name = f"[{status_icon}] {info.name} ({info.size_str}){default_mark}"
        choices.append(Choice(value=short_name, name=name))
    
    choices.append(Separator("───────────────────"))
    choices.append(Choice(value=None, name="Use default model"))
    
    return inquirer.select(
        message="Select a model:",
        choices=choices,
        default=default_model,
    ).execute()


def _get_input(inquirer, task: str) -> str:
    """Get input for the task.
    
    Determines input type from task and prompts appropriately.
    
    Args:
        task: Task name
    
    Returns:
        Input string (text, file path, or JSON)
    """
    import json
    from hftool.core.registry import get_task_config, TASK_ALIASES
    from hftool.core.parameters import get_task_schema
    from hftool.io.file_picker import FilePicker, FileType
    
    config = get_task_config(task)
    resolved_task = TASK_ALIASES.get(task, task)
    
    if config.input_type == "text":
        # Text input - simple prompt
        click.echo("")
        click.echo(click.style("Enter your prompt:", fg="cyan"))
        click.echo(click.style("(Multi-line: end with empty line, or use Ctrl+D)", fg="white", dim=True))
        
        lines = []
        while True:
            try:
                line = input("> " if not lines else "  ")
                if not line and lines:
                    break
                lines.append(line)
            except EOFError:
                break
        
        text = "\n".join(lines)
        
        if not text.strip():
            click.echo("Error: Prompt cannot be empty", err=True)
            return _get_input(inquirer, task)
        
        return text
    
    elif config.input_type in ("image", "audio", "video"):
        # File input
        file_type_map = {
            "image": FileType.IMAGE,
            "audio": FileType.AUDIO,
            "video": FileType.VIDEO,
        }
        file_type = file_type_map.get(config.input_type, FileType.ALL)
        
        # Get the file first
        file_path = _get_file_with_navigation(inquirer, config.input_type, file_type, task)
        
        # For image-to-image, also get a prompt
        if resolved_task == "image-to-image":
            click.echo("")
            click.echo(click.style("Enter edit prompt (describe what changes you want):", fg="cyan"))
            click.echo(click.style("(Multi-line: end with empty line, or use Ctrl+D)", fg="white", dim=True))
            
            lines = []
            while True:
                try:
                    line = input("> " if not lines else "  ")
                    if not line and lines:
                        break
                    lines.append(line)
                except EOFError:
                    break
            
            prompt = "\n".join(lines).strip()
            
            # Return JSON format for i2i
            return json.dumps({"image": file_path, "prompt": prompt})
        
        # For image-to-video, also get a prompt
        if resolved_task == "image-to-video":
            click.echo("")
            click.echo(click.style("Enter motion prompt (describe the video motion):", fg="cyan"))
            click.echo(click.style("(Multi-line: end with empty line, or use Ctrl+D)", fg="white", dim=True))
            
            lines = []
            while True:
                try:
                    line = input("> " if not lines else "  ")
                    if not line and lines:
                        break
                    lines.append(line)
                except EOFError:
                    break
            
            prompt = "\n".join(lines).strip()
            
            # Return JSON format for i2v
            return json.dumps({"image": file_path, "prompt": prompt})
        
        return file_path
    
    else:
        # Complex input - check for schema
        schema = get_task_schema(task)
        
        if schema:
            # Use interactive JSON builder
            use_builder = inquirer.confirm(
                message="Use interactive input builder?",
                default=True,
            ).execute()
            
            if use_builder:
                from hftool.io.interactive_input import build_interactive_input
                return build_interactive_input(task)
        
        # Fallback to manual text/JSON input
        click.echo("")
        click.echo("Enter input (text or JSON):")
        return inquirer.text(
            message="Input:",
            validate=lambda x: len(x.strip()) > 0,
        ).execute()


def _get_file_with_navigation(inquirer, input_type: str, file_type, task: str) -> str:
    """Get a file path with directory navigation support.
    
    Args:
        inquirer: InquirerPy module
        input_type: Type of input (image, audio, video)
        file_type: FileType enum
        task: Task name
    
    Returns:
        Selected file path
    """
    from hftool.io.file_picker import FilePicker
    
    while True:
        # Show file picker options
        input_method = inquirer.select(
            message=f"How to provide {input_type} input?",
            choices=[
                {"name": "Browse files (current directory)", "value": "@"},
                {"name": "Browse files (home directory)", "value": "@~"},
                {"name": "Recent files from history", "value": "@@"},
                {"name": "Enter path manually", "value": "manual"},
            ],
            default="@",
        ).execute()
        
        if input_method == "manual":
            file_path = inquirer.filepath(
                message=f"Enter {input_type} file path:",
                validate=lambda x: Path(os.path.expanduser(x)).exists(),
                only_files=True,
            ).execute()
            # Expand ~ to home directory
            return os.path.expanduser(file_path)
        else:
            # Use file picker with navigation
            picker = FilePicker(file_type=file_type)
            try:
                result = _pick_file_with_navigation(inquirer, picker, input_method, task, file_type)
                if result:
                    return result
                # If result is None, loop back to show options again
            except ValueError as e:
                click.echo(f"Error: {e}", err=True)
                # Loop back to show options again


def _pick_file_with_navigation(inquirer, picker, reference: str, task: str, file_type) -> Optional[str]:
    """Pick a file with directory navigation support and fuzzy search.
    
    Args:
        inquirer: InquirerPy module
        picker: FilePicker instance
        reference: File reference string (@, @~, @@)
        task: Task name
        file_type: FileType enum
    
    Returns:
        Selected file path or None to go back to method selection
    """
    from InquirerPy.base.control import Choice
    from InquirerPy.separator import Separator
    
    # Handle history reference directly
    if reference == "@@":
        return picker.resolve_reference(reference, task=task)
    
    # Determine starting directory
    if reference == "@~":
        current_dir = Path.home()
    elif reference.startswith("@/"):
        current_dir = Path(reference[1:]).expanduser().resolve()
    else:
        current_dir = Path.cwd()
    
    while True:
        # Get files and directories in current location
        directories = []
        files = []
        
        try:
            for item in sorted(current_dir.iterdir()):
                # Skip hidden files
                if item.name.startswith("."):
                    continue
                
                if item.is_dir():
                    directories.append(item)
                elif item.is_file() and picker._matches_file_type(item):
                    files.append(item)
        except PermissionError:
            click.echo(f"Error: Permission denied accessing {current_dir}", err=True)
            return None
        
        total_files = len(files)
        total_dirs = len(directories)
        
        # If there are many files, use fuzzy search instead of showing all
        if total_files > 50:
            result = _fuzzy_file_search(inquirer, picker, current_dir, files, directories)
            if result == "__BACK__":
                return None
            elif result == "__UP__":
                current_dir = current_dir.parent
                continue
            elif isinstance(result, tuple):
                item_type, item_path = result
                if item_type == "dir":
                    current_dir = item_path
                    continue
                else:
                    return str(item_path)
            elif result is None:
                return None
            continue
        
        # Build choices for smaller directories
        choices = []
        
        # Navigation options
        choices.append(Separator("--- Navigation ---"))
        choices.append(Choice(value="__BACK__", name="[..] Go back to input method selection"))
        if current_dir != current_dir.parent:
            parent_name = current_dir.parent.name or "/"
            choices.append(Choice(value="__UP__", name=f"[..] Parent directory ({parent_name})"))
        
        # Directories
        if directories:
            choices.append(Separator(f"--- Folders ({total_dirs}) ---"))
            for d in directories[:50]:
                choices.append(Choice(value=("dir", d), name=f"[DIR]  {d.name}/"))
            if total_dirs > 50:
                choices.append(Separator(f"    ... and {total_dirs - 50} more folders"))
        
        # Files
        if files:
            choices.append(Separator(f"--- Files ({total_files}) ---"))
            for f in files:
                size = f.stat().st_size
                size_str = picker._format_size(size)
                choices.append(Choice(value=("file", f), name=f"[FILE] {f.name} ({size_str})"))
        
        if not files and not directories:
            click.echo(f"No matching files or folders in {current_dir}", err=True)
        
        # Show prompt with current directory
        click.echo("")
        click.echo(click.style(f"Current: {current_dir}", fg="cyan", dim=True))
        click.echo(click.style("(Use arrow keys, Enter to select, Ctrl+C to go back)", dim=True))
        
        try:
            result = inquirer.select(
                message="Select file or folder:",
                choices=choices,
            ).execute()
        except KeyboardInterrupt:
            return None
        
        if result is None or result == "__BACK__":
            return None
        elif result == "__UP__":
            current_dir = current_dir.parent
        elif isinstance(result, tuple):
            item_type, item_path = result
            if item_type == "dir":
                current_dir = item_path
            else:
                return str(item_path)


def _fuzzy_file_search(inquirer, picker, current_dir: Path, files: list, directories: list) -> Optional[Any]:
    """Fuzzy search for files in a directory with many items.
    
    Args:
        inquirer: InquirerPy module
        picker: FilePicker instance
        current_dir: Current directory path
        files: List of file paths
        directories: List of directory paths
    
    Returns:
        Tuple of (type, path), "__BACK__", "__UP__", or None
    """
    from InquirerPy.base.control import Choice
    
    total_files = len(files)
    total_dirs = len(directories)
    
    click.echo("")
    click.echo(click.style(f"Current: {current_dir}", fg="cyan", dim=True))
    click.echo(click.style(f"Found {total_files} files and {total_dirs} folders", fg="yellow"))
    click.echo(click.style("Type to search, use arrow keys to navigate results", dim=True))
    
    # Build all choices for fuzzy search
    choices = []
    
    # Navigation options first
    choices.append(Choice(value="__BACK__", name="[..] Go back to input method selection"))
    if current_dir != current_dir.parent:
        parent_name = current_dir.parent.name or "/"
        choices.append(Choice(value="__UP__", name=f"[..] Parent directory ({parent_name})"))
    
    # Add all directories
    for d in directories:
        choices.append(Choice(value=("dir", d), name=f"[DIR]  {d.name}/"))
    
    # Add all files with size info
    for f in files:
        try:
            size = f.stat().st_size
            size_str = picker._format_size(size)
            choices.append(Choice(value=("file", f), name=f"[FILE] {f.name} ({size_str})"))
        except OSError:
            # File might have been deleted or inaccessible
            continue
    
    try:
        result = inquirer.fuzzy(
            message="Search files (type to filter):",
            choices=choices,
            max_height="70%",
        ).execute()
    except KeyboardInterrupt:
        return None
    
    return result


def _get_output(inquirer, task: str) -> Optional[str]:
    """Get output file path.
    
    Args:
        task: Task name
    
    Returns:
        Output file path or None for auto
    """
    from hftool.core.registry import get_task_config
    
    config = get_task_config(task)
    
    # Suggest extension based on output type
    ext_map = {
        "image": ".png",
        "audio": ".wav",
        "video": ".mp4",
        "text": ".txt",
    }
    default_ext = ext_map.get(config.output_type, ".out")
    
    click.echo("")
    output_choice = inquirer.select(
        message="Output file:",
        choices=[
            {"name": f"Auto-generate (output{default_ext})", "value": "auto"},
            {"name": "Specify path", "value": "manual"},
        ],
        default="auto",
    ).execute()
    
    if output_choice == "auto":
        return None
    
    output_path = inquirer.text(
        message=f"Output path (suggested: output{default_ext}):",
        default=f"output{default_ext}",
    ).execute()
    
    # Expand ~ to home directory
    if output_path:
        output_path = os.path.expanduser(output_path)
    
    return output_path


def _select_device(inquirer, Choice) -> str:
    """Select compute device.
    
    Returns:
        Device string
    """
    # Detect available devices
    devices = [
        Choice(value="auto", name="auto - Auto-detect best device"),
    ]
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            devices.append(Choice(value="cuda", name=f"cuda - {gpu_name}"))
            
            # Multiple GPUs
            if torch.cuda.device_count() > 1:
                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    devices.append(Choice(value=f"cuda:{i}", name=f"cuda:{i} - {name}"))
        
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append(Choice(value="mps", name="mps - Apple Silicon GPU"))
    except ImportError:
        pass
    
    devices.append(Choice(value="cpu", name="cpu - CPU only (slow)"))
    
    return inquirer.select(
        message="Device:",
        choices=devices,
        default="auto",
    ).execute()


def _select_dtype(inquirer, Choice) -> Optional[str]:
    """Select data type.
    
    Returns:
        Dtype string or None for auto
    """
    return inquirer.select(
        message="Data type:",
        choices=[
            Choice(value=None, name="auto - Auto-detect"),
            Choice(value="float16", name="float16 - Half precision (less VRAM)"),
            Choice(value="bfloat16", name="bfloat16 - Brain float (good balance)"),
            Choice(value="float32", name="float32 - Full precision (more VRAM)"),
        ],
        default=None,
    ).execute()


def _get_seed(inquirer) -> int:
    """Get random seed.
    
    Returns:
        Seed value
    """
    random_seed = random.randint(0, 2**32 - 1)
    
    seed_choice = inquirer.select(
        message="Seed:",
        choices=[
            {"name": f"Random ({random_seed})", "value": "random"},
            {"name": "Specify seed", "value": "manual"},
        ],
        default="random",
    ).execute()
    
    if seed_choice == "random":
        return random_seed
    
    seed_str = inquirer.text(
        message="Enter seed (integer):",
        validate=lambda x: x.isdigit(),
    ).execute()
    
    return int(seed_str)


def _get_extra_params(inquirer, task: str) -> Dict[str, Any]:
    """Get extra parameters for the task.
    
    Args:
        task: Task name
    
    Returns:
        Dictionary of extra parameters
    """
    from hftool.core.registry import get_task_config
    from hftool.core.parameters import get_task_schema
    
    # Check if task has common parameters to suggest
    config = get_task_config(task)
    
    # Common parameters by task type
    common_params = {}
    
    if config.output_type == "image":
        common_params = {
            "num_inference_steps": "Number of inference steps (default: varies by model)",
            "guidance_scale": "Guidance scale / CFG (default: varies by model)",
            "width": "Output width in pixels",
            "height": "Output height in pixels",
        }
    elif config.output_type == "video":
        common_params = {
            "num_inference_steps": "Number of inference steps",
            "num_frames": "Number of frames to generate",
            "fps": "Frames per second",
        }
    elif config.output_type == "audio":
        common_params = {
            "max_length": "Maximum output length",
        }
    
    if not common_params:
        return {}
    
    # Ask if user wants to set extra params
    set_params = inquirer.confirm(
        message="Set advanced parameters?",
        default=False,
    ).execute()
    
    if not set_params:
        return {}
    
    click.echo("")
    click.echo(click.style("Available parameters:", fg="cyan"))
    for param, desc in common_params.items():
        click.echo(f"  {param}: {desc}")
    click.echo("")
    
    # Get JSON input for params
    params_str = inquirer.text(
        message="Enter parameters as JSON (e.g., {\"num_inference_steps\": 30}):",
        default="{}",
    ).execute()
    
    try:
        import json
        return json.loads(params_str)
    except json.JSONDecodeError:
        click.echo("Invalid JSON, skipping extra parameters", err=True)
        return {}


def check_interactive_mode(
    ctx: click.Context,
    interactive_flag: bool,
) -> bool:
    """Check if interactive mode should be activated.
    
    Checks (in priority order):
    1. --interactive / -I CLI flag
    2. HFTOOL_INTERACTIVE environment variable
    3. interactive = true in config file
    
    Args:
        ctx: Click context
        interactive_flag: Value of --interactive CLI flag
    
    Returns:
        True if interactive mode should be used
    """
    # 1. CLI flag takes priority
    if interactive_flag:
        return True
    
    # 2. Environment variable
    env_interactive = os.environ.get("HFTOOL_INTERACTIVE", "").lower()
    if env_interactive in ("1", "true", "yes"):
        return True
    
    # 3. Config file
    try:
        from hftool.core.config import Config
        config = Config.get()
        if config.get_value("interactive", default=False):
            return True
    except Exception:
        pass
    
    return False
