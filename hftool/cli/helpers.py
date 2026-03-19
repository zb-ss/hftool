"""Shared CLI helper functions.

These are used across multiple command modules.
"""

import os
import sys
from typing import Any, Optional

import click


def parse_extra_args(args: list) -> dict:
    """Parse extra arguments passed after --.

    Converts --arg value pairs to a dictionary.
    Handles boolean flags (--flag with no value).
    """
    kwargs = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            key = arg[2:].replace("-", "_")

            # Check if next arg is a value or another flag
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                value = args[i + 1]
                # Try to parse as number or boolean
                value = parse_value(value)
                kwargs[key] = value
                i += 2
            else:
                # Boolean flag
                kwargs[key] = True
                i += 1
        else:
            i += 1

    return kwargs


def parse_value(value: str):
    """Parse a string value to appropriate type."""
    # Boolean
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False

    # Integer
    try:
        return int(value)
    except ValueError:
        pass

    # Float
    try:
        return float(value)
    except ValueError:
        pass

    # String
    return value


def should_open_output(
    open_output: Optional[bool],
    output_file: str,
    output_type: str,
) -> bool:
    """Determine if we should open the output file.

    Args:
        open_output: User's explicit preference (True/False/None)
        output_file: Path to the output file
        output_type: Type of output ("image", "audio", "video", "text")

    Returns:
        True if we should attempt to open the file
    """
    # If user explicitly specified, use that
    if open_output is not None:
        return open_output

    # Check environment variable
    env_open = os.environ.get("HFTOOL_AUTO_OPEN", "").lower()
    if env_open in ("1", "true", "yes"):
        return True
    if env_open in ("0", "false", "no"):
        return False

    # Auto-detect based on output type
    # By default, open image, audio, and video files (not text)
    openable_types = {"image", "audio", "video"}
    return output_type in openable_types


def open_file(file_path: str, verbose: bool = False) -> bool:
    """Open a file with the system's default application.

    Args:
        file_path: Path to the file to open
        verbose: Whether to print detailed messages

    Returns:
        True if the file was opened successfully
    """
    import platform
    import subprocess
    from pathlib import Path

    # Security: Validate file path
    try:
        path = Path(file_path).resolve()

        # Check file exists
        if not path.exists():
            click.echo(f"Cannot open file: {file_path} (file not found)", err=True)
            return False

        # Check it's a regular file (not a directory, symlink to dangerous location, etc.)
        if not path.is_file():
            click.echo(f"Cannot open file: {file_path} (not a regular file)", err=True)
            return False

        # Use the validated absolute path
        file_path = str(path)

    except Exception as e:
        click.echo(f"Cannot open file: invalid path ({e})", err=True)
        return False

    system = platform.system().lower()

    try:
        if system == "darwin":  # macOS
            cmd = ["open", file_path]
        elif system == "windows":
            # On Windows, use os.startfile (no subprocess needed)
            os.startfile(file_path)  # type: ignore
            if verbose:
                click.echo(f"Opened: {file_path}")
            return True
        elif system == "linux":
            # Try xdg-open first (standard), then common alternatives
            openers = ["xdg-open", "gnome-open", "kde-open", "exo-open"]
            cmd = None

            for opener in openers:
                import shutil
                if shutil.which(opener):
                    cmd = [opener, file_path]
                    break

            if cmd is None:
                click.echo(
                    f"Cannot open file: no file opener found. "
                    f"Install xdg-utils or open manually: {file_path}",
                    err=True
                )
                return False
        else:
            click.echo(f"Cannot open file: unsupported platform '{system}'", err=True)
            return False

        # Execute the open command
        if verbose:
            click.echo(f"Opening: {file_path}")

        # Use Popen to not block and detach the process
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        if verbose:
            click.echo(f"Opened: {file_path}")

        return True

    except FileNotFoundError as e:
        click.echo(f"Cannot open file: application not found ({e})", err=True)
        return False
    except PermissionError as e:
        click.echo(f"Cannot open file: permission denied ({e})", err=True)
        return False
    except Exception as e:
        click.echo(f"Cannot open file: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def get_console():
    """Get a Rich console for formatted output."""
    from rich.console import Console
    return Console()


def list_tasks_display():
    """Print list of available tasks."""
    from hftool.core.registry import list_tasks, TASK_ALIASES

    click.echo("Available tasks:")
    click.echo("")

    tasks = list_tasks()
    for name, description in sorted(tasks.items()):
        click.echo(f"  {name}")
        click.echo(f"    {description}")

    click.echo("")
    click.echo("Task aliases:")
    for alias, target in sorted(TASK_ALIASES.items()):
        click.echo(f"  {alias} -> {target}")


def resolve_model_to_repo_id(model: str) -> str:
    """Resolve a model name/shortname to repo_id."""
    from hftool.core.models import find_model_by_repo_id, MODEL_REGISTRY

    # Try to find by repo_id
    found = find_model_by_repo_id(model)
    if found:
        return found[2].repo_id

    # Try to find by short name across all tasks
    for task_name, models in MODEL_REGISTRY.items():
        if model in models:
            return models[model].repo_id

    # Assume it's a repo_id directly
    return model


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size = size / 1024
    return f"{size:.1f} PB"


def interactive_model_select(models: list) -> list:
    """Interactive model selection using numbered list.

    Args:
        models: List of model info dicts from get_models_disk_usage()

    Returns:
        List of selected model info dicts
    """
    from hftool.core.models import find_model_by_repo_id

    click.echo("")
    click.echo("Downloaded models:")
    click.echo("-" * 60)

    # Display numbered list
    for i, model_info in enumerate(models, 1):
        repo_id = model_info["repo_id"]
        size_str = model_info["size_str"]

        # Try to get friendly name
        found = find_model_by_repo_id(repo_id)
        if found:
            task_name, short_name, info = found
            display_name = f"{info.name} ({task_name})"
        else:
            display_name = repo_id

        click.echo(f"  [{i:2d}] {display_name}")
        click.echo(f"       {repo_id} - {size_str}")

    click.echo("-" * 60)
    click.echo("")
    click.echo("Enter model numbers to delete (comma-separated, ranges with -, or 'all'):")
    click.echo("Examples: 1,3,5  or  1-3  or  1,3-5,7  or  all")
    click.echo("")

    try:
        selection = click.prompt("Selection", default="").strip()
    except click.Abort:
        return []

    if not selection:
        return []

    # Parse selection
    selected_indices = set()

    if selection.lower() == "all":
        return models

    try:
        parts = selection.split(",")
        for part in parts:
            part = part.strip()
            if "-" in part:
                # Range (e.g., "1-5")
                start, end = part.split("-", 1)
                start_idx = int(start.strip())
                end_idx = int(end.strip())
                for i in range(start_idx, end_idx + 1):
                    if 1 <= i <= len(models):
                        selected_indices.add(i - 1)
            else:
                # Single number
                idx = int(part)
                if 1 <= idx <= len(models):
                    selected_indices.add(idx - 1)
    except ValueError:
        click.echo("Invalid selection format.", err=True)
        return []

    return [models[i] for i in sorted(selected_indices)]


def check_task_deps(task_config, verbose: bool):
    """Check if required dependencies are installed."""
    from hftool.utils.deps import is_available, is_ffmpeg_available

    missing = []
    for dep in task_config.required_deps:
        if not is_available(dep):
            missing.append(dep)

    if missing:
        click.echo(f"Missing dependencies: {', '.join(missing)}", err=True)
        click.echo(f"Install with: pip install {' '.join(missing)}", err=True)
        sys.exit(1)

    if task_config.requires_ffmpeg and not is_ffmpeg_available():
        click.echo("ffmpeg is required for this task but was not found.", err=True)
        click.echo("Please install ffmpeg: https://ffmpeg.org/download.html", err=True)
        sys.exit(1)

    if verbose:
        click.echo("All dependencies satisfied.")
