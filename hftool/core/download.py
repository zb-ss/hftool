"""Model download manager for hftool.

Handles downloading models from HuggingFace Hub with:
- Configurable storage location (HFTOOL_MODELS_DIR or ~/.hftool/models/)
- Progress bar support
- Interactive prompts
- Model caching and verification
"""

import os
import sys
from pathlib import Path
from typing import Optional, Callable, List

import click


def get_models_dir() -> Path:
    """Get the models directory path.
    
    Priority:
    1. HFTOOL_MODELS_DIR environment variable
    2. ~/.hftool/models/
    
    Returns:
        Path to models directory
    """
    env_dir = os.environ.get("HFTOOL_MODELS_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    
    return Path.home() / ".hftool" / "models"


def get_model_path(repo_id: str) -> Path:
    """Get the local path for a model.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., "openai/whisper-large-v3")
    
    Returns:
        Path where the model should be stored
    """
    models_dir = get_models_dir()
    # Convert repo_id to path-safe format
    safe_name = repo_id.replace("/", "--")
    return models_dir / safe_name


def is_model_downloaded(repo_id: str) -> bool:
    """Check if a model has been downloaded.
    
    Args:
        repo_id: HuggingFace repository ID
    
    Returns:
        True if model exists locally
    """
    model_path = get_model_path(repo_id)
    
    if not model_path.exists():
        return False
    
    # Check for common model files
    config_files = ["config.json", "model_index.json", "tokenizer_config.json"]
    for config_file in config_files:
        if (model_path / config_file).exists():
            return True
    
    # Check if directory has any files
    try:
        return any(model_path.iterdir())
    except OSError:
        return False


def get_download_status(repo_id: str) -> str:
    """Get download status string for display.
    
    Args:
        repo_id: HuggingFace repository ID
    
    Returns:
        Status string: "downloaded", "partial", or "not downloaded"
    """
    model_path = get_model_path(repo_id)
    
    if not model_path.exists():
        return "not downloaded"
    
    # Check for config file (indicates complete download)
    config_files = ["config.json", "model_index.json"]
    for config_file in config_files:
        if (model_path / config_file).exists():
            return "downloaded"
    
    # Directory exists but may be incomplete
    try:
        if any(model_path.iterdir()):
            return "partial"
    except OSError:
        pass
    
    return "not downloaded"


def download_model(
    repo_id: str,
    revision: Optional[str] = None,
    ignore_patterns: Optional[List[str]] = None,
    force: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Download a model from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repository ID
        revision: Specific revision/commit to download
        ignore_patterns: File patterns to exclude from download
        force: Re-download even if already exists
        progress_callback: Optional callback for progress updates (current, total)
    
    Returns:
        Path to downloaded model
    
    Raises:
        ImportError: If huggingface_hub is not installed
        Exception: If download fails
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for downloading models. "
            "Install with: pip install huggingface_hub"
        )
    
    model_path = get_model_path(repo_id)
    
    # Check if already downloaded
    if not force and is_model_downloaded(repo_id):
        return model_path
    
    # Create models directory
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set up ignore patterns
    patterns = ignore_patterns or []
    # Only ignore root-level documentation files
    # DO NOT ignore *.txt - tokenizers need merges.txt
    # DO NOT ignore *.safetensors.index.json - sharded models need these
    default_ignores = [
        "README.md",
        "LICENSE*",
        ".gitattributes",
    ]
    patterns.extend(default_ignores)
    
    # Download with huggingface_hub
    downloaded_path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(model_path),
        ignore_patterns=patterns if patterns else None,
    )
    
    return Path(downloaded_path)


def check_dependency_satisfied(dep: str) -> bool:
    """Check if a dependency requirement is already satisfied.
    
    Args:
        dep: Dependency spec like "diffusers>=0.36.0"
    
    Returns:
        True if requirement is satisfied
    """
    try:
        from packaging.requirements import Requirement
        from packaging.version import Version
        import importlib.metadata
        
        req = Requirement(dep)
        try:
            installed_version = Version(importlib.metadata.version(req.name))
            return installed_version in req.specifier
        except importlib.metadata.PackageNotFoundError:
            return False
    except ImportError:
        # packaging not available, assume not satisfied to be safe
        return False


def install_pip_dependencies(dependencies: List[str], use_pipx: bool = True, force: bool = False) -> bool:
    """Install or upgrade pip dependencies for a model.
    
    Args:
        dependencies: List of pip package specs to install (e.g., "diffusers>=0.36.0")
        use_pipx: If True, try to inject into pipx venv first
        force: If True, install even if already satisfied
    
    Returns:
        True if installation succeeded
    """
    import subprocess
    import shutil
    
    if not dependencies:
        return True
    
    # Filter out already satisfied dependencies
    if not force:
        unsatisfied = [dep for dep in dependencies if not check_dependency_satisfied(dep)]
        if not unsatisfied:
            return True
        dependencies = unsatisfied
    
    click.echo(f"Installing/upgrading dependencies: {', '.join(dependencies)}")
    
    # Try pipx inject first (if hftool was installed via pipx)
    if use_pipx and shutil.which("pipx"):
        try:
            # Check if hftool is installed via pipx
            result = subprocess.run(
                ["pipx", "list", "--short"],
                capture_output=True,
                text=True,
            )
            if "hftool" in result.stdout:
                # Use pipx runpip to install into hftool's venv
                for dep in dependencies:
                    click.echo(f"  Upgrading {dep} via pipx...")
                    install_cmd = ["pipx", "runpip", "hftool", "install", "--upgrade", dep]
                    # flash-attn needs special handling
                    if "flash-attn" in dep:
                        install_cmd.extend(["--no-build-isolation"])
                    
                    proc = subprocess.run(install_cmd, capture_output=True, text=True)
                    if proc.returncode != 0:
                        click.echo(f"    Warning: Failed to install {dep}: {proc.stderr}", err=True)
                    else:
                        click.echo(f"    Installed {dep}")
                return True
        except Exception as e:
            click.echo(f"  pipx injection failed: {e}, falling back to pip", err=True)
    
    # Fall back to regular pip via subprocess (more reliable than pip.main)
    try:
        import sys
        for dep in dependencies:
            click.echo(f"  Upgrading {dep} via pip...")
            install_cmd = [sys.executable, "-m", "pip", "install", "--upgrade", dep]
            if "flash-attn" in dep:
                install_cmd.append("--no-build-isolation")
            proc = subprocess.run(install_cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                click.echo(f"    Warning: Failed to install {dep}: {proc.stderr}", err=True)
            else:
                click.echo(f"    Installed {dep}")
        return True
    except Exception as e:
        click.echo(f"  pip installation failed: {e}", err=True)
        click.echo(f"  Please install manually: pip install --upgrade {' '.join(dependencies)}")
        return False


def download_model_with_progress(
    repo_id: str,
    size_gb: float,
    revision: Optional[str] = None,
    ignore_patterns: Optional[List[str]] = None,
    force: bool = False,
    pip_dependencies: Optional[List[str]] = None,
) -> Path:
    """Download a model with progress display.
    
    Args:
        repo_id: HuggingFace repository ID
        size_gb: Approximate size in GB (for display)
        revision: Specific revision/commit to download
        ignore_patterns: File patterns to exclude
        force: Re-download even if already exists
        pip_dependencies: Additional pip packages to install
    
    Returns:
        Path to downloaded model
    """
    # Install pip dependencies first (before download)
    if pip_dependencies:
        install_pip_dependencies(pip_dependencies)
    
    # Check if already downloaded
    if not force and is_model_downloaded(repo_id):
        click.echo(f"Model already downloaded: {repo_id}")
        return get_model_path(repo_id)
    
    model_path = get_model_path(repo_id)
    
    click.echo(f"Downloading: {repo_id}")
    click.echo(f"Size: ~{size_gb:.1f} GB")
    click.echo(f"Location: {model_path}")
    click.echo("")
    
    try:
        # Try to use rich for better progress display
        try:
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn, TimeRemainingColumn
            from rich.console import Console
            
            console = Console()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"Downloading {repo_id}...", total=None)
                
                path = download_model(
                    repo_id=repo_id,
                    revision=revision,
                    ignore_patterns=ignore_patterns,
                    force=force,
                )
                
                progress.update(task, completed=True)
            
            click.echo(f"\nDownload complete: {path}")
            return path
            
        except ImportError:
            # Fall back to simple progress
            click.echo("Downloading (this may take a while)...")
            path = download_model(
                repo_id=repo_id,
                revision=revision,
                ignore_patterns=ignore_patterns,
                force=force,
            )
            click.echo(f"Download complete: {path}")
            return path
            
    except KeyboardInterrupt:
        click.echo("\nDownload cancelled.", err=True)
        raise
    except Exception as e:
        click.echo(f"\nDownload failed: {e}", err=True)
        raise


def prompt_download(
    repo_id: str,
    size_gb: float,
    task_name: str,
    model_name: str,
    pip_dependencies: Optional[List[str]] = None,
) -> Optional[Path]:
    """Prompt user to download a model interactively.
    
    Args:
        repo_id: HuggingFace repository ID
        size_gb: Approximate size in GB
        task_name: Task name for display
        model_name: Model name for display
        pip_dependencies: Additional pip packages to install
    
    Returns:
        Path to downloaded model, or None if user cancelled
    """
    click.echo("")
    click.echo("=" * 60)
    click.echo(f"Model not found: {model_name}")
    click.echo("=" * 60)
    click.echo("")
    click.echo(f"  Task:     {task_name}")
    click.echo(f"  Model:    {model_name}")
    click.echo(f"  Repo:     {repo_id}")
    click.echo(f"  Size:     ~{size_gb:.1f} GB")
    click.echo(f"  Location: {get_model_path(repo_id)}")
    if pip_dependencies:
        click.echo(f"  Requires: {', '.join(pip_dependencies)}")
    click.echo("")
    
    try:
        if click.confirm("Download this model now?", default=True):
            return download_model_with_progress(
                repo_id=repo_id,
                size_gb=size_gb,
                pip_dependencies=pip_dependencies,
            )
        else:
            click.echo("")
            click.echo("Download cancelled. To download manually, run:")
            click.echo(f"  hftool download -t {task_name}")
            click.echo("")
            click.echo("Or set HFTOOL_MODELS_DIR to use a custom location.")
            return None
    except KeyboardInterrupt:
        click.echo("\n\nDownload cancelled.")
        return None


def ensure_model_available(
    repo_id: str,
    size_gb: float,
    task_name: str,
    model_name: str,
    auto_download: bool = False,
    pip_dependencies: Optional[List[str]] = None,
) -> Path:
    """Ensure a model is available, prompting to download if needed.
    
    Args:
        repo_id: HuggingFace repository ID
        size_gb: Approximate size in GB
        task_name: Task name for display
        model_name: Model name for display
        auto_download: If True, download without prompting
        pip_dependencies: Additional pip packages to install
    
    Returns:
        Path to model
    
    Raises:
        SystemExit: If model not available and user cancelled download
    """
    # Check if already downloaded
    if is_model_downloaded(repo_id):
        # Still need to install pip dependencies even if model is downloaded
        if pip_dependencies:
            install_pip_dependencies(pip_dependencies)
        return get_model_path(repo_id)
    
    # Check environment variable for auto-download behavior
    auto_env = os.environ.get("HFTOOL_AUTO_DOWNLOAD", "").lower()
    if auto_env in ("1", "true", "yes"):
        auto_download = True
    elif auto_env in ("0", "false", "no"):
        auto_download = False
    
    if auto_download:
        return download_model_with_progress(
            repo_id=repo_id,
            size_gb=size_gb,
            pip_dependencies=pip_dependencies,
        )
    
    # Interactive prompt
    result = prompt_download(
        repo_id=repo_id,
        size_gb=size_gb,
        task_name=task_name,
        model_name=model_name,
        pip_dependencies=pip_dependencies,
    )
    
    if result is None:
        # User cancelled - provide helpful instructions
        click.echo("")
        click.echo("To use this task, you need to download the model first.")
        click.echo("")
        click.echo("Options:")
        click.echo(f"  1. Run: hftool download -t {task_name}")
        click.echo(f"  2. Set HFTOOL_AUTO_DOWNLOAD=1 to auto-download")
        click.echo(f"  3. Use a custom model path with -m /path/to/model")
        click.echo("")
        sys.exit(1)
    
    return result


def delete_model(repo_id: str) -> bool:
    """Delete a downloaded model.
    
    Args:
        repo_id: HuggingFace repository ID
    
    Returns:
        True if model was deleted, False if not found
    """
    import shutil
    
    model_path = get_model_path(repo_id)
    
    if not model_path.exists():
        return False
    
    shutil.rmtree(model_path)
    return True


def list_downloaded_models() -> List[str]:
    """List all downloaded models.
    
    Returns:
        List of repo_ids for downloaded models
    """
    models_dir = get_models_dir()
    
    if not models_dir.exists():
        return []
    
    models = []
    for path in models_dir.iterdir():
        if path.is_dir():
            # Convert path-safe name back to repo_id
            repo_id = path.name.replace("--", "/")
            if is_model_downloaded(repo_id):
                models.append(repo_id)
    
    return sorted(models)


def get_models_disk_usage() -> dict:
    """Get disk usage information for downloaded models.
    
    Returns:
        Dictionary with 'total_bytes', 'total_str', and 'models' (list of dicts)
    """
    models_dir = get_models_dir()
    
    if not models_dir.exists():
        return {"total_bytes": 0, "total_str": "0 B", "models": []}
    
    def get_dir_size(path: Path) -> int:
        total = 0
        try:
            for entry in path.rglob("*"):
                if entry.is_file():
                    total += entry.stat().st_size
        except OSError:
            pass
        return total
    
    def format_size(size_bytes: int) -> str:
        size = float(size_bytes)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size = size / 1024
        return f"{size:.1f} PB"
    
    models = []
    total_bytes = 0
    
    for path in models_dir.iterdir():
        if path.is_dir():
            repo_id = path.name.replace("--", "/")
            size = get_dir_size(path)
            total_bytes += size
            models.append({
                "repo_id": repo_id,
                "path": str(path),
                "size_bytes": size,
                "size_str": format_size(size),
            })
    
    return {
        "total_bytes": total_bytes,
        "total_str": format_size(total_bytes),
        "models": sorted(models, key=lambda x: x["size_bytes"], reverse=True),
    }
