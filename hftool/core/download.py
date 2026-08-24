"""Model download manager for hftool.

Handles downloading models from HuggingFace Hub with:
- Configurable storage location (HFTOOL_MODELS_DIR or ~/.hftool/models/)
- Progress bar support
- Interactive prompts
- Model caching and verification
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Callable, List, Dict

import click


_REVISION_MARKER = ".hftool-revision"


def _revision_matches(model_path: Path, revision: Optional[str]) -> bool:
    """Return whether a local model directory matches a pinned revision."""
    if revision is None:
        return True
    try:
        return (model_path / _REVISION_MARKER).read_text(encoding="utf-8").strip() == revision
    except OSError:
        return False


def _record_revision(model_path: Path, revision: Optional[str]) -> None:
    """Record the immutable catalog revision after a successful download."""
    if revision is not None:
        (model_path / _REVISION_MARKER).write_text(f"{revision}\n", encoding="utf-8")


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


def is_model_downloaded(repo_id: str, revision: Optional[str] = None) -> bool:
    """Check if a model has been downloaded completely.

    Args:
        repo_id: HuggingFace repository ID

    Returns:
        True if model exists locally with required config files
    """
    model_path = get_model_path(repo_id)

    if not model_path.exists():
        return False
    if not _revision_matches(model_path, revision):
        return False

    # Check for common model files - at least one must exist for a valid download
    config_files = ["config.json", "model_index.json", "tokenizer_config.json"]
    for config_file in config_files:
        if (model_path / config_file).exists():
            return True

    # Directory exists but no config files = incomplete download
    # Return False so it will be re-downloaded
    return False


def get_download_status(repo_id: str, revision: Optional[str] = None) -> str:
    """Get download status string for display.
    
    Args:
        repo_id: HuggingFace repository ID
    
    Returns:
        Status string: "downloaded", "partial", or "not downloaded"
    """
    model_path = get_model_path(repo_id)
    
    if not model_path.exists():
        return "not downloaded"
    if not _revision_matches(model_path, revision):
        return "partial"
    
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


def get_partial_downloads() -> List[Dict[str, str]]:
    """Get list of partially downloaded models.
    
    Returns:
        List of dicts with repo_id and path for partial downloads
    """
    models_dir = get_models_dir()
    
    if not models_dir.exists():
        return []
    
    partial = []
    for path in models_dir.iterdir():
        if path.is_dir():
            repo_id = path.name.replace("--", "/")
            status = get_download_status(repo_id)
            
            if status == "partial":
                partial.append({
                    "repo_id": repo_id,
                    "path": str(path),
                })
    
    return partial


def get_hf_token() -> Optional[str]:
    """Get HuggingFace token for authentication.

    Checks in order:
    1. HF_TOKEN environment variable
    2. HUGGINGFACE_TOKEN environment variable
    3. huggingface_hub cached token (from `huggingface-cli login`)

    Returns:
        Token string or None if not found
    """
    # Check environment variables first
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        return token

    # Check huggingface_hub cached token
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            return token
    except Exception:
        pass

    return None


def download_model(
    repo_id: str,
    revision: Optional[str] = None,
    ignore_patterns: Optional[List[str]] = None,
    force: bool = False,
    resume: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Download a model from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID
        revision: Specific revision/commit to download
        ignore_patterns: File patterns to exclude from download
        force: Re-download even if already exists
        resume: Resume partial downloads (default: True)
        progress_callback: Optional callback for progress updates (current, total)

    Returns:
        Path to downloaded model

    Raises:
        ImportError: If huggingface_hub is not installed
        Exception: If download fails

    Authentication:
        For gated models (requiring license acceptance), set HF_TOKEN environment
        variable or run `huggingface-cli login` first.
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
    if not force and is_model_downloaded(repo_id, revision):
        return model_path

    # Create models directory
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Set up ignore patterns
    patterns = list(ignore_patterns or [])
    # Only ignore root-level documentation files
    # DO NOT ignore *.txt - tokenizers need merges.txt
    # DO NOT ignore *.safetensors.index.json - sharded models need these
    default_ignores = [
        "README.md",
        "LICENSE*",
        ".gitattributes",
    ]
    patterns.extend(default_ignores)

    # Get authentication token for gated models
    token = get_hf_token()

    try:
        # Download with huggingface_hub
        # resume_download is enabled by default in huggingface_hub
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(model_path),
            ignore_patterns=patterns if patterns else None,
            resume_download=resume,  # Enable resume capability
            token=token,  # Pass token for gated models
        )
    except Exception as e:
        error_str = str(e).lower()
        # Check for gated model access errors
        if any(x in error_str for x in ["gated", "access", "401", "403", "forbidden", "unauthorized"]):
            raise RuntimeError(
                f"Access denied to gated model: {repo_id}\n\n"
                f"This model requires accepting the license agreement and authentication.\n\n"
                f"To fix this:\n"
                f"  1. Visit https://huggingface.co/{repo_id} and accept the license terms\n"
                f"  2. Create an access token at https://huggingface.co/settings/tokens\n"
                f"  3. Run: huggingface-cli login\n"
                f"     Or set: export HF_TOKEN=your_token_here\n"
            ) from e
        raise

    resolved_path = Path(downloaded_path)
    _record_revision(resolved_path, revision)
    return resolved_path


def check_dependency_satisfied(dep: str) -> bool:
    """Check if a dependency requirement is already satisfied.

    Args:
        dep: Dependency spec like "diffusers>=0.36.0" or git URL

    Returns:
        True if requirement is satisfied
    """
    import importlib.metadata

    # Git URLs - extract package name and check if installed
    # We can't check exact version but can check if package exists
    if dep.startswith("git+") or dep.startswith("https://"):
        # Extract package name from git URL
        # e.g., "git+https://github.com/huggingface/diffusers" -> "diffusers"
        url = dep.replace("git+", "").rstrip("/")
        package_name = url.split("/")[-1].replace(".git", "")

        try:
            # Check if package is installed
            importlib.metadata.version(package_name)
            # Package is installed - assume git version is satisfied
            # (User can force reinstall with --force if needed)
            return True
        except importlib.metadata.PackageNotFoundError:
            return False

    try:
        from packaging.requirements import Requirement
        from packaging.version import Version

        req = Requirement(dep)
        try:
            installed_version = Version(importlib.metadata.version(req.name))
            # prereleases=True so dev/pre-release builds (e.g. diffusers
            # "0.38.0.dev0" installed from git main in the Docker image)
            # satisfy a ">=" spec. Without it, packaging excludes
            # pre-releases by default and we'd wrongly attempt a reinstall.
            return req.specifier.contains(installed_version, prereleases=True)
        except importlib.metadata.PackageNotFoundError:
            return False
    except (ImportError, Exception):
        # packaging not available or parse error, assume not satisfied to be safe
        return False


def _clear_dependency_cache():
    """Clear the dependency check cache after installing new packages."""
    try:
        from hftool.utils.deps import _DEPENDENCY_CACHE
        _DEPENDENCY_CACHE.clear()
    except (ImportError, AttributeError):
        pass


def _running_in_pipx_hftool_env() -> bool:
    """Return True when current Python belongs to pipx-managed hftool venv."""
    exe_parts = [part.lower() for part in Path(sys.executable).resolve().parts]
    prefix_parts = [part.lower() for part in Path(sys.prefix).resolve().parts]

    def _looks_like_pipx_hftool(parts: List[str]) -> bool:
        return "pipx" in parts and "venvs" in parts and "hftool" in parts

    return _looks_like_pipx_hftool(exe_parts) or _looks_like_pipx_hftool(prefix_parts)


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
    
    # Try pipx inject first only when this process is running inside pipx's hftool venv.
    # Otherwise pipx may install into a different environment than the active interpreter.
    should_use_pipx = use_pipx and shutil.which("pipx") and _running_in_pipx_hftool_env()

    if should_use_pipx:
        try:
            failed_deps = []
            for dep in dependencies:
                click.echo(f"  Upgrading {dep} via pipx...")
                install_cmd = ["pipx", "runpip", "hftool", "install", "--upgrade", dep]
                # flash-attn needs special handling
                if "flash-attn" in dep:
                    install_cmd.extend(["--no-build-isolation"])

                proc = subprocess.run(install_cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    failed_deps.append(dep)
                    click.echo(f"    Warning: Failed to install {dep}: {proc.stderr}", err=True)
                else:
                    click.echo(f"    Installed {dep}")

            if not failed_deps:
                _clear_dependency_cache()
                return True

            click.echo(
                f"  pipx install failed for: {', '.join(failed_deps)}; falling back to pip",
                err=True,
            )
            dependencies = failed_deps
        except Exception as e:
            click.echo(f"  pipx injection failed: {e}, falling back to pip", err=True)
    elif use_pipx and shutil.which("pipx") and not _running_in_pipx_hftool_env():
        click.echo("  pipx detected but current runtime is not pipx hftool; using pip", err=True)

    # Fall back to regular pip via subprocess (more reliable than pip.main)
    constraints_file = _write_torch_constraints()
    try:
        failed_deps = []
        for dep in dependencies:
            click.echo(f"  Upgrading {dep} via pip...")
            install_cmd = [sys.executable, "-m", "pip", "install", "--upgrade", dep]
            if constraints_file:
                install_cmd.extend(["-c", constraints_file])
            if "flash-attn" in dep:
                install_cmd.append("--no-build-isolation")
            proc = subprocess.run(install_cmd, capture_output=True, text=True)

            # Torch constraint conflict — the package depends on a
            # different torch version.  Install with --no-deps to avoid
            # overwriting ROCm PyTorch, then install its non-torch
            # sub-dependencies separately.
            if (
                proc.returncode != 0
                and constraints_file
                and "conflicting" in (proc.stderr or "").lower()
            ):
                click.echo(
                    f"    Torch version conflict detected — installing {dep} "
                    f"without replacing PyTorch...",
                    err=True,
                )
                proc = _install_with_torch_protection(dep, constraints_file)

            # Debian/Ubuntu externally-managed Python (PEP 668): retry explicitly.
            if (
                proc.returncode != 0
                and "externally-managed-environment" in (proc.stderr or "").lower()
            ):
                click.echo(
                    "    Externally managed Python detected, retrying with --break-system-packages...",
                    err=True,
                )
                retry_cmd = install_cmd + ["--break-system-packages"]
                proc = subprocess.run(retry_cmd, capture_output=True, text=True)

            if proc.returncode != 0:
                failed_deps.append(dep)
                click.echo(f"    Warning: Failed to install {dep}: {proc.stderr}", err=True)
            else:
                click.echo(f"    Installed {dep}")

        if failed_deps:
            click.echo(
                f"  Failed to install required dependencies: {', '.join(failed_deps)}",
                err=True,
            )
            click.echo(
                f"  Please install manually: pip install --upgrade {' '.join(failed_deps)}",
                err=True,
            )
            return False

        # Clear dependency cache so check_dependency re-checks after install
        _clear_dependency_cache()
        return True
    except Exception as e:
        click.echo(f"  pip installation failed: {e}", err=True)
        click.echo(f"  Please install manually: pip install --upgrade {' '.join(dependencies)}")
        return False
    finally:
        if constraints_file:
            try:
                os.remove(constraints_file)
            except OSError:
                pass


def _write_torch_constraints() -> Optional[str]:
    """Create a temporary pip constraints file that pins the current PyTorch.

    When installing model dependencies at runtime (e.g., ``chatterbox-tts``),
    pip may try to replace the existing ROCm PyTorch with a CUDA build from
    PyPI.  A constraints file tells pip "these packages must stay at their
    current versions", preventing silent overwrites.

    Returns:
        Path to the temporary constraints file, or None if torch isn't
        installed (nothing to protect).
    """
    import importlib.metadata
    import tempfile

    pins: list[str] = []
    for pkg in ("torch", "torchaudio", "torchvision"):
        try:
            version = importlib.metadata.version(pkg)
            pins.append(f"{pkg}=={version}")
        except importlib.metadata.PackageNotFoundError:
            pass

    if not pins:
        return None

    fd, path = tempfile.mkstemp(prefix="hftool_torch_constraints_", suffix=".txt")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(pins) + "\n")
    return path


def _install_with_torch_protection(dep: str, constraints_file: str) -> subprocess.CompletedProcess:
    """Install a package without letting it replace PyTorch.

    Strategy:
    1. Install the main package with ``--no-deps``
    2. Read its declared dependencies from package metadata
    3. Install non-torch dependencies (with constraints to protect PyTorch
       transitively as well)

    Args:
        dep: Package spec to install (e.g., "chatterbox-tts")
        constraints_file: Path to torch constraints file

    Returns:
        CompletedProcess from the final install step
    """
    import importlib.metadata

    # Step 1: install main package without any dependencies
    nodeps_cmd = [sys.executable, "-m", "pip", "install", "--no-deps", dep]
    proc = subprocess.run(nodeps_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return proc

    # Step 2: discover its declared sub-dependencies
    pkg_name = dep.split("[")[0].split(">=")[0].split("==")[0].split("<")[0].strip()
    try:
        requires = importlib.metadata.requires(pkg_name) or []
    except importlib.metadata.PackageNotFoundError:
        # Package installed but metadata not found — best effort
        return proc

    # Step 3: filter out torch ecosystem packages
    _torch_pkgs = frozenset({
        "torch", "torchaudio", "torchvision", "triton",
        "nvidia-cublas-cu12", "nvidia-cuda-cupti-cu12",
        "nvidia-cuda-nvrtc-cu12", "nvidia-cuda-runtime-cu12",
        "nvidia-cudnn-cu12", "nvidia-cufft-cu12", "nvidia-curand-cu12",
        "nvidia-cusolver-cu12", "nvidia-cusparse-cu12",
        "nvidia-cusparselt-cu12", "nvidia-nccl-cu12",
        "nvidia-nvjitlink-cu12", "nvidia-nvtx-cu12",
    })

    safe_deps: list[str] = []
    for req_str in requires:
        # Skip extras-conditional deps (e.g., "foo ; extra == 'dev'")
        if "extra ==" in req_str:
            continue
        # Extract bare package name
        name = req_str.split(";")[0].split("[")[0].split(">=")[0].split("==")[0].split("<")[0].split("!")[0].strip()
        if name.lower() not in _torch_pkgs and not name.lower().startswith("nvidia-"):
            safe_deps.append(req_str.split(";")[0].strip())

    if not safe_deps:
        return proc

    click.echo(f"    Installing {len(safe_deps)} sub-dependencies (torch protected)...")

    _install_deps_recursive(safe_deps, constraints_file, _torch_pkgs, max_rounds=3)

    return proc  # Return success from the main --no-deps install


def _install_deps_recursive(
    deps: list[str],
    constraints_file: str,
    protected_pkgs: frozenset,
    max_rounds: int = 3,
) -> None:
    """Install dependencies, chasing transitive deps up to *max_rounds*.

    For each dep:
    1. Try ``pip install dep -c constraints`` (fast, respects torch pins).
    2. On conflict, fall back to ``pip install --no-deps dep`` and queue
       its own sub-deps for the next round.

    Converges in 2-3 rounds for most packages.
    """
    import importlib.metadata

    pending = list(deps)

    for round_num in range(max_rounds):
        if not pending:
            break
        next_round: list[str] = []

        for subdep in pending:
            name = (
                subdep.split(";")[0].split("[")[0].split(">=")[0]
                .split("==")[0].split("<")[0].split("!")[0].strip().lower()
            )
            if name in protected_pkgs or name.startswith("nvidia-"):
                continue

            # Already installed?
            try:
                importlib.metadata.version(name)
                continue
            except importlib.metadata.PackageNotFoundError:
                pass

            # Try with constraints first
            sub_cmd = [
                sys.executable, "-m", "pip", "install",
                "-c", constraints_file,
                subdep.split(";")[0].strip(),
            ]
            sub_proc = subprocess.run(sub_cmd, capture_output=True, text=True)

            if sub_proc.returncode != 0:
                # Conflict — install bare, queue its deps
                retry_cmd = [
                    sys.executable, "-m", "pip", "install", "--no-deps",
                    subdep.split(";")[0].strip(),
                ]
                subprocess.run(retry_cmd, capture_output=True, text=True)

                # Queue this package's own deps for next round
                try:
                    child_reqs = importlib.metadata.requires(name) or []
                    for cr in child_reqs:
                        if "extra ==" not in cr:
                            next_round.append(cr)
                except importlib.metadata.PackageNotFoundError:
                    pass

        pending = next_round


def get_model_file_path(repo_id: str, filename: str) -> Path:
    """Return a safe local path for one repository file."""
    relative_path = Path(filename)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Unsafe model filename: {filename}")
    return get_model_path(repo_id) / relative_path


def get_model_file_status(
    repo_id: str,
    filename: str,
    revision: Optional[str] = None,
) -> str:
    """Return download status for one exact adapter or profile file."""
    file_path = get_model_file_path(repo_id, filename)
    if (
        file_path.is_file()
        and file_path.stat().st_size > 0
        and _revision_matches(get_model_path(repo_id), revision)
    ):
        return "downloaded"
    if file_path.exists() or file_path.parent.exists():
        return "partial"
    return "not downloaded"


def get_model_download_status(model_info) -> str:
    """Return one status for a base model and any exact profile adapter."""
    base_status = get_download_status(model_info.repo_id, model_info.revision)
    if model_info.adapter is None:
        return base_status

    adapter_status = get_model_file_status(
        model_info.adapter.repo_id,
        model_info.adapter.weight_name,
        model_info.adapter.revision,
    )
    if base_status == "downloaded" and adapter_status == "downloaded":
        return "downloaded"
    if base_status == "not downloaded" and adapter_status == "not downloaded":
        return "not downloaded"
    return "partial"


def download_model_file_with_progress(
    repo_id: str,
    filename: str,
    size_gb: float,
    revision: Optional[str] = None,
    force: bool = False,
) -> Path:
    """Download one exact repository file without fetching sibling checkpoints."""
    destination = get_model_file_path(repo_id, filename)
    if not force and get_model_file_status(repo_id, filename, revision) == "downloaded":
        click.echo(f"Model file already downloaded: {repo_id}/{filename}")
        return destination

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImportError(
            "huggingface_hub is required for downloading model adapters. "
            "Install with: pip install huggingface_hub"
        ) from error

    destination.parent.mkdir(parents=True, exist_ok=True)
    click.echo(f"Downloading profile file: {repo_id}/{filename}")
    click.echo(f"Size: ~{size_gb:.2f} GB")
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_dir=str(get_model_path(repo_id)),
            force_download=force,
            token=get_hf_token(),
        )
    except Exception as error:
        error_text = str(error).lower()
        if any(code in error_text for code in ("401", "403", "gated", "forbidden")):
            raise RuntimeError(
                f"Access denied to profile file {repo_id}/{filename}. "
                "Accept its license and authenticate with huggingface-cli login."
            ) from error
        raise
    resolved_path = Path(downloaded_path)
    _record_revision(get_model_path(repo_id), revision)
    return resolved_path


def ensure_model_file_available(
    repo_id: str,
    filename: str,
    size_gb: float,
    task_name: str,
    model_name: str,
    auto_download: bool = False,
    revision: Optional[str] = None,
) -> Path:
    """Ensure one exact profile file exists, prompting before a large download."""
    if get_model_file_status(repo_id, filename, revision) == "downloaded":
        return get_model_file_path(repo_id, filename)

    auto_env = os.environ.get("HFTOOL_AUTO_DOWNLOAD", "").lower()
    if auto_env in ("1", "true", "yes"):
        auto_download = True
    elif auto_env in ("0", "false", "no"):
        auto_download = False

    if not auto_download:
        click.echo("")
        click.echo(f"Profile adapter not found: {model_name}")
        click.echo(f"  Repo: {repo_id}")
        click.echo(f"  File: {filename}")
        click.echo(f"  Size: ~{size_gb:.2f} GB")
        if not click.confirm("Download this adapter now?", default=True):
            raise RuntimeError(
                f"Adapter download cancelled. Run 'hftool download -t {task_name} "
                f"-m {model_name}' before generation."
            )

    return download_model_file_with_progress(
        repo_id=repo_id,
        filename=filename,
        size_gb=size_gb,
        revision=revision,
    )


def download_model_with_progress(
    repo_id: str,
    size_gb: float,
    revision: Optional[str] = None,
    ignore_patterns: Optional[List[str]] = None,
    force: bool = False,
    resume: bool = True,
    pip_dependencies: Optional[List[str]] = None,
) -> Path:
    """Download a model with progress display.
    
    Args:
        repo_id: HuggingFace repository ID
        size_gb: Approximate size in GB (for display)
        revision: Specific revision/commit to download
        ignore_patterns: File patterns to exclude
        force: Re-download even if already exists
        resume: Resume partial downloads (default: True)
        pip_dependencies: Additional pip packages to install
    
    Returns:
        Path to downloaded model
    """
    # Install pip dependencies first (before download)
    if pip_dependencies:
        if not install_pip_dependencies(pip_dependencies):
            raise RuntimeError(
                f"Failed to install required dependencies for {repo_id}: {', '.join(pip_dependencies)}"
            )
    
    # Check if already downloaded
    if not force and is_model_downloaded(repo_id, revision):
        click.echo(f"Model already downloaded: {repo_id}")
        return get_model_path(repo_id)
    
    model_path = get_model_path(repo_id)
    
    # Check if resuming partial download
    status = get_download_status(repo_id, revision)
    if status == "partial" and resume:
        click.echo(f"Resuming download: {repo_id}")
    else:
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
                    resume=resume,
                )
                
                progress.update(task, completed=True)
            
            click.echo(f"\nDownload complete: {path}")
            return path
            
        except ImportError:
            # Fall back to simple progress
            if status == "partial" and resume:
                click.echo("Resuming download (this may take a while)...")
            else:
                click.echo("Downloading (this may take a while)...")
            
            path = download_model(
                repo_id=repo_id,
                revision=revision,
                ignore_patterns=ignore_patterns,
                force=force,
                resume=resume,
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
    gated: bool = False,
    revision: Optional[str] = None,
    ignore_patterns: Optional[List[str]] = None,
) -> Optional[Path]:
    """Prompt user to download a model interactively.

    Args:
        repo_id: HuggingFace repository ID
        size_gb: Approximate size in GB
        task_name: Task name for display
        model_name: Model name for display
        pip_dependencies: Additional pip packages to install
        gated: Whether the model requires license acceptance and HF token

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

    # Show gated model warning and check for token
    if gated:
        click.echo("")
        click.echo(click.style("  ⚠ GATED MODEL - Requires authentication", fg="yellow"))

        token = get_hf_token()
        if token:
            click.echo(click.style("  ✓ HuggingFace token found", fg="green"))
        else:
            click.echo(click.style("  ✗ No HuggingFace token found", fg="red"))
            click.echo("")
            click.echo("  This model requires:")
            click.echo(f"    1. Accept license at: https://huggingface.co/{repo_id}")
            click.echo("    2. Login with: huggingface-cli login")
            click.echo("       Or set: export HF_TOKEN=your_token_here")
            click.echo("")

            if not click.confirm("Continue anyway?", default=False):
                click.echo("")
                click.echo("To authenticate:")
                click.echo("  1. Create a token at https://huggingface.co/settings/tokens")
                click.echo("  2. Run: huggingface-cli login")
                click.echo(f"  3. Then retry: hftool download -t {task_name}")
                return None

    click.echo("")

    try:
        if click.confirm("Download this model now?", default=True):
            return download_model_with_progress(
                repo_id=repo_id,
                size_gb=size_gb,
                revision=revision,
                ignore_patterns=ignore_patterns,
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
    gated: bool = False,
    revision: Optional[str] = None,
    ignore_patterns: Optional[List[str]] = None,
) -> Path:
    """Ensure a model is available, prompting to download if needed.

    Args:
        repo_id: HuggingFace repository ID
        size_gb: Approximate size in GB
        task_name: Task name for display
        model_name: Model name for display
        auto_download: If True, download without prompting
        pip_dependencies: Additional pip packages to install
        gated: Whether the model requires license acceptance and HF token

    Returns:
        Path to model

    Raises:
        SystemExit: If model not available and user cancelled download
    """
    # Check if already downloaded
    if is_model_downloaded(repo_id, revision):
        # Still need to install pip dependencies even if model is downloaded
        if pip_dependencies:
            if not install_pip_dependencies(pip_dependencies):
                raise RuntimeError(
                    f"Failed to install required dependencies for {model_name}: "
                    f"{', '.join(pip_dependencies)}"
                )
        return get_model_path(repo_id)

    # Check environment variable for auto-download behavior
    auto_env = os.environ.get("HFTOOL_AUTO_DOWNLOAD", "").lower()
    if auto_env in ("1", "true", "yes"):
        auto_download = True
    elif auto_env in ("0", "false", "no"):
        auto_download = False

    # For gated models with auto-download, check for token first
    if auto_download and gated:
        token = get_hf_token()
        if not token:
            click.echo(click.style("Warning: Gated model requires HuggingFace authentication.", fg="yellow"))
            click.echo("  Run: huggingface-cli login")
            click.echo("  Or set: export HF_TOKEN=your_token_here")
            click.echo(f"  Accept license at: https://huggingface.co/{repo_id}")
            click.echo("")

    if auto_download:
        return download_model_with_progress(
            repo_id=repo_id,
            size_gb=size_gb,
            revision=revision,
            ignore_patterns=ignore_patterns,
            pip_dependencies=pip_dependencies,
        )

    # Interactive prompt
    result = prompt_download(
        repo_id=repo_id,
        size_gb=size_gb,
        task_name=task_name,
        model_name=model_name,
        pip_dependencies=pip_dependencies,
        gated=gated,
        revision=revision,
        ignore_patterns=ignore_patterns,
    )

    if result is None:
        # User cancelled - provide helpful instructions
        click.echo("")
        click.echo("To use this task, you need to download the model first.")
        click.echo("")
        click.echo("Options:")
        click.echo(f"  1. Run: hftool download -t {task_name}")
        click.echo("  2. Set HFTOOL_AUTO_DOWNLOAD=1 to auto-download")
        click.echo("  3. Use a custom model path with -m /path/to/model")
        if gated:
            click.echo("")
            click.echo("For gated models, also ensure you have:")
            click.echo(f"  - Accepted the license at https://huggingface.co/{repo_id}")
            click.echo("  - Run: huggingface-cli login")
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
