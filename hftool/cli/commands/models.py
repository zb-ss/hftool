"""Models commands — models, download, status, info, clean."""

import sys
from typing import Optional

import click

from hftool.core.completion import complete_tasks, complete_models


def _download_profile_adapter(info, download_file, force: bool) -> None:
    """Download the exact declared file for an adapter-backed profile."""
    if info.adapter is None:
        return
    download_file(
        repo_id=info.adapter.repo_id,
        filename=info.adapter.weight_name,
        size_gb=info.adapter.size_gb,
        revision=info.adapter.revision,
        force=force,
    )


# =============================================================================
# MODELS COMMAND
# =============================================================================

@click.command("models")
@click.option("--task", "-t", default=None, shell_complete=complete_tasks, help="Filter by task (e.g., t2i, tts)")
@click.option("--downloaded", "-d", is_flag=True, help="Show only downloaded models")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def models_command(
    ctx: click.Context,
    task: Optional[str],
    downloaded: bool,
    as_json: bool,
):
    """List available models for tasks.

    \b
    Examples:
      hftool models                      # List all models
      hftool models -t text-to-image     # List models for T2I
      hftool models -t t2i               # Same (using alias)
      hftool models --downloaded         # Show downloaded models only
    """
    from hftool.core.models import MODEL_REGISTRY
    from hftool.core.download import get_model_download_status, get_models_dir
    from hftool.core.registry import TASK_ALIASES

    verbose = ctx.obj.get("verbose", False)

    if verbose:
        click.echo(f"Models directory: {get_models_dir()}")
        click.echo("")

    if as_json:
        import json
        output = {}

        tasks_to_show = [TASK_ALIASES.get(task, task)] if task else list(MODEL_REGISTRY.keys())

        for task_name in tasks_to_show:
            if task_name not in MODEL_REGISTRY:
                continue
            models = MODEL_REGISTRY[task_name]
            output[task_name] = {}
            for short_name, info in models.items():
                status = get_model_download_status(info)
                if downloaded and status != "downloaded":
                    continue
                output[task_name][short_name] = {
                    "repo_id": info.repo_id,
                    "name": info.name,
                    "size_gb": info.size_gb,
                    "is_default": info.is_default,
                    "status": status,
                    "catalog_status": info.status_label,
                    "profile": info.profile,
                    "use_case": info.use_case,
                    "license": info.license,
                    "commercial_use": info.commercial_use,
                    "min_vram_gb": info.min_vram_gb,
                    "recommended_vram_gb": info.recommended_vram_gb,
                    "dtype": info.dtype,
                    "pipeline_class": info.pipeline_class,
                    "aliases": info.aliases,
                    "description": info.description,
                }

        click.echo(json.dumps(output, indent=2))
        return

    # Text output
    if task:
        resolved_task = TASK_ALIASES.get(task, task)
        if resolved_task not in MODEL_REGISTRY:
            click.echo(f"Unknown task: {task}", err=True)
            click.echo(f"Available tasks: {', '.join(MODEL_REGISTRY.keys())}", err=True)
            sys.exit(1)
        tasks_to_show = [(resolved_task, MODEL_REGISTRY[resolved_task])]
    else:
        tasks_to_show = list(MODEL_REGISTRY.items())

    for task_name, models in tasks_to_show:
        click.echo(f"\n{task_name}:")
        click.echo("-" * 60)

        for short_name, info in models.items():
            status = get_model_download_status(info)

            if downloaded and status != "downloaded":
                continue

            # Status indicator
            if status == "downloaded":
                status_str = click.style("[✓]", fg="green")
            elif status == "partial":
                status_str = click.style("[~]", fg="yellow")
            else:
                status_str = click.style("[ ]", fg="white")

            # Default indicator
            default_str = click.style(f" ({info.status_label})", fg="cyan")

            click.echo(f"  {status_str} {short_name}{default_str}")
            click.echo(f"      {info.name} ({info.size_str})")
            click.echo(f"      {info.repo_id}")
            runtime = (
                f"{info.min_vram_gb:g}+ GB VRAM · {info.dtype}"
                if info.min_vram_gb is not None and info.dtype
                else "runtime varies"
            )
            click.echo(f"      {info.use_case or 'general'} · {runtime} · {info.license or 'unknown license'}")
            if info.description:
                click.echo(f"      {info.description}")

    click.echo("")
    click.echo("Legend: [✓] downloaded  [~] partial  [ ] not downloaded")
    click.echo(f"Models directory: {get_models_dir()}")


# =============================================================================
# DOWNLOAD COMMAND
# =============================================================================

@click.command("download")
@click.option("--task", "-t", default=None, shell_complete=complete_tasks, help="Download default model for task")
@click.option("--model", "-m", default=None, shell_complete=complete_models, help="Specific model to download (short name or repo_id)")
@click.option("--all", "download_all", is_flag=True, help="Download default models for all tasks")
@click.option("--force", "-f", is_flag=True, help="Re-download even if already exists")
@click.option("--resume/--no-resume", default=True, help="Resume partial downloads (default: enabled)")
@click.pass_context
def download_command(
    ctx: click.Context,
    task: Optional[str],
    model: Optional[str],
    download_all: bool,
    force: bool,
    resume: bool,
):
    """Download models from HuggingFace Hub.

    \b
    Examples:
      hftool download -t text-to-image        # Download default T2I model
      hftool download -t t2i -m sdxl          # Download specific model
      hftool download -m openai/whisper-large-v3  # Download by repo_id
      hftool download --all                   # Download all default models

    \b
    Environment Variables:
      HFTOOL_MODELS_DIR    Custom directory for model storage
                           Default: ~/.hftool/models/
    """
    from hftool.core.models import (
        MODEL_REGISTRY, get_default_model_info,
        get_model_info, find_model_by_repo_id
    )
    from hftool.core.download import (
        download_model_file_with_progress,
        download_model_with_progress,
        get_models_dir,
    )
    from hftool.core.registry import TASK_ALIASES

    click.echo(f"Models directory: {get_models_dir()}")
    click.echo("")

    if download_all:
        # Download default model for each task
        click.echo("Downloading default models for all tasks...")
        click.echo("")

        for task_name in MODEL_REGISTRY.keys():
            try:
                info = get_default_model_info(task_name)
                click.echo(f"[{task_name}]")
                download_model_with_progress(
                    repo_id=info.repo_id,
                    size_gb=info.size_gb,
                    revision=info.revision,
                    ignore_patterns=info.ignore_patterns,
                    force=force,
                    resume=resume,
                    pip_dependencies=info.pip_dependencies if info.pip_dependencies else None,
                )
                _download_profile_adapter(info, download_model_file_with_progress, force)
                click.echo("")
            except Exception as e:
                click.echo(f"  Error: {e}", err=True)
        return

    if model:
        # Download specific model
        # First check if it's a repo_id
        found = find_model_by_repo_id(model)
        if found:
            task_name, short_name, info = found
        elif task:
            # Look up by short name within task
            resolved_task = TASK_ALIASES.get(task, task)
            try:
                info = get_model_info(resolved_task, model)
            except ValueError as e:
                click.echo(f"Error: {e}", err=True)
                sys.exit(1)
        else:
            # Try to find model across all tasks
            for task_name, models in MODEL_REGISTRY.items():
                if model in models:
                    info = models[model]
                    break
            else:
                click.echo(f"Error: Model '{model}' not found.", err=True)
                click.echo("Specify a task with -t or use full repo_id.", err=True)
                sys.exit(1)

        download_model_with_progress(
            repo_id=info.repo_id,
            size_gb=info.size_gb,
            revision=info.revision,
            ignore_patterns=info.ignore_patterns,
            force=force,
            resume=resume,
            pip_dependencies=info.pip_dependencies if info.pip_dependencies else None,
        )
        _download_profile_adapter(info, download_model_file_with_progress, force)
        return

    if task:
        # Download default model for task
        resolved_task = TASK_ALIASES.get(task, task)
        try:
            info = get_default_model_info(resolved_task)
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

        click.echo(f"Downloading default model for {resolved_task}...")
        download_model_with_progress(
            repo_id=info.repo_id,
            size_gb=info.size_gb,
            revision=info.revision,
            ignore_patterns=info.ignore_patterns,
            force=force,
            resume=resume,
            pip_dependencies=info.pip_dependencies if info.pip_dependencies else None,
        )
        _download_profile_adapter(info, download_model_file_with_progress, force)
        return

    # No arguments - show help
    click.echo("Please specify what to download:")
    click.echo("  hftool download -t <task>           # Download default model for task")
    click.echo("  hftool download -t <task> -m <model> # Download specific model")
    click.echo("  hftool download --all               # Download all default models")
    click.echo("")
    click.echo("Available tasks:")
    for task_name in MODEL_REGISTRY.keys():
        click.echo(f"  {task_name}")


# =============================================================================
# STATUS COMMAND
# =============================================================================

@click.command("status")
@click.pass_context
def status_command(ctx: click.Context):
    """Show download status and disk usage.

    Displays information about:
    - Models directory location
    - Downloaded models and their sizes
    - Partial downloads (resumable)
    - Total disk usage
    """
    from hftool.core.download import get_models_dir, get_models_disk_usage, get_partial_downloads
    from hftool.core.models import find_model_by_repo_id

    models_dir = get_models_dir()
    click.echo(f"Models directory: {models_dir}")
    click.echo("")

    # Check for partial downloads
    partial_downloads = get_partial_downloads()
    if partial_downloads:
        click.echo(click.style("Partial downloads (resumable):", fg="yellow"))
        click.echo("-" * 60)
        for partial in partial_downloads:
            repo_id = partial["repo_id"]
            click.echo(f"  {repo_id}")
            click.echo(f"    Resume: hftool download -m {repo_id}")
        click.echo("")

    usage = get_models_disk_usage()

    if not usage["models"]:
        click.echo("No models downloaded yet.")
        click.echo("")
        click.echo("To download models, run:")
        click.echo("  hftool download -t <task>")
        return

    click.echo("Downloaded models:")
    click.echo("-" * 60)

    for model_info in usage["models"]:
        repo_id = model_info["repo_id"]
        size_str = model_info["size_str"]

        # Try to find model info
        found = find_model_by_repo_id(repo_id)
        if found:
            task_name, short_name, info = found
            click.echo(f"  {info.name}")
            click.echo(f"    Task: {task_name}")
            click.echo(f"    Size: {size_str}")
            click.echo(f"    Repo: {repo_id}")
        else:
            click.echo(f"  {repo_id}")
            click.echo(f"    Size: {size_str}")
            click.echo("    (Custom/unknown model)")
        click.echo("")

    click.echo("-" * 60)
    click.echo(f"Total disk usage: {usage['total_str']}")


# =============================================================================
# INFO COMMAND
# =============================================================================

@click.command("info")
@click.argument("model_name")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def info_command(ctx: click.Context, model_name: str, as_json: bool):
    """Show detailed information about a model.

    MODEL_NAME can be a short name (e.g., 'whisper-large-v3'),
    full repo ID (e.g., 'openai/whisper-large-v3'), or any model identifier.

    \b
    Examples:
      hftool info whisper-large-v3
      hftool info openai/whisper-large-v3
      hftool info z-image-turbo
      hftool info stabilityai/stable-diffusion-xl-base-1.0
    """
    from hftool.core.models import find_model_by_repo_id, MODEL_REGISTRY
    from hftool.core.download import get_model_download_status, get_model_path
    import json as json_module

    # Try to find the model
    found = find_model_by_repo_id(model_name)

    if not found:
        # Try to find by short name across all tasks
        found_by_short = None
        for task_name, models in MODEL_REGISTRY.items():
            if model_name in models:
                info = models[model_name]
                found_by_short = (task_name, model_name, info)
                break

        if found_by_short:
            found = found_by_short
        else:
            click.echo(f"Error: Model '{model_name}' not found.", err=True)
            click.echo("", err=True)
            click.echo("Use 'hftool models' to see available models.", err=True)
            sys.exit(1)

    task_name, short_name, info = found

    # Get download status
    status = get_model_download_status(info)
    is_downloaded = status == "downloaded"

    # Get local path if downloaded
    local_path = None
    if is_downloaded:
        local_path = str(get_model_path(info.repo_id))

    recommended_settings = dict(info.inference_defaults)

    # Generate HuggingFace URL
    hf_url = f"https://huggingface.co/{info.repo_id}"

    if as_json:
        # JSON output
        output = {
            "name": info.name,
            "short_name": short_name,
            "repo_id": info.repo_id,
            "revision": info.revision,
            "task": task_name,
            "type": info.model_type.value,
            "size_gb": info.size_gb,
            "size_str": info.size_str,
            "is_default": info.is_default,
            "description": info.description,
            "status": status,
            "catalog_status": info.status_label,
            "profile": info.profile,
            "license": info.license,
            "commercial_use": info.commercial_use,
            "gated": info.gated,
            "min_vram_gb": info.min_vram_gb,
            "recommended_vram_gb": info.recommended_vram_gb,
            "dtype": info.dtype,
            "pipeline_class": info.pipeline_class,
            "aliases": info.aliases,
            "is_downloaded": is_downloaded,
            "local_path": local_path,
            "recommended_settings": recommended_settings,
            "huggingface_url": hf_url,
        }

        if info.adapter:
            output["adapter"] = {
                "repo_id": info.adapter.repo_id,
                "revision": info.adapter.revision,
                "weight_name": info.adapter.weight_name,
                "size_gb": info.adapter.size_gb,
            }

        if info.pip_dependencies:
            output["pip_dependencies"] = info.pip_dependencies

        click.echo(json_module.dumps(output, indent=2))
    else:
        # Text output
        click.echo("")
        click.echo(click.style(info.name, fg="cyan", bold=True))
        click.echo("=" * 60)
        click.echo("")

        click.echo(click.style("Basic Information", bold=True))
        click.echo(f"  Repository:     {info.repo_id}")
        click.echo(f"  Revision:       {info.revision or 'Unpinned'}")
        click.echo(f"  Short Name:     {short_name}")
        click.echo(f"  Task:           {task_name}")
        click.echo(f"  Type:           {info.model_type.value}")
        click.echo(f"  Size:           {info.size_str}")
        click.echo(f"  Default:        {'Yes' if info.is_default else 'No'}")
        click.echo(f"  Catalog Status: {info.status_label}")
        click.echo(f"  Profile:        {info.profile or 'base model'}")
        click.echo(f"  License:        {info.license or 'Unknown'}")
        click.echo(f"  Commercial Use: {info.commercial_use if info.commercial_use is not None else 'Unknown'}")
        click.echo(f"  Runtime Dtype:  {info.dtype or 'Auto'}")
        click.echo(
            f"  VRAM:           {info.min_vram_gb or '?'} GB minimum / "
            f"{info.recommended_vram_gb or '?'} GB recommended"
        )

        if info.description:
            click.echo(f"  Description:    {info.description}")

        click.echo("")

        # Download status
        click.echo(click.style("Download Status", bold=True))
        if is_downloaded:
            click.echo(click.style("  Status:         ✓ Downloaded", fg="green"))
            click.echo(f"  Location:       {local_path}")
        else:
            click.echo(click.style("  Status:         Not downloaded", fg="yellow"))
            click.echo(f"  To download:    hftool download -m {short_name}")

        click.echo("")

        # Recommended settings
        if recommended_settings:
            click.echo(click.style("Recommended Settings", bold=True))
            for key, value in recommended_settings.items():
                # Format key nicely
                display_key = key.replace("_", " ").title()
                click.echo(f"  {display_key + ':':<20} {value}")
            click.echo("")

        if info.adapter:
            click.echo(click.style("Profile Adapter", bold=True))
            click.echo(f"  Repository:     {info.adapter.repo_id}")
            click.echo(f"  Revision:       {info.adapter.revision or 'Unpinned'}")
            click.echo(f"  Weight:         {info.adapter.weight_name}")
            click.echo(f"  Size:           {info.adapter.size_gb:.2f} GB")
            click.echo("")

        # Dependencies
        if info.pip_dependencies:
            click.echo(click.style("Dependencies", bold=True))
            for dep in info.pip_dependencies:
                click.echo(f"  - {dep}")
            click.echo("")

        # Links
        click.echo(click.style("Links", bold=True))
        click.echo(f"  HuggingFace:    {hf_url}")
        click.echo("")


# =============================================================================
# CLEAN COMMAND
# =============================================================================

@click.command("clean")
@click.option("--model", "-m", "models", multiple=True, help="Delete specific model(s) - can be used multiple times")
@click.option("--all", "delete_all", is_flag=True, help="Delete all downloaded models")
@click.option("--select", "-s", is_flag=True, help="Interactive selection mode")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.pass_context
def clean_command(
    ctx: click.Context,
    models: tuple,
    delete_all: bool,
    select: bool,
    yes: bool,
):
    """Delete downloaded models to free disk space.

    \b
    Examples:
      hftool clean                          # Interactive selection
      hftool clean -s                       # Same as above (explicit)
      hftool clean -m whisper-large-v3      # Delete specific model
      hftool clean -m model1 -m model2      # Delete multiple models
      hftool clean --all                    # Delete all models
      hftool clean --all -y                 # Delete without confirmation
    """
    from hftool.core.download import delete_model, get_models_disk_usage
    from hftool.cli.helpers import resolve_model_to_repo_id, format_size, interactive_model_select

    # Delete all models
    if delete_all:
        usage = get_models_disk_usage()
        if not usage["models"]:
            click.echo("No models to delete.")
            return

        click.echo(f"This will delete {len(usage['models'])} models ({usage['total_str']})")

        if not yes:
            if not click.confirm("Are you sure?"):
                click.echo("Cancelled.")
                return

        for model_info in usage["models"]:
            repo_id = model_info["repo_id"]
            if delete_model(repo_id):
                click.echo(f"Deleted: {repo_id}")

        click.echo("Done.")
        return

    # Delete specific model(s) by name
    if models:
        for model in models:
            repo_id = resolve_model_to_repo_id(model)

            if not yes:
                if not click.confirm(f"Delete {repo_id}?"):
                    click.echo(f"Skipped: {repo_id}")
                    continue

            if delete_model(repo_id):
                click.echo(f"Deleted: {repo_id}")
            else:
                click.echo(f"Model not found: {repo_id}")
        return

    # Interactive selection mode (default when no arguments)
    usage = get_models_disk_usage()
    if not usage["models"]:
        click.echo("No models downloaded.")
        click.echo("")
        click.echo("To download models, run:")
        click.echo("  hftool download -t <task>")
        return

    # Show interactive selection
    selected = interactive_model_select(usage["models"])

    if not selected:
        click.echo("No models selected.")
        return

    # Calculate total size
    total_size = sum(m["size_bytes"] for m in selected)
    total_str = format_size(total_size)

    click.echo("")
    click.echo(f"Selected {len(selected)} model(s) to delete ({total_str}):")
    for model_info in selected:
        click.echo(f"  - {model_info['repo_id']} ({model_info['size_str']})")
    click.echo("")

    if not yes:
        if not click.confirm("Delete these models?"):
            click.echo("Cancelled.")
            return

    # Delete selected models
    for model_info in selected:
        repo_id = model_info["repo_id"]
        if delete_model(repo_id):
            click.echo(f"Deleted: {repo_id}")

    click.echo("Done.")
