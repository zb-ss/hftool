"""Tool commands — doctor, benchmark, completion."""

import sys
from typing import Optional

import click

from hftool.core.completion import complete_tasks, complete_models, complete_devices, complete_dtypes


# =============================================================================
# BENCHMARK COMMAND
# =============================================================================

@click.command("benchmark")
@click.option("--task", "-t", required=False, shell_complete=complete_tasks, help="Task to benchmark")
@click.option("--model", "-m", required=False, shell_complete=complete_models, help="Model to benchmark")
@click.option("--all", "benchmark_all", is_flag=True, help="Benchmark all downloaded models")
@click.option("--device", "-d", default="auto", shell_complete=complete_devices, help="Device to use")
@click.option("--dtype", default=None, shell_complete=complete_dtypes, help="Data type")
@click.option("--skip-large", is_flag=True, help="Skip models larger than 15GB")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def benchmark_command(
    ctx: click.Context,
    task: Optional[str],
    model: Optional[str],
    benchmark_all: bool,
    device: str,
    dtype: Optional[str],
    skip_large: bool,
    as_json: bool,
):
    """Benchmark model performance (load time, inference time, VRAM).

    \b
    Examples:
      hftool benchmark -t text-to-image -m z-image-turbo
      hftool benchmark -t asr -m whisper-large-v3
      hftool benchmark --all                    # Benchmark all downloaded models
      hftool benchmark --all --skip-large       # Skip models >15GB

    Results are cached in ~/.hftool/benchmarks.json for reference.
    """
    from hftool.core.benchmark import run_benchmark, save_benchmark, load_benchmarks, get_benchmarks_file
    from hftool.core.models import MODEL_REGISTRY, get_default_model_info
    from hftool.core.download import get_download_status
    import json as json_module

    verbose = ctx.obj.get("verbose", False) and not as_json

    if benchmark_all:
        # Benchmark all downloaded models
        click.echo("Benchmarking all downloaded models...")
        click.echo("")

        results = []
        for task_name, models in MODEL_REGISTRY.items():
            for short_name, info in models.items():
                status = get_download_status(info.repo_id)

                if status != "downloaded":
                    continue

                if skip_large and info.size_gb > 15:
                    if verbose:
                        click.echo(f"Skipping {info.name} ({info.size_str}) - too large")
                    continue

                click.echo(f"Benchmarking {info.name} ({task_name})...")

                result = run_benchmark(
                    task=task_name,
                    model=short_name,
                    device=device,
                    dtype=dtype,
                    verbose=verbose,
                )

                save_benchmark(result)
                results.append(result)
                click.echo("")

        if as_json:
            output = [
                {
                    "task": r.task,
                    "model": r.model,
                    "repo_id": r.repo_id,
                    "load_time": r.load_time,
                    "inference_time": r.inference_time,
                    "total_time": r.total_time,
                    "vram_peak": r.vram_peak,
                    "vram_allocated": r.vram_allocated,
                    "success": r.success,
                    "error": r.error,
                }
                for r in results
            ]
            click.echo(json_module.dumps(output, indent=2))
        else:
            click.echo("=" * 60)
            click.echo(f"Benchmarked {len(results)} models")
            click.echo(f"Results saved to: {get_benchmarks_file()}")

        return

    if not task or not model:
        click.echo("Error: Must specify --task and --model, or use --all", err=True)
        click.echo("", err=True)
        click.echo("Examples:", err=True)
        click.echo("  hftool benchmark -t text-to-image -m z-image-turbo", err=True)
        click.echo("  hftool benchmark --all", err=True)
        sys.exit(1)

    # Benchmark specific model
    result = run_benchmark(
        task=task,
        model=model,
        device=device,
        dtype=dtype,
        verbose=verbose,
    )

    # Save result
    save_benchmark(result)

    # Output result
    if as_json:
        output = {
            "task": result.task,
            "model": result.model,
            "repo_id": result.repo_id,
            "timestamp": result.timestamp,
            "device": result.device,
            "dtype": result.dtype,
            "load_time": result.load_time,
            "inference_time": result.inference_time,
            "total_time": result.total_time,
            "vram_peak": result.vram_peak,
            "vram_allocated": result.vram_allocated,
            "test_prompt": result.test_prompt,
            "test_params": result.test_params,
            "success": result.success,
            "error": result.error,
        }
        click.echo(json_module.dumps(output, indent=2))
    else:
        click.echo("")
        click.echo("=" * 60)
        click.echo("Benchmark Results")
        click.echo("=" * 60)
        click.echo(f"Task:            {result.task}")
        click.echo(f"Model:           {result.model}")
        click.echo(f"Device:          {result.device}")

        if result.success:
            click.echo(click.style(f"Status:          ✓ Success", fg="green"))
            click.echo("")
            click.echo(f"Load time:       {result.load_time:.2f}s")
            click.echo(f"Inference time:  {result.inference_time:.2f}s")
            click.echo(f"Total time:      {result.total_time:.2f}s")

            if result.vram_peak:
                click.echo("")
                click.echo(f"VRAM peak:       {result.vram_peak:.2f} GB")
                click.echo(f"VRAM allocated:  {result.vram_allocated:.2f} GB")
        else:
            click.echo(click.style(f"Status:          ✗ Failed", fg="red"))
            click.echo(f"Error:           {result.error}")

        click.echo("")
        click.echo(f"Results saved to: {get_benchmarks_file()}")


# =============================================================================
# COMPLETION COMMAND
# =============================================================================

@click.command("completion")
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]), required=False)
@click.option("--install", is_flag=True, help="Install completion for current shell")
@click.pass_context
def completion_command(ctx: click.Context, shell: Optional[str], install: bool):
    """Show or install shell completion scripts.

    \b
    Examples:
      hftool completion bash               # Show bash completion script
      hftool completion zsh                # Show zsh completion script
      hftool completion fish               # Show fish completion script
      hftool completion --install          # Auto-detect and install
      hftool completion bash --install     # Install bash completion

    \b
    After installation, restart your shell or run:
      source ~/.bashrc    # for bash
      source ~/.zshrc     # for zsh
      # fish completion loads automatically
    """
    from hftool.core.completion import (
        get_shell_name,
        get_completion_script,
        install_completion
    )

    # Auto-detect shell if not specified
    if shell is None:
        shell = get_shell_name()
        if shell is None:
            click.echo("Error: Could not detect shell. Please specify: bash, zsh, or fish", err=True)
            sys.exit(1)

    # Install completion
    if install:
        try:
            if install_completion(shell):
                click.echo(f"Completion installed for {shell}")
                click.echo("")
                click.echo("Restart your shell or run:")
                if shell == "bash":
                    click.echo("  source ~/.bashrc")
                elif shell == "zsh":
                    click.echo("  source ~/.zshrc")
                elif shell == "fish":
                    click.echo("  # fish completion loads automatically")
            else:
                click.echo(f"Completion already installed for {shell}")
        except Exception as e:
            click.echo(f"Error installing completion: {e}", err=True)
            sys.exit(1)
        return

    # Show completion script
    try:
        script = get_completion_script(shell)
        click.echo(script)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# =============================================================================
# DOCTOR COMMAND
# =============================================================================

@click.command("doctor")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def doctor_command(ctx: click.Context, as_json: bool):
    """Run system diagnostics and check hftool health.

    \b
    Checks:
      - Python version
      - PyTorch installation
      - GPU availability
      - ffmpeg (required for video/audio)
      - Network connectivity
      - Optional features
      - Configuration files

    \b
    Examples:
      hftool doctor              # Run all checks
      hftool doctor --json       # Output as JSON

    \b
    Exit codes:
      0 = All checks passed
      1 = Warnings found
      2 = Errors found
    """
    from hftool.core.doctor import run_doctor_checks, format_doctor_report
    import json as json_module

    # Run all checks
    report = run_doctor_checks()

    # Output results
    if as_json:
        output = report.to_dict()
        click.echo(json_module.dumps(output, indent=2))
    else:
        output = format_doctor_report(report, use_color=True)
        if output:  # Plain text fallback
            click.echo(output)
        # Otherwise rich already printed to console

    # Exit with appropriate code
    sys.exit(report.get_exit_code())
