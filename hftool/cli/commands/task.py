"""Task execution command — run and run_task_command."""

import os
import sys
from typing import Any, Dict, Optional

import click

from hftool.core.completion import (
    complete_tasks,
    complete_models,
    complete_devices,
    complete_dtypes,
    complete_input,
)


# =============================================================================
# RUN COMMAND (explicit subcommand alternative)
# =============================================================================

@click.command("run")
@click.option("--task", "-t", required=True, shell_complete=complete_tasks, help="Task to perform")
@click.option("--model", "-m", default=None, shell_complete=complete_models, help="Model name or path")
@click.option("--input", "-i", "input_data", default=None, shell_complete=complete_input, help="Input data (@ references or @? for interactive)")
@click.option("--output-file", "-o", default=None, help="Output file path")
@click.option("--device", "-d", default="auto", shell_complete=complete_devices, help="Device to use")
@click.option("--gpu", "-g", default=None, envvar="HFTOOL_GPU", help="GPU(s) to use: 'auto', 'all', '0', '1', '0,1'")
@click.option("--dtype", default=None, shell_complete=complete_dtypes, help="Data type")
@click.option("--seed", type=int, default=None, help="Random seed for reproducibility")
@click.option("--interactive", is_flag=True, help="Interactive JSON builder mode")
@click.option("--dry-run", is_flag=True, help="Preview model, GPU, and settings without downloading or loading")
@click.option("--open/--no-open", default=None, help="Open output file with default application")
@click.pass_context
def run_command(
    ctx: click.Context,
    task: str,
    model: Optional[str],
    input_data: Optional[str],
    output_file: Optional[str],
    device: str,
    gpu: Optional[str],
    dtype: Optional[str],
    seed: Optional[int],
    interactive: bool,
    dry_run: bool,
    open: Optional[bool],
):
    """Run a task with the specified model."""
    from hftool.cli.commands.setup import ensure_pytorch_ready

    verbose = ctx.obj.get("verbose", False)
    # Use command-level --open if specified, otherwise use global
    open_output = open if open is not None else ctx.obj.get("open")
    # Use command-level --seed if specified, otherwise use global
    final_seed = seed if seed is not None else ctx.obj.get("seed")
    # Use command-level --interactive if specified, otherwise use global
    final_interactive = interactive or ctx.obj.get("interactive", False)
    # Use command-level --gpu if specified, otherwise use global
    final_gpu = gpu if gpu is not None else ctx.obj.get("gpu")

    # Ensure PyTorch is ready (this may import torch)
    final_dry_run = dry_run or ctx.obj.get("dry_run", False)
    if not final_dry_run and not ensure_pytorch_ready():
        sys.exit(1)

    run_task_command(
        ctx,
        task,
        model,
        input_data,
        output_file,
        device,
        dtype,
        final_seed,
        final_interactive,
        verbose,
        open_output=open_output,
        gpu=final_gpu,
        dry_run=final_dry_run,
    )


# =============================================================================
# TASK EXECUTION (internal)
# =============================================================================

def run_task_command(
    ctx: click.Context,
    task: str,
    model: Optional[str],
    input_data: Optional[str],
    output_file: Optional[str],
    device: str,
    dtype: Optional[str],
    seed: Optional[int],
    interactive: bool,
    verbose: bool,
    quiet: bool = False,
    output_json: bool = False,
    embed_metadata: bool = True,
    open_output: Optional[bool] = None,
    wizard_extra_kwargs: Optional[Dict[str, Any]] = None,
    gpu: Optional[str] = None,
    dry_run: bool = False,
):
    """Execute a task (internal helper)."""
    import random
    import json as json_module

    from hftool.cli.helpers import (
        parse_extra_args,
        should_open_output,
        open_file,
        check_task_deps,
    )

    # Parse extra arguments (after --)
    extra_args = ctx.obj.get("extra_args", ()) if ctx.obj else ()
    extra_kwargs = parse_extra_args(list(extra_args))

    # Merge wizard extra_kwargs (from interactive mode) with CLI extra args
    # CLI args take priority over wizard params
    if wizard_extra_kwargs:
        extra_kwargs = {**wizard_extra_kwargs, **extra_kwargs}

    # Generate random seed if not provided
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    # Add seed to extra_kwargs (will be passed to model if supported)
    if "generator_seed" not in extra_kwargs and "seed" not in extra_kwargs:
        extra_kwargs["seed"] = seed

    # Parse GPU selection for logging and extra_kwargs
    gpu_indices = []
    if gpu and device in ("auto", "cuda"):
        from hftool.core.device import parse_gpu_selection
        gpu_indices = parse_gpu_selection(gpu)
        if gpu_indices:
            # Pass GPU info to extra_kwargs for tasks that need it
            extra_kwargs["_gpu_indices"] = gpu_indices

    # Quiet and JSON modes suppress verbose output
    if verbose and not quiet and not output_json:
        click.echo(f"Task: {task}")
        click.echo(f"Model: {model or '(default)'}")
        click.echo(f"Input: {input_data}")
        click.echo(f"Output: {output_file or '(auto)'}")
        click.echo(f"Device: {device}")
        if gpu and gpu_indices:
            click.echo(f"GPU: {gpu} (indices: {gpu_indices})")
        if extra_kwargs:
            # Filter out internal keys for display
            display_kwargs = {k: v for k, v in extra_kwargs.items() if not k.startswith("_")}
            if display_kwargs:
                click.echo(f"Extra args: {display_kwargs}")

    try:
        # Import here to avoid slow startup for --help
        from hftool.core.registry import get_task_config, TASK_ALIASES
        from hftool.core.models import get_default_model_info, get_model_info
        from hftool.core.download import ensure_model_available
        from hftool.core.config import Config

        # Resolve task alias
        resolved_task = TASK_ALIASES.get(task, task)

        # Handle batch mode
        batch_source = ctx.obj.get("batch") if ctx.obj else None
        batch_json_file = ctx.obj.get("batch_json") if ctx.obj else None
        batch_output_dir = ctx.obj.get("batch_output_dir") if ctx.obj else None

        if batch_source or batch_json_file:
            from hftool.core.batch import load_batch_inputs, load_batch_json, process_batch

            if not quiet and not output_json:
                click.echo("Running in batch mode...")
                click.echo("")

            # Load inputs
            if batch_json_file:
                # Load from JSON file
                batch_entries = load_batch_json(batch_json_file)

                # For JSON batch, we don't use the simple file list processing
                # Instead, each entry can have its own params
                if not quiet and not output_json:
                    click.echo(f"Loaded {len(batch_entries)} entries from JSON batch file")
                    click.echo("")

                # Process each entry with its own params
                results = []
                success_count = 0
                failure_count = 0

                for i, entry in enumerate(batch_entries):
                    entry_input = entry["input"]
                    entry_output = entry.get("output")
                    entry_params = entry.get("params", {})

                    # Merge params (entry params override command-line params)
                    merged_kwargs = {**extra_kwargs, **entry_params}

                    if not quiet and not output_json:
                        click.echo(f"[{i+1}/{len(batch_entries)}] Processing: {entry_input}")

                    # Run single task
                    try:
                        run_task_command(
                            ctx=ctx,
                            task=task,
                            model=model,
                            input_data=entry_input,
                            output_file=entry_output,
                            device=device,
                            dtype=dtype,
                            seed=seed,
                            interactive=False,
                            verbose=False,  # Suppress verbose for batch
                            quiet=True,  # Suppress output
                            output_json=False,
                            embed_metadata=embed_metadata,
                            open_output=False,  # Don't open files in batch
                        )
                        success_count += 1
                        if not quiet and not output_json:
                            click.echo("  ✓ Success")
                    except Exception as e:
                        failure_count += 1
                        if not quiet and not output_json:
                            click.echo(click.style(f"  ✗ Failed: {e}", fg="red"), err=True)

                # Print summary
                if not quiet and not output_json:
                    click.echo("")
                    click.echo("=" * 60)
                    click.echo(f"Batch processing complete: {success_count} succeeded, {failure_count} failed")
                elif output_json:
                    result_data = {
                        "success": True,
                        "batch_mode": "json",
                        "total": len(batch_entries),
                        "succeeded": success_count,
                        "failed": failure_count,
                    }
                    click.echo(json_module.dumps(result_data, indent=2))

                return

            else:
                # Load from file/directory
                inputs = load_batch_inputs(batch_source)

                if not inputs:
                    click.echo(f"No inputs found in: {batch_source}", err=True)
                    sys.exit(1)

                if not quiet and not output_json:
                    click.echo(f"Loaded {len(inputs)} inputs")
                    click.echo("")

                # Determine output extension based on task
                task_config = get_task_config(resolved_task)
                output_ext_map = {
                    "image": ".png",
                    "audio": ".wav",
                    "video": ".mp4",
                    "text": ".txt",
                }
                output_extension = output_ext_map.get(task_config.output_type, ".out")

                # Process batch
                results, success_count, failure_count = process_batch(
                    task=task,
                    inputs=inputs,
                    model=model,
                    device=device,
                    dtype=dtype,
                    output_dir=batch_output_dir,
                    output_extension=output_extension,
                    extra_kwargs=extra_kwargs,
                    verbose=not quiet and not output_json,
                )

                # Print summary
                if not quiet and not output_json:
                    click.echo("")
                    click.echo("=" * 60)
                    click.echo(f"Batch processing complete: {success_count} succeeded, {failure_count} failed")
                elif output_json:
                    result_data = {
                        "success": True,
                        "batch_mode": "file",
                        "total": len(inputs),
                        "succeeded": success_count,
                        "failed": failure_count,
                        "results": [
                            {
                                "input": r.input_file,
                                "output": r.output_file,
                                "success": r.success,
                                "error": r.error,
                                "execution_time": r.execution_time,
                            }
                            for r in results
                        ],
                    }
                    click.echo(json_module.dumps(result_data, indent=2))

                return

        # Load configuration early
        config = Config.get()

        # Apply config defaults if CLI args not provided
        # Device: use config if still "auto"
        if device == "auto":
            device = config.get_value("device", task=resolved_task, default="auto")

        # Dtype: use config if None
        if dtype is None:
            dtype = config.get_value("dtype", task=resolved_task, default=None)

        # Model: use config if None
        if model is None:
            model = config.get_value("model", task=resolved_task, default=None)

        # Resolve model alias if model is set
        if model:
            model = config.resolve_model_alias(model)

        # Merge config task-specific params (lower priority than extra_kwargs)
        # Get task-specific config section as dict
        task_params = {}
        if resolved_task in config._config:
            task_config_section = config._config[resolved_task]
            if isinstance(task_config_section, dict):
                # Extract only parameter-like keys (not 'model', 'device', 'dtype')
                reserved_keys = {'model', 'device', 'dtype'}
                task_params = {k: v for k, v in task_config_section.items() if k not in reserved_keys}

        # Merge: config params < extra_kwargs (CLI has priority)
        extra_kwargs = {**task_params, **extra_kwargs}

        # Handle interactive mode and file references
        if interactive or (input_data and input_data == "@?"):
            # Interactive JSON builder
            try:
                from hftool.io.interactive_input import build_interactive_input
                input_data = build_interactive_input(resolved_task)
            except ValueError as e:
                click.echo(f"Error: {e}", err=True)
                sys.exit(1)
        elif input_data and input_data.startswith("@"):
            # Resolve @ file reference
            try:
                from hftool.io.file_picker import resolve_file_reference
                input_data = resolve_file_reference(input_data, task=resolved_task)
                if verbose:
                    click.echo(f"Resolved file reference to: {input_data}")
            except ValueError as e:
                click.echo(f"Error resolving file reference: {e}", err=True)
                sys.exit(1)
        elif input_data is None and interactive:
            # Interactive mode but no schema available - build basic JSON
            try:
                from hftool.io.interactive_input import build_interactive_input
                input_data = build_interactive_input(resolved_task)
            except ValueError:
                # Fall back to text prompt
                try:
                    input_data = click.prompt("Enter input data")
                except click.Abort:
                    click.echo("Input cancelled", err=True)
                    sys.exit(1)

        # At this point input_data must be set
        if input_data is None:
            click.echo("Error: No input data provided", err=True)
            sys.exit(1)

        # Get task configuration
        task_config = get_task_config(resolved_task)

        # Determine which model to use
        model_info = None
        pip_dependencies = None
        model_gated = False

        if model is None:
            # Use default model
            model_info = get_default_model_info(resolved_task)
            model_repo_id = model_info.repo_id
            model_size = model_info.size_gb
            model_name = model_info.name
            pip_dependencies = model_info.pip_dependencies
            model_gated = getattr(model_info, 'gated', False)
        else:
            # Check if model is a local path
            if os.path.exists(model):
                model_repo_id = model
                model_size = 0
                model_name = os.path.basename(model)
            else:
                # Try to find model info
                try:
                    model_info = get_model_info(resolved_task, model)
                    model_repo_id = model_info.repo_id
                    model_size = model_info.size_gb
                    model_name = model_info.name
                    pip_dependencies = model_info.pip_dependencies
                    model_gated = getattr(model_info, 'gated', False)
                except ValueError:
                    # Not in registry - assume it's a HuggingFace repo_id
                    model_repo_id = model
                    model_size = 5.0  # Estimate
                    model_name = model.split("/")[-1] if "/" in model else model

        if verbose:
            click.echo(f"Using model: {model_repo_id}")
            if pip_dependencies:
                click.echo(f"Model dependencies: {', '.join(pip_dependencies)}")

        if model_info:
            from hftool.core.executor import TaskRequest, preview_task

            preview = preview_task(
                TaskRequest(
                    task_name=resolved_task,
                    model=model,
                    input_data=input_data,
                    output_path=output_file,
                    device=device,
                    dtype=dtype,
                    gpu=gpu,
                    seed=seed,
                    extra_kwargs=extra_kwargs,
                )
            )
            if dry_run:
                _display_task_preview(preview, output_json)
                return

            cpu_offload = os.environ.get("HFTOOL_CPU_OFFLOAD", "").lower()
            multi_gpu = gpu == "all" or bool(gpu and "," in gpu)
            if (
                preview.gpu_adequate is False
                and not multi_gpu
                and cpu_offload not in {"1", "2", "true", "yes"}
            ):
                raise RuntimeError(
                    f"Insufficient live GPU headroom. {preview.gpu_message} "
                    "Free VRAM, choose another GPU, use --gpu all, or set "
                    "HFTOOL_CPU_OFFLOAD=1."
                )
        elif dry_run:
            custom_preview = {
                "task": resolved_task,
                "model": model_repo_id,
                "catalog_status": "custom",
                "download_status": "local" if os.path.exists(model_repo_id) else "unknown",
                "device": device,
                "dtype": dtype,
                "inference_overrides": {
                    key: value for key, value in extra_kwargs.items() if not key.startswith("_")
                },
            }
            if output_json:
                click.echo(json_module.dumps(custom_preview, indent=2))
            else:
                click.echo("Dry run (no download or model load)")
                for key, value in custom_preview.items():
                    click.echo(f"  {key.replace('_', ' ').title()}: {value}")
            return

        # Ensure model is available (prompts to download if needed)
        if not os.path.exists(model_repo_id):
            model_path = ensure_model_available(
                repo_id=model_repo_id,
                size_gb=model_size,
                task_name=resolved_task,
                model_name=model_name,
                pip_dependencies=pip_dependencies,
                gated=model_gated,
                revision=model_info.revision if model_info else None,
                ignore_patterns=model_info.ignore_patterns if model_info else None,
            )
            # Use the local path for loading
            model_to_load = str(model_path)
        else:
            model_to_load = model_repo_id

        adapter_path = None
        if model_info and model_info.adapter:
            from hftool.core.download import ensure_model_file_available

            adapter = model_info.adapter
            adapter_path = ensure_model_file_available(
                repo_id=adapter.repo_id,
                filename=adapter.weight_name,
                size_gb=adapter.size_gb,
                task_name=resolved_task,
                model_name=model_name,
                revision=adapter.revision,
            )

        # Check dependencies
        check_task_deps(task_config, verbose)

        # Run the task (quiet mode suppresses progress bars)
        # Get model metadata for load kwargs (e.g., subfolder for LTX-2)
        if model_info:
            from hftool.core.executor import build_model_runtime_kwargs

            load_defaults, inference_defaults = build_model_runtime_kwargs(
                model_info,
                str(adapter_path) if adapter_path else None,
            )
            model_metadata = {**load_defaults, **inference_defaults}
        else:
            model_metadata = {}

        result = _run_task(
            task_name=resolved_task,
            task_config=task_config,
            model=model_to_load,
            input_data=input_data,
            output_file=output_file,
            device=device,
            dtype=dtype,
            verbose=verbose and not quiet and not output_json,
            model_metadata=model_metadata,
            # Filter out internal keys (starting with _) before passing to pipeline
            **{k: v for k, v in extra_kwargs.items() if not k.startswith("_")}
        )

        # Embed metadata in output file
        if output_file and embed_metadata and os.path.exists(output_file):
            from hftool.core.metadata import embed_metadata as do_embed_metadata

            # Extract prompt from input_data
            prompt = None
            if isinstance(input_data, str):
                # Try to parse JSON first
                try:
                    data = json_module.loads(input_data)
                    if isinstance(data, dict):
                        # Look for common prompt keys
                        prompt = data.get("prompt") or data.get("text") or data.get("caption")
                except json_module.JSONDecodeError:
                    # Not JSON, use as-is (but limit length)
                    prompt = input_data[:500] if len(input_data) > 500 else input_data

            do_embed_metadata(
                file_path=output_file,
                task=resolved_task,
                model=model or model_repo_id,
                prompt=prompt,
                seed=seed,
                extra_params=extra_kwargs,
                verbose=verbose and not quiet and not output_json,
            )

        # Print result summary based on output mode
        if output_json:
            # JSON output mode
            result_data = {
                "success": True,
                "task": resolved_task,
                "model": model or model_repo_id,
                "input": input_data,
                "output": output_file,
                "seed": seed,
                "device": device,
                "dtype": dtype,
            }

            # Add result text if available
            if isinstance(result, str):
                result_data["text"] = result
            elif isinstance(result, dict) and "text" in result:
                result_data["text"] = result["text"]

            click.echo(json_module.dumps(result_data, indent=2))
        elif quiet:
            # Quiet mode - only output file path
            if output_file:
                click.echo(output_file)
            elif isinstance(result, str):
                click.echo(result)
            elif isinstance(result, dict) and "text" in result:
                click.echo(result["text"])
        else:
            # Normal output mode
            if output_file:
                click.echo(f"Output saved to: {output_file}")

                # Show reproduction command
                if verbose or seed is not None:
                    repro_parts = ["hftool", "-t", resolved_task]
                    if model:
                        repro_parts.extend(["-m", model])
                    repro_parts.extend(["-i", f'"{input_data}"'])
                    if output_file:
                        repro_parts.extend(["-o", output_file])
                    if seed is not None:
                        repro_parts.extend(["--seed", str(seed)])
                    click.echo(f"Seed: {seed}")
                    click.echo(f"To reproduce: {' '.join(repro_parts)}")

                # Determine if we should open the file
                should_open = should_open_output(
                    open_output=open_output,
                    output_file=output_file,
                    output_type=task_config.output_type,
                )

                if should_open:
                    open_file(output_file, verbose)
            elif isinstance(result, str):
                click.echo(result)
            elif isinstance(result, dict) and "text" in result:
                click.echo(result["text"])

        # Record to history (success)
        from hftool.core.history import History
        history = History.get()
        history.add(
            task=resolved_task,
            model=model,
            input_data=input_data,
            output_file=output_file,
            device=device,
            dtype=dtype,
            seed=seed,
            extra_args=extra_kwargs,
            success=True,
        )

    except SystemExit:
        raise
    except Exception as e:
        # Record to history (failure)
        from hftool.core.history import History
        history = History.get()
        history.add(
            task=task if 'resolved_task' not in locals() else resolved_task,
            model=model,
            input_data=input_data or "",
            output_file=output_file,
            device=device,
            dtype=dtype,
            seed=seed,
            extra_args=extra_kwargs if 'extra_kwargs' in locals() else {},
            success=False,
            error_message=str(e),
        )

        # Handle error output based on mode
        if output_json:
            error_data = {
                "success": False,
                "error": str(e),
                "task": task if 'resolved_task' not in locals() else resolved_task,
                "model": model,
                "input": input_data or "",
                "output": output_file,
            }
            click.echo(json_module.dumps(error_data, indent=2))
        else:
            click.echo(f"Error: {e}", err=True)
            if verbose:
                import traceback
                traceback.print_exc()
        sys.exit(1)


def _run_task(
    task_name: str,
    task_config,
    model: str,
    input_data: str,
    output_file: Optional[str],
    device: str,
    dtype: Optional[str],
    verbose: bool,
    model_metadata: Optional[Dict[str, Any]] = None,
    **kwargs
):
    """Run the specified task."""
    from hftool.io.output_handler import get_output_path, OutputType

    # Map task output types to OutputType
    output_type_map = {
        "text": OutputType.TEXT,
        "image": OutputType.IMAGE,
        "audio": OutputType.AUDIO,
        "video": OutputType.VIDEO,
    }

    # Determine output path if not specified
    if output_file is None:
        import json as json_module
        output_type = output_type_map.get(task_config.output_type, OutputType.TEXT)

        # Extract actual file path from JSON input if needed (for i2i tasks)
        actual_input_path = None
        if task_config.input_type != "text":
            # Try to parse JSON input to extract image path
            if input_data.strip().startswith("{"):
                try:
                    data = json_module.loads(input_data)
                    img_path = data.get("image")
                    # Handle both single path and list of paths
                    if isinstance(img_path, list):
                        actual_input_path = img_path[0] if img_path else None
                    else:
                        actual_input_path = img_path
                except (json_module.JSONDecodeError, TypeError):
                    pass
            else:
                actual_input_path = input_data

        output_file = get_output_path(
            input_path=actual_input_path,
            output_type=output_type,
        )

    # Load and run task handler
    if task_name == "text-to-image":
        from hftool.tasks.text_to_image import create_task
        task_handler = create_task(device=device, dtype=dtype)
    elif task_name == "image-to-image":
        from hftool.tasks.image_to_image import create_task
        task_handler = create_task(device=device, dtype=dtype)
    elif task_name in ("text-to-video", "image-to-video"):
        from hftool.tasks.text_to_video import create_task
        mode = task_config.config.get("mode", "t2v")
        task_handler = create_task(device=device, dtype=dtype, mode=mode)
    elif task_name == "text-to-speech":
        from hftool.tasks.text_to_speech import create_task
        task_handler = create_task(device=device, dtype=dtype)
    elif task_name == "automatic-speech-recognition":
        from hftool.tasks.speech_to_text import create_task
        task_handler = create_task(device=device, dtype=dtype)
    else:
        # Fallback to generic transformers pipeline
        from hftool.tasks.transformers_generic import create_task
        task_handler = create_task(task_name=task_name, device=device, dtype=dtype)

    if verbose:
        click.echo(f"Loading model: {model}")

    # Separate load-time parameters from inference-time parameters in model metadata
    # Load-time: subfolder, revision, variant, etc.
    # Inference-time: num_inference_steps, guidance_scale, height, width, num_frames, etc.
    load_param_names = {"subfolder", "revision", "variant", "torch_dtype", "use_safetensors"}
    load_kwargs = {}
    infer_kwargs = {}
    if model_metadata:
        for key, value in model_metadata.items():
            if key in load_param_names or key.startswith("_"):
                load_kwargs[key] = value
            else:
                infer_kwargs[key] = value

    # Execute task
    result = task_handler.execute(
        model=model,
        input_data=input_data,
        output_path=output_file,
        load_kwargs=load_kwargs,
        infer_kwargs=infer_kwargs,
        **kwargs
    )

    return result


def _display_task_preview(preview, output_json: bool) -> None:
    """Render the shared no-side-effect task preview for CLI users."""
    import json

    data = preview.as_dict()
    if output_json:
        click.echo(json.dumps(data, indent=2))
        return

    click.echo("Dry run (no download or model load)")
    click.echo(f"  Task: {data['task']}")
    click.echo(f"  Model: {data['model_name']} ({data['model']})")
    click.echo(f"  Repository: {data['repo_id']}")
    click.echo(f"  Revision: {data['revision'] or 'unversioned'}")
    if data["adapter_repo_id"]:
        click.echo(
            f"  Adapter: {data['adapter_repo_id']}@{data['adapter_revision']} / "
            f"{data['adapter_weight_name']}"
        )
    click.echo(f"  Catalog: {data['catalog_status']} · Local: {data['download_status']}")
    click.echo(f"  Download: ~{data['total_download_gb']:.2f} GB")
    click.echo(
        f"  VRAM: {data['min_vram_gb'] or '?'} GB minimum · "
        f"{data['recommended_vram_gb'] or '?'} GB recommended"
    )
    click.echo(f"  Dtype: {data['dtype'] or 'auto'}")
    click.echo(f"  License: {data['license'] or 'unknown'}")
    click.echo(f"  GPU: {data['gpu_message']}")
    click.echo(f"  Defaults: {data['inference_defaults']}")
    if data["dependencies"]:
        click.echo(f"  Dependencies: {', '.join(data['dependencies'])}")
