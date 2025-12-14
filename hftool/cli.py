#!/usr/bin/env python3
"""hftool CLI - Command-line interface for Hugging Face models.

Supports:
- Text-to-Image (Z-Image, SDXL, FLUX)
- Text-to-Video (HunyuanVideo, CogVideoX, Wan2.2)
- Text-to-Speech (VibeVoice, Bark, MMS-TTS)
- Speech-to-Text (Whisper)
- And other transformers pipeline tasks
"""

import sys
from typing import Optional

import click


@click.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@click.option("--task", "-t", default=None, help="Task to perform (e.g., text-to-image, text-to-video, tts, asr)")
@click.option("--model", "-m", default=None, help="Model name or path (uses task default if not specified)")
@click.option("--input", "-i", "input_data", default=None, help="Input data (text, file path, or URL)")
@click.option("--output-file", "-o", default=None, help="Output file path")
@click.option("--device", "-d", default="auto", help="Device to use (auto, cuda, mps, cpu)")
@click.option("--dtype", default=None, help="Data type (bfloat16, float16, float32)")
@click.option("--list-tasks", is_flag=True, help="List all available tasks")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def main(
    ctx: click.Context,
    task: Optional[str],
    model: Optional[str],
    input_data: str,
    output_file: Optional[str],
    device: str,
    dtype: Optional[str],
    list_tasks: bool,
    verbose: bool,
):
    """hftool - Run Hugging Face models from the command line.
    
    Examples:
    
    \b
    # Text-to-Image with Z-Image
    hftool -t text-to-image -m Tongyi-MAI/Z-Image-Turbo -i "A cat in space" -o cat.png
    
    \b
    # Text-to-Video with HunyuanVideo
    hftool -t text-to-video -m hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \\
           -i "A person walking on a beach" -o beach.mp4
    
    \b
    # Text-to-Speech with VibeVoice
    hftool -t tts -m microsoft/VibeVoice-Realtime-0.5B -i "Hello world" -o hello.wav
    
    \b
    # Speech-to-Text with Whisper
    hftool -t asr -m openai/whisper-large-v3 -i recording.wav -o transcript.txt
    
    \b
    # Pass extra arguments to the model (after --)
    hftool -t text-to-image -m Tongyi-MAI/Z-Image-Turbo -i "A cat" -o cat.png \\
           -- --num_inference_steps 9 --guidance_scale 0.0
    """
    # Handle --list-tasks
    if list_tasks:
        _list_tasks()
        return
    
    # Validate required arguments when not listing tasks
    if task is None:
        click.echo("Error: Missing option '--task' / '-t'.", err=True)
        sys.exit(1)
    if input_data is None:
        click.echo("Error: Missing option '--input' / '-i'.", err=True)
        sys.exit(1)
    
    # Parse extra arguments (after --)
    extra_kwargs = _parse_extra_args(ctx.args)
    
    if verbose:
        click.echo(f"Task: {task}")
        click.echo(f"Model: {model or '(default)'}")
        click.echo(f"Input: {input_data}")
        click.echo(f"Output: {output_file or '(auto)'}")
        click.echo(f"Device: {device}")
        if extra_kwargs:
            click.echo(f"Extra args: {extra_kwargs}")
    
    try:
        # Import here to avoid slow startup for --help
        from hftool.core.registry import get_task_config, get_default_model, TASK_ALIASES
        
        # Resolve task alias
        resolved_task = TASK_ALIASES.get(task, task)
        
        # Get task configuration
        task_config = get_task_config(resolved_task)
        
        # Use default model if not specified
        if model is None:
            model = get_default_model(resolved_task)
            if verbose:
                click.echo(f"Using default model: {model}")
        
        # Check dependencies
        _check_task_deps(task_config, verbose)
        
        # Run the task
        result = _run_task(
            task_name=resolved_task,
            task_config=task_config,
            model=model,
            input_data=input_data,
            output_file=output_file,
            device=device,
            dtype=dtype,
            verbose=verbose,
            **extra_kwargs
        )
        
        # Print result summary
        if output_file:
            click.echo(f"Output saved to: {output_file}")
        elif isinstance(result, str):
            click.echo(result)
        elif isinstance(result, dict) and "text" in result:
            click.echo(result["text"])
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _list_tasks():
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


def _parse_extra_args(args: list) -> dict:
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
                value = _parse_value(value)
                kwargs[key] = value
                i += 2
            else:
                # Boolean flag
                kwargs[key] = True
                i += 1
        else:
            i += 1
    
    return kwargs


def _parse_value(value: str):
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


def _check_task_deps(task_config, verbose: bool):
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


def _run_task(
    task_name: str,
    task_config,
    model: str,
    input_data: str,
    output_file: Optional[str],
    device: str,
    dtype: Optional[str],
    verbose: bool,
    **kwargs
):
    """Run the specified task."""
    from hftool.io.input_loader import load_input, detect_input_type, InputType
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
        output_type = output_type_map.get(task_config.output_type, OutputType.TEXT)
        output_file = get_output_path(
            input_path=input_data if task_config.input_type != "text" else None,
            output_type=output_type,
        )
    
    # Load and run task handler
    if task_name == "text-to-image":
        from hftool.tasks.text_to_image import create_task
        task = create_task(device=device, dtype=dtype)
    elif task_name in ("text-to-video", "image-to-video"):
        from hftool.tasks.text_to_video import create_task
        mode = task_config.config.get("mode", "t2v")
        task = create_task(device=device, dtype=dtype, mode=mode)
    elif task_name == "text-to-speech":
        from hftool.tasks.text_to_speech import create_task
        task = create_task(device=device, dtype=dtype)
    elif task_name == "automatic-speech-recognition":
        from hftool.tasks.speech_to_text import create_task
        task = create_task(device=device, dtype=dtype)
    else:
        # Fallback to generic transformers pipeline
        from hftool.tasks.transformers_generic import create_task
        task = create_task(task_name=task_name, device=device, dtype=dtype)
    
    if verbose:
        click.echo(f"Loading model: {model}")
    
    # Execute task
    result = task.execute(
        model=model,
        input_data=input_data,
        output_path=output_file,
        **kwargs
    )
    
    return result


if __name__ == "__main__":
    main()
