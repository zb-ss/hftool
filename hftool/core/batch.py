"""Batch processing utilities for hftool.

Process multiple inputs from files or directories with parallel execution support.
"""

import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
import click


@dataclass
class BatchResult:
    """Result of processing a single batch item."""
    
    input_file: str
    output_file: Optional[str]
    success: bool
    error: Optional[str] = None
    execution_time: Optional[float] = None


def load_batch_inputs(
    batch_source: str,
    file_pattern: Optional[str] = None,
) -> List[str]:
    """Load batch inputs from file or directory.
    
    Args:
        batch_source: File path or directory path
        file_pattern: Optional glob pattern for filtering files
    
    Returns:
        List of input file paths
    """
    path = Path(batch_source)
    
    # Security: Validate path exists (M-3)
    if not path.exists():
        raise ValueError(f"Batch source not found: {batch_source}")
    
    inputs = []
    
    if path.is_file():
        # Read list from file
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    inputs.append(line)
    
    elif path.is_dir():
        # Glob directory
        if file_pattern:
            inputs = [str(p) for p in path.glob(file_pattern)]
        else:
            # Get all files (not directories)
            inputs = [str(p) for p in path.iterdir() if p.is_file()]
    
    else:
        raise ValueError(f"Invalid batch source: {batch_source}")
    
    # Security: Validate all input files exist and are regular files (M-3)
    validated_inputs = []
    for input_path in inputs:
        try:
            p = Path(input_path).resolve()
            if p.exists() and p.is_file():
                validated_inputs.append(str(p))
            else:
                click.echo(f"Warning: Skipping invalid input: {input_path}", err=True)
        except Exception as e:
            click.echo(f"Warning: Skipping invalid path {input_path}: {e}", err=True)
    
    return validated_inputs


def load_batch_json(batch_file: str) -> List[Dict[str, Any]]:
    """Load batch inputs from JSON array file.
    
    Each entry should be a dictionary with at least "input" key.
    Optional keys: "output", "params" (dict of extra parameters)
    
    Args:
        batch_file: Path to JSON file
    
    Returns:
        List of input dictionaries
    """
    # Security: Validate file exists and size (M-3)
    path = Path(batch_file)
    
    if not path.exists():
        raise ValueError(f"Batch file not found: {batch_file}")
    
    if not path.is_file():
        raise ValueError(f"Batch file is not a regular file: {batch_file}")
    
    # Check file size (max 10MB for JSON)
    file_size = path.stat().st_size
    max_size = 10 * 1024 * 1024  # 10MB
    if file_size > max_size:
        raise ValueError(f"Batch file too large: {file_size} bytes (max {max_size})")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Batch JSON must be an array")
    
    # Validate entries
    validated = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            click.echo(f"Warning: Skipping entry {i}: not a dictionary", err=True)
            continue
        
        if "input" not in entry:
            click.echo(f"Warning: Skipping entry {i}: missing 'input' key", err=True)
            continue
        
        validated.append(entry)
    
    return validated


def generate_output_filename(
    input_path: str,
    index: int,
    output_dir: Optional[str],
    output_extension: str,
    prefix: str = "",
    use_input_name: bool = True,
) -> str:
    """Generate output filename for batch processing.
    
    Args:
        input_path: Input file path
        index: Index in batch (0-based)
        output_dir: Output directory (None = same as input)
        output_extension: Extension for output file (e.g., ".png", ".mp3")
        prefix: Optional prefix for output filename
        use_input_name: Use input filename as base (True) or use numbers (False)
    
    Returns:
        Output file path
    """
    input_p = Path(input_path)
    
    if use_input_name:
        # Use input filename as base
        base_name = input_p.stem
        output_name = f"{prefix}{base_name}{output_extension}"
    else:
        # Use index numbers
        output_name = f"{prefix}{index:04d}{output_extension}"
    
    # Determine output directory
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / output_name
    else:
        # Same directory as input
        output_path = input_p.parent / output_name
    
    return str(output_path)


def process_batch(
    task: str,
    inputs: List[str],
    model: Optional[str],
    device: str,
    dtype: Optional[str],
    output_dir: Optional[str],
    output_extension: str,
    extra_kwargs: Dict[str, Any],
    prefix: str = "",
    use_input_names: bool = True,
    continue_on_error: bool = True,
    verbose: bool = True,
) -> tuple:
    """Process batch of inputs sequentially.
    
    Args:
        task: Task to perform
        inputs: List of input file paths
        model: Model to use
        device: Device
        dtype: Data type
        output_dir: Output directory (None = same as input)
        output_extension: Output file extension
        extra_kwargs: Extra parameters to pass
        prefix: Prefix for output filenames
        use_input_names: Use input filenames (True) or numbers (False)
        continue_on_error: Continue processing on error
        verbose: Show progress
    
    Returns:
        Tuple of (results, success_count, failure_count)
    """
    import time
    from hftool.core.registry import get_task_config, TASK_ALIASES
    from hftool.core.models import get_default_model_info, get_model_info
    from hftool.core.download import ensure_model_available
    
    # Resolve task
    resolved_task = TASK_ALIASES.get(task, task)
    task_config = get_task_config(resolved_task)
    
    # Get model info
    if model is None:
        model_info = get_default_model_info(resolved_task)
        model_repo_id = model_info.repo_id
    else:
        try:
            model_info = get_model_info(resolved_task, model)
            model_repo_id = model_info.repo_id
        except ValueError:
            model_repo_id = model
            model_info = None
    
    # Ensure model is available
    if not os.path.exists(model_repo_id):
        model_path = ensure_model_available(
            repo_id=model_repo_id,
            size_gb=model_info.size_gb if model_info else 5.0,
            task_name=resolved_task,
            model_name=model or "default",
        )
        model_to_load = str(model_path)
    else:
        model_to_load = model_repo_id
    
    # Load task handler once (reuse for all inputs)
    if verbose:
        click.echo(f"Loading model: {model_to_load}")
    
    if resolved_task == "text-to-image":
        from hftool.tasks.text_to_image import create_task
        task_handler = create_task(device=device, dtype=dtype)
    elif resolved_task == "image-to-image":
        from hftool.tasks.image_to_image import create_task
        task_handler = create_task(device=device, dtype=dtype)
    elif resolved_task in ("text-to-video", "image-to-video"):
        from hftool.tasks.text_to_video import create_task
        mode = task_config.config.get("mode", "t2v")
        task_handler = create_task(device=device, dtype=dtype, mode=mode)
    elif resolved_task == "text-to-speech":
        from hftool.tasks.text_to_speech import create_task
        task_handler = create_task(device=device, dtype=dtype)
    elif resolved_task == "automatic-speech-recognition":
        from hftool.tasks.speech_to_text import create_task
        task_handler = create_task(device=device, dtype=dtype)
    else:
        from hftool.tasks.transformers_generic import create_task
        task_handler = create_task(task_name=resolved_task, device=device, dtype=dtype)
    
    # Process inputs
    results = []
    success_count = 0
    failure_count = 0
    
    for i, input_path in enumerate(inputs):
        if verbose:
            click.echo(f"\n[{i+1}/{len(inputs)}] Processing: {Path(input_path).name}")
        
        # Generate output filename
        output_path = generate_output_filename(
            input_path,
            i,
            output_dir,
            output_extension,
            prefix,
            use_input_names,
        )
        
        start_time = time.time()
        
        try:
            # Execute task
            result = task_handler.execute(
                model=model_to_load,
                input_data=input_path,
                output_path=output_path,
                **extra_kwargs
            )
            
            exec_time = time.time() - start_time
            
            if verbose:
                click.echo(f"  ✓ Success - Output: {output_path} ({exec_time:.2f}s)")
            
            results.append(BatchResult(
                input_file=input_path,
                output_file=output_path,
                success=True,
                execution_time=exec_time,
            ))
            success_count += 1
            
        except Exception as e:
            exec_time = time.time() - start_time
            
            if verbose:
                click.echo(click.style(f"  ✗ Failed: {e}", fg="red"), err=True)
            
            results.append(BatchResult(
                input_file=input_path,
                output_file=None,
                success=False,
                error=str(e),
                execution_time=exec_time,
            ))
            failure_count += 1
            
            if not continue_on_error:
                click.echo("Stopping batch processing due to error.", err=True)
                break
    
    return (results, success_count, failure_count)
