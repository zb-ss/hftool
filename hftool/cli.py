import click
import json
import os
import sys
from typing import Optional, List, Any, Tuple
from urllib.parse import urlparse

# --- Optional Dependency Handling ---

_PIL_INSTALLED = True
try:
    from PIL import Image
except ImportError:
    _PIL_INSTALLED = False

_SOUNDFILE_INSTALLED = True
try:
    import soundfile as sf
except ImportError:
    _SOUNDFILE_INSTALLED = False

_REQUESTS_INSTALLED = True
try:
    import requests
except ImportError:
    _REQUESTS_INSTALLED = False

# --- Constants ---

# Heuristic list of tasks that typically require file/URL input that isn't just text.
# This helps guide input validation and dependency checks.
FILE_INPUT_TASKS = {
    "audio-classification",
    "automatic-speech-recognition",
    "image-classification",
    "image-segmentation",
    "image-to-text",
    "object-detection",
    "zero-shot-image-classification",
    "zero-shot-object-detection",
    # Add other relevant tasks here
}

AUDIO_TASKS = {
    "audio-classification",
    "automatic-speech-recognition",
    # Add other audio tasks
}

IMAGE_TASKS = {
    "image-classification",
    "image-segmentation",
    "image-to-text",
    "object-detection",
    "zero-shot-image-classification",
    "zero-shot-object-detection",
    # Add other image tasks
}

# --- Helper Functions ---

def _is_url(path: str) -> bool:
    """Checks if a string is a valid URL."""
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def _check_optional_deps(is_url_input: bool, is_file_input: bool, task: str) -> None:
    """Checks if required optional dependencies are installed based on input type and task."""
    if is_url_input:
        if not _REQUESTS_INSTALLED:
            click.echo(
                "Error: URL input requires the 'requests' library. "
                "Install with: pip install 'hftool[http]'",
                err=True,
            )
            sys.exit(1)
        if task in IMAGE_TASKS and not _PIL_INSTALLED:
             click.echo(
                "Error: Image URL input requires the 'Pillow' library. "
                "Install with: pip install 'hftool[image,http]'",
                err=True,
            )
             sys.exit(1)
        # Add check for audio URLs if needed, requires soundfile/librosa + requests
        # elif task in AUDIO_TASKS and not _SOUNDFILE_INSTALLED: ...

    elif is_file_input:
        if task in IMAGE_TASKS and not _PIL_INSTALLED:
            click.echo(
                "Error: Image file input requires the 'Pillow' library. "
                "Install with: pip install 'hftool[image]'",
                err=True,
            )
            sys.exit(1)
        elif task in AUDIO_TASKS and not _SOUNDFILE_INSTALLED:
            click.echo(
                "Error: Audio file input requires the 'soundfile' library. "
                "Install with: pip install 'hftool[audio]'",
                err=True,
            )
            sys.exit(1)


# --- Click Command ---

# Use context_settings to allow forwarding arguments after '--'
CONTEXT_SETTINGS = dict(ignore_unknown_options=True, allow_extra_args=True)

@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--task",
    required=True,
    type=str,
    help="The name of the Hugging Face pipeline task (e.g., 'text-generation').",
)
@click.option(
    "--model",
    default=None,
    type=str,
    help="Optional Hugging Face model ID (e.g., 'gpt2'). Defaults to task default.",
)
@click.option(
    "--input",
    "input_data", # Use a different variable name to avoid conflict with builtin 'input'
    required=True,
    type=str,
    help="Input data: raw text, a file path, or a URL.",
)
@click.option(
    "--output-json",
    is_flag=True,
    default=False,
    help="Output results in JSON format.",
)
@click.option(
    "--device",
    default=None, # Let transformers handle default device selection unless specified
    type=str,
    help="Device for inference (e.g., 'cpu', 'cuda', 'cuda:0', 'mps').",
)
@click.version_option(package_name="hftool")
@click.argument('pipeline_args', nargs=-1, type=click.UNPROCESSED) # Catch extra args for pipeline
def main(
    task: str,
    model: Optional[str],
    input_data: str,
    output_json: bool,
    device: Optional[str],
    pipeline_args: Tuple[str, ...],
) -> None:
    """
    A CLI for running Hugging Face transformer pipelines.

    Pass task-specific arguments after '--'. For example:

    hftool --task zero-shot-classification --input "..." --model "..." -- --candidate-labels "positive,negative"
    """
    try:
        # Import transformers here to make CLI startup faster when just showing help
        from transformers import pipeline
        from transformers.pipelines import SUPPORTED_TASKS

    except ImportError:
        click.echo(
            "Error: The 'transformers' library is not installed. "
            "Please install it: pip install transformers torch",
            err=True,
        )
        sys.exit(1)

    # --- Input Validation and Dependency Checks ---
    is_url_input = _is_url(input_data)
    # Treat input as a file path if it's not a URL and the task expects file input,
    # or if it explicitly looks like a path (heuristic).
    is_file_input = not is_url_input and (
        task in FILE_INPUT_TASKS or os.path.sep in input_data or os.path.exists(input_data)
    )

    if is_file_input and not os.path.exists(input_data):
        click.echo(f"Error: Input file not found: {input_data}", err=True)
        sys.exit(1)

    _check_optional_deps(is_url_input, is_file_input, task)

    # --- Parse Pipeline Arguments ---
    # Convert the flat tuple from click into a dictionary for pipeline kwargs
    # Assumes arguments like --key value or --flag
    # This is a basic parser; more complex argument structures might need refinement.
    pipeline_kwargs = {}
    i = 0
    while i < len(pipeline_args):
        arg = pipeline_args[i]
        if arg.startswith("--"):
            key = arg[2:].replace("-", "_") # Convert --candidate-labels to candidate_labels
            if i + 1 < len(pipeline_args) and not pipeline_args[i+1].startswith("--"):
                # Argument with a value
                value_str = pipeline_args[i+1]
                # Attempt to parse value as JSON (list, dict, bool, number) or keep as string
                try:
                    pipeline_kwargs[key] = json.loads(value_str)
                except json.JSONDecodeError:
                    # Handle simple comma-separated lists common in HF tasks
                    if ',' in value_str:
                         pipeline_kwargs[key] = [item.strip() for item in value_str.split(',')]
                    else:
                         pipeline_kwargs[key] = value_str # Keep as string if not JSON/list
                i += 2
            else:
                # Boolean flag
                pipeline_kwargs[key] = True
                i += 1
        else:
            click.echo(f"Warning: Ignoring unexpected pipeline argument: {arg}", err=True)
            i += 1


    # --- Pipeline Creation ---
    try:
        click.echo(f"Loading pipeline for task: {task}" + (f", model: {model}" if model else "") + "...", err=True)
        # Determine device ID: -1 for auto (CPU/GPU), >= 0 for specific GPU
        device_id = -1
        resolved_device = None
        if device:
            if device == "cpu":
                device_id = -1
                resolved_device = "cpu"
            elif device.startswith("cuda"):
                 try:
                     if ":" in device:
                         device_id = int(device.split(":")[1])
                     else:
                         device_id = 0 # Default to cuda:0 if just 'cuda'
                     resolved_device = f"cuda:{device_id}"
                 except (ValueError, IndexError):
                     click.echo(f"Error: Invalid cuda device format: {device}. Use 'cuda' or 'cuda:N'.", err=True)
                     sys.exit(1)
            elif device == "mps":
                # Transformers pipeline handles 'mps' directly via device argument
                 resolved_device = "mps"
                 device_id = None # Let pipeline handle MPS mapping
            else:
                 click.echo(f"Warning: Unsupported device '{device}'. Letting transformers auto-detect.", err=True)
                 resolved_device = None
                 device_id = -1 # Fallback to auto-detect

        # Pass device_id for older transformers compatibility, device for newer
        pipe = pipeline(
            task=task,
            model=model, # Can be None, pipeline handles default
            device=resolved_device, # Use resolved device string for newer transformers
            # device=device_id if device_id is not None else -1, # Use device_id for older transformers if needed
            framework="pt", # Explicitly request PyTorch
            **pipeline_kwargs # Pass extra arguments like candidate_labels
        )
        click.echo("Pipeline loaded.", err=True)

    except ImportError as e:
         # Catch specific import errors related to missing task dependencies
         click.echo(f"Error: Missing dependency for task '{task}'. {e}", err=True)
         click.echo("You might need to install additional libraries.", err=True)
         sys.exit(1)
    except Exception as e:
        click.echo(f"Error creating pipeline: {e}", err=True)
        # More specific error checking for model not found, task not supported etc. could be added
        # Example: Check if task is in SUPPORTED_TASKS
        if task not in SUPPORTED_TASKS:
             click.echo(f"Note: Task '{task}' is not in the explicitly known list. Ensure it's valid.", err=True)
             click.echo(f"Supported tasks list (may be incomplete): {list(SUPPORTED_TASKS.keys())}", err=True)
        sys.exit(1)

    # --- Inference ---
    try:
        click.echo("Running inference...", err=True)
        # The pipeline function handles different input types (text, file paths, URLs if supported by task)
        results = pipe(input_data) # Pass the original input_data string
        click.echo("Inference complete.", err=True)

    except Exception as e:
        click.echo(f"Error during inference: {e}", err=True)
        sys.exit(1)

    # --- Output ---
    if output_json:
        try:
            # Attempt to serialize directly; handle potential non-serializable objects if necessary
            json_output = json.dumps(results, indent=2, default=str) # Use default=str as fallback
            print(json_output)
        except TypeError as e:
            click.echo(f"Error: Could not serialize results to JSON: {e}", err=True)
            # Fallback to simple print if JSON fails
            print(results)
    else:
        # Pretty print for common result structures
        if isinstance(results, list):
            for item in results:
                print(item)
        elif isinstance(results, dict):
             for key, value in results.items():
                 print(f"{key}: {value}")
        else:
            print(results)


if __name__ == "__main__":
    main()
