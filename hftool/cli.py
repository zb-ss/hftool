import click
import json
import os
import sys
from typing import Optional, Tuple
from urllib.parse import urlparse
import warnings

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

# --- More Optional Dependencies ---

_LIBROSA_INSTALLED = True
try:
    import librosa
except ImportError:
    _LIBROSA_INSTALLED = False

_CV2_INSTALLED = True
try:
    import cv2
except ImportError:
    _CV2_INSTALLED = False

_BITSANDBYTES_INSTALLED = True
try:
    import bitsandbytes
except ImportError:
    _BITSANDBYTES_INSTALLED = False


_DIFFUSERS_INSTALLED = True
try:
    # Diffusers often benefits from float16, requires torch
    import torch
    from diffusers import DiffusionPipeline
    # Check for accelerate for performance hints
    try:
        import accelerate
        _ACCELERATE_INSTALLED = True
    except ImportError:
        _ACCELERATE_INSTALLED = False

except ImportError:
    _DIFFUSERS_INSTALLED = False
    _ACCELERATE_INSTALLED = False # Can't be installed if diffusers isn't
    # Define dummy types if not installed to avoid NameErrors later
    class DiffusionPipeline: pass
    class torch:
        float16 = None


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

# Tasks primarily handled by the diffusers library
DIFFUSERS_TASKS = {
    "text-to-image",
    "image-to-image", # Add other diffuser tasks as needed
    "stable-diffusion", # Alias for text-to-image? Or handle specific pipelines
    "stable-diffusion-xl",
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
        # Diffusers tasks might take image files as input (e.g., image-to-image)
        if task in DIFFUSERS_TASKS and not _PIL_INSTALLED:
             click.echo(
                "Error: This diffuser task requires 'Pillow' for image file input. "
                "Install with: pip install 'hftool[diffusers]'", # diffusers extra includes Pillow
                err=True,
            )
             sys.exit(1)
        elif task in IMAGE_TASKS and not _PIL_INSTALLED:
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

    # Check for diffusers dependency if it's a diffuser task
    if task in DIFFUSERS_TASKS:
        if not _DIFFUSERS_INSTALLED:
            click.echo(
                f"Error: Task '{task}' requires the 'diffusers' library. "
                "Install with: pip install 'hftool[diffusers]'",
                err=True,
            )
            sys.exit(1)
        # Diffusers output (and some inputs) need Pillow, check again here explicitly for the task
        if not _PIL_INSTALLED:
             click.echo(
                "Error: Diffuser tasks require 'Pillow' for image handling. "
                "Install with: pip install 'hftool[diffusers]'",
                err=True,
            )
             sys.exit(1)
        # Optionally warn if accelerate isn't installed
        if not _ACCELERATE_INSTALLED:
            click.echo(
                "Warning: The 'accelerate' library is not installed. "
                "Installing it ('pip install accelerate') is recommended for diffusers performance.",
                err=True, # Use err=True for warnings too, to separate from stdout
            )

    # Note: We don't add specific checks for librosa/opencv here, as their need
    # is highly dependent on the specific model/pipeline being used internally.
    # The user should install the relevant extra if a pipeline fails due to
    # missing librosa or cv2. We *will* check for bitsandbytes if quantization
    # flags are used later.


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
@click.option(
    "--output-file",
    default=None,
    type=click.Path(dir_okay=False, writable=True), # Ensure it's a writable file path
    help="Path to save output image (required for diffuser tasks like text-to-image).",
)
@click.option(
    "--load-in-8bit",
    is_flag=True,
    default=False,
    help="Load model in 8-bit precision (requires 'bitsandbytes' and compatible hardware).",
)
@click.option(
    "--load-in-4bit",
    is_flag=True,
    default=False,
    help="Load model in 4-bit precision (requires 'bitsandbytes' and compatible hardware).",
)
@click.version_option(package_name="hftool")
@click.argument('pipeline_args', nargs=-1, type=click.UNPROCESSED) # Catch extra args for pipeline
def main(
    task: str,
    model: Optional[str],
    input_data: str,
    output_json: bool,
    device: Optional[str],
    output_file: Optional[str],
    load_in_8bit: bool,
    load_in_4bit: bool,
    pipeline_args: Tuple[str, ...],
) -> None:
    """
    A CLI for running Hugging Face transformer pipelines.

    Pass task-specific arguments after '--'. For example:

    hftool --task zero-shot-classification --input "..." --model "..." -- --candidate-labels "positive,negative"

    hftool --task text-to-image --model stabilityai/stable-diffusion-2-1-base \\
           --input "An astronaut riding a horse" --output-file astro.png -- --height 512
    """
    # Determine pipeline type early
    is_diffusers_task = task in DIFFUSERS_TASKS

    # Defer heavy imports until needed
    pipeline_module = None
    supported_tasks_dict = {}
    if not is_diffusers_task:
        try:
            pipeline_module = __import__("transformers", fromlist=["pipeline", "SUPPORTED_TASKS"])
            supported_tasks_dict = pipeline_module.SUPPORTED_TASKS
        except ImportError:
             click.echo(
                "Error: The 'transformers' library is not installed. "
                "Please install it: pip install transformers torch",
                err=True,
            )
             sys.exit(1)
    # Note: Diffusers check happens in _check_optional_deps if task is in DIFFUSERS_TASKS

    # --- Input Validation and Dependency Checks ---

    # Quantization checks
    if load_in_4bit and load_in_8bit:
        click.echo("Error: Cannot use --load-in-4bit and --load-in-8bit simultaneously.", err=True)
        sys.exit(1)
    if (load_in_4bit or load_in_8bit) and not _BITSANDBYTES_INSTALLED:
         click.echo(
            "Error: Quantization (--load-in-4bit or --load-in-8bit) requires the 'bitsandbytes' library. "
            "Install with: pip install 'hftool[with_quantization]'",
            err=True,
        )
         sys.exit(1)
    # Add warning if quantization is attempted on non-GPU devices? (bitsandbytes primarily GPU)
    # if (load_in_4bit or load_in_8bit) and (not device or 'cpu' in device):
    #     click.echo("Warning: Quantization is typically used with CUDA GPUs. Performance on CPU may vary.", err=True)


    is_url_input = _is_url(input_data)
    # Treat input as a file path if it's not a URL and the task expects file input,
    # or if it explicitly looks like a path (heuristic).
    # For diffusers, input might be text (prompt) or a file (image-to-image)
    is_file_input = not is_url_input and (
        task in FILE_INPUT_TASKS or os.path.sep in input_data or os.path.exists(input_data)
        or (is_diffusers_task and os.path.exists(input_data)) # Check existence for potential diffuser input image
    )

    if is_file_input and not os.path.exists(input_data):
        click.echo(f"Error: Input file not found: {input_data}", err=True)
        sys.exit(1)

    _check_optional_deps(is_url_input, is_file_input, task)

    # Specific check for diffuser output
    if is_diffusers_task and not output_file:
        click.echo("Error: --output-file is required for diffuser tasks.", err=True)
        sys.exit(1)
    if not is_diffusers_task and output_file:
        click.echo("Warning: --output-file is specified but the task is not a known diffuser task. It might be ignored.", err=True)


    # --- Parse Pipeline Arguments ---
    # Convert the flat tuple from click into a dictionary for pipeline kwargs
    # Assumes arguments like --key value or --flag
    # This is a basic parser; more complex argument structures might need refinement.
    pipeline_kwargs = {}
    i = 0
    # Map --input to 'prompt' for convenience in text-to-image, if not already set in pipeline_args
    if is_diffusers_task and task == "text-to-image" and "prompt" not in pipeline_args:
        pipeline_kwargs['prompt'] = input_data
        click.echo(f"Mapping --input to 'prompt': \"{input_data}\"", err=True)
    # TODO: Add similar logic for other diffuser tasks if needed (e.g., map --input file to 'image' for image-to-image)

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
    pipe = None
    try:
        click.echo(f"Loading pipeline for task: {task}" + (f", model: {model}" if model else "") + "...", err=True)

        # Determine device string (common for both libraries now)
        resolved_device = device # Pass user input directly, libraries handle 'cuda', 'cuda:0', 'mps', 'cpu'
        if not resolved_device:
            # Basic auto-detection hint, actual detection is done by the library
            # Check CUDA availability using torch if diffusers or transformers are installed
            cuda_available = False
            mps_available = False
            if _DIFFUSERS_INSTALLED or pipeline_module:
                 try:
                     import torch # Import torch here if needed for device check
                     cuda_available = torch.cuda.is_available()
                     if hasattr(torch.backends, 'mps'):
                         mps_available = torch.backends.mps.is_available()
                 except ImportError:
                     click.echo("Warning: torch not found, cannot auto-detect CUDA/MPS device.", err=True)


            if cuda_available:
                resolved_device = "cuda"
            elif mps_available:
                resolved_device = "mps"
            else:
                resolved_device = "cpu" # Default fallback
            click.echo(f"Auto-detected device: {resolved_device}", err=True)


        if is_diffusers_task:
            # Load Diffusers Pipeline
            if not _DIFFUSERS_INSTALLED: # Should have been caught earlier, but double-check
                 raise ImportError("Diffusers library is required but not installed.")
            if not model:
                click.echo(f"Error: --model is required for diffuser task '{task}'.", err=True)
                sys.exit(1)

            # Add torch_dtype for performance if using GPU and torch is available
            dtype_kwargs = {}
            if resolved_device != "cpu" and hasattr(torch, "float16"): # Check if float16 exists
                dtype_kwargs["torch_dtype"] = torch.float16
                click.echo("Using torch.float16 for diffuser pipeline.", err=True)

            # Add quantization arguments if specified
            quantization_kwargs = {}
            if load_in_8bit:
                quantization_kwargs["load_in_8bit"] = True
                click.echo("Loading model in 8-bit.", err=True)
            elif load_in_4bit:
                 quantization_kwargs["load_in_4bit"] = True
                 click.echo("Loading model in 4-bit.", err=True)


            # Suppress safety checker warning if needed (example)
            # Users can override this by passing --safety-checker <path_or_None>
            if "safety_checker" not in pipeline_kwargs:
                 # Setting to None might be deprecated, handle potential warnings or future changes.
                 # pipeline_kwargs["safety_checker"] = None
                 # Consider requiring explicit --safety-checker None argument from user if they want to disable it.
                 click.echo("Note: Safety checker not specified, using default.", err=True)


            pipe = DiffusionPipeline.from_pretrained(
                model,
                **dtype_kwargs,
                **quantization_kwargs, # Add quantization args
                **pipeline_kwargs # Pass extra arguments like height, width, guidance_scale
            )
            # Move pipeline to the specified device (unless quantization handles it)
            # Note: bitsandbytes might handle device placement for quantized models
            if not load_in_8bit and not load_in_4bit:
            pipe.to(resolved_device)

        else:
            # Load Transformers Pipeline
            if not pipeline_module: # Should have exited earlier if import failed
                raise RuntimeError("Transformers module not loaded correctly.")

            # Add quantization arguments if specified
            quantization_kwargs = {}
            if load_in_8bit:
                quantization_kwargs["load_in_8bit"] = True
                click.echo("Loading model in 8-bit.", err=True)
            elif load_in_4bit:
                 quantization_kwargs["load_in_4bit"] = True
                 click.echo("Loading model in 4-bit.", err=True)
            # Note: Transformers pipeline might take device_map="auto" with quantization
            if load_in_8bit or load_in_4bit:
                quantization_kwargs["device_map"] = "auto"
                click.echo("Using device_map='auto' for quantized model.", err=True)
                resolved_device = None # Let device_map handle placement

            # Transformers pipeline handles device string directly in newer versions
            pipe = pipeline_module.pipeline(
                task=task,
                model=model, # Can be None
                device=resolved_device, # Pass the string 'cuda', 'mps', 'cpu' (ignored if device_map is used)
                framework="pt", # Explicitly request PyTorch
                **quantization_kwargs, # Add quantization args
                **pipeline_kwargs # Pass extra arguments like candidate_labels
            )

        click.echo("Pipeline loaded.", err=True)

    except ImportError as e:
        # Catch specific import errors related to missing task dependencies
        click.echo(f"Error: Missing dependency for task '{task}'. {e}", err=True)
        click.echo("You might need to install additional libraries (e.g., 'pip install \"hftool[diffusers]\"' or task-specific ones).", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error creating pipeline: {e}", err=True)
        # More specific error checking
        if not is_diffusers_task and task not in supported_tasks_dict:
             click.echo(f"Note: Task '{task}' is not in the explicitly known transformers list. Ensure it's valid.", err=True)
             click.echo(f"Supported transformers tasks list (may be incomplete): {list(supported_tasks_dict.keys())}", err=True)
        # Add similar check for known diffusers tasks if desired
        sys.exit(1)

    # --- Inference ---
    results = None
    try:
        click.echo("Running inference...", err=True)
        if is_diffusers_task:
            # Diffusers pipelines typically take keyword arguments
            # Use the parsed pipeline_kwargs which includes the mapped --input as 'prompt' if applicable
            with warnings.catch_warnings(): # Suppress potential future warnings from diffusers/torch
                warnings.simplefilter("ignore")
                # Ensure prompt is passed if not already in kwargs (e.g. if user used --prompt explicitly)
                if 'prompt' not in pipeline_kwargs and task == "text-to-image":
                    pipeline_kwargs['prompt'] = input_data

                # Handle input for image-to-image task
                if task == "image-to-image":
                    if 'image' not in pipeline_kwargs: # Check if user passed --image explicitly
                         if is_file_input:
                             if not _PIL_INSTALLED: raise ImportError("Pillow is required for image input.")
                             from PIL import Image # Requires Pillow
                             pipeline_kwargs['image'] = Image.open(input_data)
                             click.echo(f"Loading input image from file: {input_data}", err=True)
                         elif is_url_input:
                              if not _PIL_INSTALLED: raise ImportError("Pillow is required for image input.")
                              if not _REQUESTS_INSTALLED: raise ImportError("Requests is required for URL input.")
                              import requests
                              from PIL import Image
                              from io import BytesIO
                              response = requests.get(input_data, stream=True)
                              response.raise_for_status()
                              pipeline_kwargs['image'] = Image.open(BytesIO(response.content))
                              click.echo(f"Loading input image from URL: {input_data}", err=True)
                         else:
                              click.echo("Error: image-to-image task requires an image file path or URL via --input, or --image argument.", err=True)
                              sys.exit(1)

                    if 'prompt' not in pipeline_kwargs: # Prompt often still needed/useful for img2img
                        click.echo("Warning: 'prompt' not provided for image-to-image task via --input or --prompt.", err=True)


                results = pipe(**pipeline_kwargs) # Pass kwargs directly

        else:
            # Transformers pipeline often takes positional input
            results = pipe(input_data) # Pass the original input_data string

        click.echo("Inference complete.", err=True)

    except Exception as e:
        click.echo(f"Error during inference: {e}", err=True)
        # Add more specific error handling if needed (e.g. OOM errors)
        sys.exit(1)

    # --- Output ---
    if is_diffusers_task:
        # Handle diffuser output (typically images)
        if hasattr(results, 'images') and isinstance(results.images, list) and len(results.images) > 0:
            img = results.images[0] # Get the first image
            try:
                # Ensure Pillow is available (should be checked earlier)
                if not _PIL_INSTALLED: raise ImportError("Pillow is required to save the image.")
                from PIL import Image
                if not isinstance(img, Image.Image):
                     click.echo("Error: Diffuser pipeline did not return a PIL Image.", err=True)
                     sys.exit(1)

                img.save(output_file)
                click.echo(f"Output image saved to: {output_file}", err=False) # Print to stdout on success
                # Optionally save more images if output_file suggests a pattern or directory?
            except ImportError as e:
                 click.echo(f"Error saving image: Missing required library. {e}", err=True)
                 sys.exit(1)
            except Exception as e:
                click.echo(f"Error saving image to {output_file}: {e}", err=True)
                sys.exit(1)
        else:
            click.echo("Warning: Diffuser pipeline did not return the expected image output format.", err=True)
            # Print raw results if not image?
            if output_json: print(json.dumps(str(results), indent=2)) # Best effort JSON
            else: print(results)

    else:
        # Handle transformers output (text/json)
        if output_json:
            try:
                # Attempt to serialize directly; handle potential non-serializable objects
                # Convert tensors/numpy arrays if needed (basic example)
                def default_serializer(obj):
                    # Use isinstance checks for better type safety if torch/numpy are potentially loaded
                    if _DIFFUSERS_INSTALLED and isinstance(obj, torch.Tensor):
                         return obj.cpu().numpy().tolist() # Ensure tensor is on CPU before converting
                    # Add numpy check if numpy could be a direct dependency or part of results
                    # elif isinstance(obj, np.ndarray):
                    #    return obj.tolist()
                    try:
                        # Attempt standard JSON serialization first
                        json.dumps(obj)
                        return obj
                    except TypeError:
                         # Fallback to string if standard serialization fails
                         return str(obj)


                json_output = json.dumps(results, indent=2, default=default_serializer)
                print(json_output)
            except TypeError as e:
                click.echo(f"Error: Could not serialize results to JSON: {e}", err=True)
                # Fallback to simple print if JSON fails
                print(results)
        else:
            # Pretty print for common result structures
            if isinstance(results, list):
                # Handle list of dicts common in HF pipelines
                if all(isinstance(item, dict) for item in results):
                     for item in results:
                         print(json.dumps(item, indent=2, default=str)) # Pretty print each dict
                         print("-" * 20) # Separator
                else:
                    for item in results:
                        print(item)
            elif isinstance(results, dict):
                 # Pretty print the dict using JSON
                 print(json.dumps(results, indent=2, default=str))
            else:
                print(results)


if __name__ == "__main__":
    main()
