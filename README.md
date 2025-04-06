# hftool

A command-line interface for interacting with Hugging Face transformer models via the `pipeline` function.

## Description

`hftool` provides a simple way to run Hugging Face `transformers` pipelines directly from your terminal. You can specify the task, model, and input data (text, file path, or URL) and get the inference results printed to standard output.

## Installation

You can install `hftool` using `pip` or `pipx` (recommended for CLI tools).

**Using pipx (Recommended):**

```bash
pipx install hftool
# To include optional dependencies for image, audio, or URL handling:
# pipx install "hftool[with_image]"
# pipx install "hftool[with_audio]"
# pipx install "hftool[with_audio_advanced]" # Includes librosa, scipy
# pipx install "hftool[with_http]"
# pipx install "hftool[with_diffusers]" # For text-to-image, etc. (includes Pillow, accelerate)
# pipx install "hftool[with_vision_advanced]" # Includes opencv-python-headless
# pipx install "hftool[with_quantization]" # Includes bitsandbytes (Linux/CUDA often required)
# pipx install "hftool[all]" # Install all optional dependencies
```

**Using pip:**

```bash
pip install hftool
# To include optional dependencies:
# pip install "hftool[with_image]"
# pip install "hftool[with_audio]"
# pip install "hftool[with_audio_advanced]"
# pip install "hftool[with_http]"
# pip install "hftool[with_diffusers]"
# pip install "hftool[with_vision_advanced]"
# pip install "hftool[with_quantization]"
# pip install "hftool[all]"
```

## Usage

The basic command structure is:

```bash
hftool --task <task_name> --input <input_data> [OPTIONS]
```

**Arguments and Options:**

*   `--task` (Required): The name of the transformers or diffusers pipeline task (e.g., 'text-generation', 'image-classification', 'text-to-image').
*   `--input` (Required): The input data. Can be raw text (e.g., prompt for text-generation/text-to-image), a local file path (e.g., for image-classification, image-to-image), or a URL (requires `hftool[with_http]`, plus potentially `hftool[with_image]` for image URLs or `hftool[with_audio]` for audio URLs).
*   `--model` (Optional): The Hugging Face model ID (e.g., 'gpt2', 'stabilityai/stable-diffusion-2-1-base'). Defaults to the pipeline's default for `transformers` tasks; **required** for `diffusers` tasks.
*   `--output-json` (Optional Flag): Output results in JSON format (primarily for `transformers` tasks).
*   `--output-file` (Optional): Path to save output image (required for image-generating `diffusers` tasks like `text-to-image`).
*   `--device` (Optional): Specify the device ('cpu', 'cuda', 'cuda:0', 'mps', etc.). Defaults to auto-detection. Ignored if using quantization flags.
*   `--load-in-8bit` (Optional Flag): Load the model using 8-bit quantization (requires `hftool[with_quantization]` and compatible hardware, usually Linux/NVIDIA GPU). Reduces memory usage.
*   `--load-in-4bit` (Optional Flag): Load the model using 4-bit quantization (requires `hftool[with_quantization]` and compatible hardware). Further reduces memory usage. Cannot be used with `--load-in-8bit`.
*   `--help`: Show the help message.

**Examples:**

1.  **Text Generation:**
    ```bash
    hftool --task text-generation --input "Hello, my name is" --model gpt2
    ```

2.  **Zero-Shot Classification:**
    ```bash
    hftool --task zero-shot-classification \
           --input "This is a great tool for NLP tasks." \
           --model facebook/bart-large-mnli \
           --candidate-labels "positive,negative,neutral" # Task-specific args passed after '--'
    ```
    *(Note: Task-specific arguments like `--candidate-labels` are passed directly to the pipeline after a `--` separator, which is standard practice for `click` to handle arbitrary extra arguments.)*


3.  **Image Classification (using a local file):**
    *(Requires `pip install "hftool[with_image]"`)*
    ```bash
    hftool --task image-classification --input ./my_cat.jpg --model google/vit-base-patch16-224
    ```

4.  **Image Classification (using a URL):**
    *(Requires `pip install "hftool[with_image,with_http]"`)
    ```bash
    hftool --task image-classification --input https://example.com/images/cat.jpg --model google/vit-base-patch16-224 --output-json
    ```

5.  **Automatic Speech Recognition (using a local file):**
    *(Requires `pip install "hftool[with_audio]"`)
    ```bash
    hftool --task automatic-speech-recognition --input ./speech.wav --model facebook/wav2vec2-base-960h
    ```

6.  **Text-to-Image Generation (using Stable Diffusion):**
    *(Requires `pip install "hftool[with_diffusers]"`. May need significant RAM/VRAM).*
    ```bash
 # device can be rocm for amd users or if you want to use a specific card then rocm:0
    hftool --task text-to-image \
           --model stabilityai/stable-diffusion-2-1-base \
           --input "A renaissance painting of a cat riding a bicycle" \
           --output-file cat_bicycle.png \
           --device cuda \ 
           -- --height 768 --width 768 --guidance_scale 9 # Pass diffuser-specific args after --
    ```
    *(Note: The first run will download the model, which can be large).*

7.  **Text Generation with 4-bit Quantization:**
    *(Requires `pip install "hftool[with_quantization]"`. Best on Linux with NVIDIA GPU).*
    ```bash
    hftool --task text-generation \
           --model meta-llama/Meta-Llama-3-8B \
           --input "Explain the benefits of quantization in large language models:" \
           --load-in-4bit \
           -- --max_new_tokens 250 # Pass generation args after --
    ```
    *(Note: Access to models like Llama 3 may require Hugging Face login/token).*


## Development

To install for development:

```bash
git clone https://github.com/zashboy-websites/hftool # Replace with actual repo URL
cd hftool
pip install -e ".[all]" # Install in editable mode with all extras
```
