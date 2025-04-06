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
# pipx install "hftool[image]"
# pipx install "hftool[audio]"
# pipx install "hftool[http]"
# pipx install "hftool[all]" # Install all optional dependencies
```

**Using pip:**

```bash
pip install hftool
# To include optional dependencies:
# pip install "hftool[image]"
# pip install "hftool[audio]"
# pip install "hftool[http]"
# pip install "hftool[all]"
```

## Usage

The basic command structure is:

```bash
hftool --task <task_name> --input <input_data> [OPTIONS]
```

**Arguments and Options:**

*   `--task` (Required): The name of the transformers pipeline task (e.g., 'text-generation', 'image-classification', 'zero-shot-classification').
*   `--input` (Required): The input data. Can be raw text, a local file path, or a URL (requires `hftool[http]` and potentially `hftool[image]` or `hftool[audio]`).
*   `--model` (Optional): The Hugging Face model ID (e.g., 'gpt2', 'facebook/bart-large-cnn'). Defaults to the pipeline's default for the specified task.
*   `--output-json` (Optional Flag): Output the results in JSON format.
*   `--device` (Optional): Specify the device ('cpu', 'cuda', 'cuda:0', 'mps', etc.). Defaults to auto-detection by `transformers`.
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
    *(Requires `pip install "hftool[image]"`)*
    ```bash
    hftool --task image-classification --input ./my_cat.jpg --model google/vit-base-patch16-224
    ```

4.  **Image Classification (using a URL):**
    *(Requires `pip install "hftool[image,http]"`)
    ```bash
    hftool --task image-classification --input https://example.com/images/cat.jpg --model google/vit-base-patch16-224 --output-json
    ```

5.  **Automatic Speech Recognition (using a local file):**
    *(Requires `pip install "hftool[audio]"`)
    ```bash
    hftool --task automatic-speech-recognition --input ./speech.wav --model facebook/wav2vec2-base-960h
    ```

## Development

To install for development:

```bash
git clone https://github.com/placeholder/hftool # Replace with actual repo URL
cd hftool
pip install -e ".[all]" # Install in editable mode with all extras
```
