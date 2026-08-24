# Other Tasks

> [Back to README](../README.md) | [All Tasks](../README.md#supported-tasks)

## Text Generation (LLMs)

Run language models:

```bash
# Basic generation
hftool -t llm -m meta-llama/Llama-3.2-1B-Instruct \
       -i "Explain quantum computing in simple terms:" \
       -o response.txt \
       -- --max_new_tokens 200
```

---

## Other Tasks

```bash
# Image Classification
hftool -t image-classification -m google/vit-base-patch16-224 \
       -i photo.jpg -o result.json

# Object Detection
hftool -t object-detection -m facebook/detr-resnet-50 \
       -i street.jpg -o detections.json

# Summarization
hftool -t summarization -m facebook/bart-large-cnn \
       -i article.txt -o summary.txt

# Translation
hftool -t translation -m Helsinki-NLP/opus-mt-en-de \
       -i "Hello, how are you?" -o translation.txt
```
