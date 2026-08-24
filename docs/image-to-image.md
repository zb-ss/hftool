# Image-to-Image

> [Back to README](../README.md) | [All Tasks](../README.md#supported-tasks)

Transform existing images with Qwen Image Edit (default), FLUX.2 Klein, or SDXL:

```bash
# Basic image editing with Qwen Image Edit (default)
hftool -t i2i \
       -i '{"image": "photo.jpg", "prompt": "turn this into a watercolor painting"}' \
       -o watercolor.png

# Multi-image editing - combine multiple images (Qwen feature)
hftool -t i2i \
       -i '{"image": ["person1.jpg", "person2.jpg"], "prompt": "Both people standing together in a park"}' \
       -o combined.png

# With custom parameters
hftool -t i2i \
       -i '{"image": "portrait.jpg", "prompt": "as a Renaissance painting"}' \
       -o renaissance.png \
       -- --seed 42 --true_cfg_scale 4.0 --num_inference_steps 50

# FLUX.2 Klein 4B - commercial-friendly four-step multi-reference editing
hftool -t i2i -m flux2-klein-4b \
       -i '{"image": "person.jpg", "prompt": "the person from image 1 as an astronaut on Mars"}' \
       -o astronaut.png

# FLUX.2 Klein with multiple reference images
hftool -t i2i -m flux2-klein-4b \
       -i '{"image": ["cat.jpg", "dog.jpg"], "prompt": "the cat from image 1 and dog from image 2 playing together"}' \
       -o pets.png \
       -- --seed 42

# Style transfer with SDXL Refiner (smaller model, faster)
hftool -t i2i -m sdxl-refiner \
       -i '{"image": "landscape.jpg", "prompt": "professional photography, enhanced colors"}' \
       -o enhanced.png \
       -- --strength 0.3
```

**Supported models:**
- `qwen-image-edit` (default, ~57 GB download) - Quality editing, character consistency, multi-image support
- `flux2-klein-4b` (~23.7 GB download) - Four-step unified editing, Apache-2.0, 13/16 GB minimum/recommended VRAM
- `flux2-klein-9b` (~29 GB download) - Gated four-step editing, non-commercial license
- `flux2-klein` - Backward-compatible alias for `flux2-klein-9b`; use the explicit 4B name for the commercial model
- `stabilityai/stable-diffusion-xl-refiner-1.0` (6.2 GB) - Fast refinement and subtle changes
- `stabilityai/stable-diffusion-xl-base-1.0` (6.5 GB) - Stronger style transfer

**Input format:** JSON with `image` (path or list of paths) and `prompt` (edit description)

**Qwen Image Edit parameters** (pass after `--`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--seed` | random | Random seed for reproducibility |
| `--true_cfg_scale` | 4.0 | True CFG scale (higher = stronger prompt adherence) |
| `--num_inference_steps` | 40 | Number of denoising steps |
| `--guidance_scale` | 1.0 | Standard CFG guidance scale |
| `--negative_prompt` | " " | What to avoid in generation |

**FLUX.2 Klein parameters** (pass after `--`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--seed` | random | Random seed for reproducibility |
| `--num_inference_steps` | 4 | Number of denoising steps (optimized for 4) |
| `--guidance_scale` | 1.0 | CFG guidance scale |
| `--height` | 1024 | Output image height |
| `--width` | 1024 | Output image width |

**FLUX.2 Klein tips:**
- Reference images in prompts using "image 1", "image 2", etc.
- Supports up to 10 reference images per generation
- The 4B entry is Apache-2.0 and targets consumer GPUs.
- The 9B entry is gated and non-commercial; accept its terms before download.
- Current image pipelines require Diffusers 0.40 or newer.

**SDXL Refiner/Base parameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--seed` | random | Random seed for reproducibility |
| `--strength` | 0.3-0.7 | How much to change the image (0.0-1.0) |
| `--num_inference_steps` | 30 | Number of denoising steps |
| `--guidance_scale` | 7.5 | CFG guidance scale |

**Qwen Image Edit features:**
- Character consistency: Preserves identity in imaginative edits
- Multi-image input: Combine multiple images into one scene
- Industrial design: Batch product design and material replacement
- Geometric reasoning: Generate auxiliary construction lines

**Memory requirements:** use `hftool info <model>` or the TUI for the catalog
minimum/recommended values. The Qwen checkpoint is large; use explicit multi-GPU
or CPU offload when one card lacks headroom:

```bash
# Use multi-GPU (distributes across available GPUs)
hftool -t i2i -i '{"image": "photo.jpg", "prompt": "..."}' -o out.png --gpu all

# Use CPU offload (slower but works on 16-24GB GPUs)
HFTOOL_CPU_OFFLOAD=1 hftool -t i2i -i '{"image": "photo.jpg", "prompt": "..."}' -o out.png

# Use sequential CPU offload (most memory efficient, slowest)
HFTOOL_CPU_OFFLOAD=2 hftool -t i2i -i '{"image": "photo.jpg", "prompt": "..."}' -o out.png
```

**Note:** Current image models require diffusers >= 0.40.0. Upgrade with:
```bash
pip install --upgrade 'diffusers>=0.40.0'
# Or for pipx:
pipx runpip hftool install --upgrade 'diffusers>=0.40.0'
```
