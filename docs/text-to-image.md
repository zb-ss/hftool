# Text-to-Image

> [Back to README](../README.md) | [All Tasks](../README.md#supported-tasks)

Generate images with Z-Image-Turbo (state-of-the-art open-source model):

```bash
# Basic usage (uses default model)
hftool -t t2i -i "A cat wearing a space helmet" -o cat_space.png

# With specific model
hftool -t t2i -m Tongyi-MAI/Z-Image-Turbo \
       -i "A photorealistic sunset over mountains" \
       -o sunset.png

# With custom parameters (Z-Image-Turbo uses 9 steps, guidance_scale=0)
hftool -t t2i -m Tongyi-MAI/Z-Image-Turbo \
       -i "A renaissance painting of a robot" \
       -o robot.png \
       -- --num_inference_steps 9 --guidance_scale 0.0 --height 1024 --width 1024
```

**Other supported models:**
- `stabilityai/stable-diffusion-xl-base-1.0`
- `black-forest-labs/FLUX.1-schnell`
