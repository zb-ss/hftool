# Text-to-Video

> [Back to README](../README.md) | [All Tasks](../README.md#supported-tasks)

Generate videos with LTX-2, HunyuanVideo, or other models:

```bash
# LTX-2 (fast, high quality - requires diffusers main branch)
hftool -t t2v -m ltx2 -i "A cat playing with a ball in slow motion" -o cat.mp4

# LTX-2 Image-to-Video (animate an image)
hftool -t i2v -m ltx2-i2v \
       -i '{"image": "photo.jpg", "prompt": "The person waves hello"}' \
       -o animated.mp4

# HunyuanVideo-1.5 (480p, ~2.5 second video)
hftool -t t2v -i "A person walking on a beach at sunset" -o beach.mp4

# With specific model and parameters
hftool -t t2v -m hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \
       -i "A timelapse of clouds moving over a city" \
       -o clouds.mp4 \
       -- --num_frames 61 --num_inference_steps 30

# HunyuanVideo Image-to-Video
hftool -t i2v -m hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v \
       -i '{"image": "photo.jpg", "prompt": "The person waves hello"}' \
       -o animated.mp4
```

**Supported models:**
- `Lightricks/LTX-2` (ltx2, ltx2-i2v) - Fast, high quality. Requires diffusers main branch
- `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v` - High quality 480p
- `THUDM/CogVideoX-5b`
- `Wan-AI/Wan2.1-T2V-1.3B`

**LTX-2 parameters** (pass after `--`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--seed` | random | Random seed for reproducibility |
| `--num_inference_steps` | 50 | Number of denoising steps |
| `--guidance_scale` | 3.0 | CFG guidance scale |
| `--num_frames` | 97 | Number of frames to generate |
| `--height` | 512 | Video height (must be divisible by 32) |
| `--width` | 768 | Video width (must be divisible by 32) |

**Note:** Requires system `ffmpeg` for video encoding. LTX-2 requires diffusers from main branch (auto-installed on first use).
