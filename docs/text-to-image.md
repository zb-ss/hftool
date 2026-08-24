# Text-to-Image

> [Back to README](../README.md) | [Model Management](models.md)

The recommended path is `hftool tui`: choose Text-to-Image, compare the
catalog status, license, download size, and VRAM requirements, then generate.
The CLI remains available for reproducible and automated work.

```bash
# Default: Z-Image Turbo, 8 steps, guidance 0
hftool -t t2i -i "A cat wearing a space helmet" -o cat.png

# Consumer-GPU interactive generation: Apache-2.0, four steps
hftool -t t2i -m flux2-klein-4b -i "A cinematic workshop" -o workshop.png

# Very fast drafts and backgrounds: two-step catalog default
hftool -t t2i -m sana-sprint-1.6b -i "Soft geometric background" -o background.png

# Qwen 2512 with the exact four-step BF16 Lightning adapter profile
hftool -t t2i -m qwen-image-2512-lightning \
  -i "Book cover titled OPEN HORIZONS" -o cover.png

# Inspect defaults, download state, and live GPU headroom first
hftool -t t2i -m flux2-klein-4b -i "test" --dry-run
```

## Current catalog

| Short name | Status | Intended use | Min / recommended VRAM | License |
|------------|--------|--------------|------------------------|---------|
| `z-image-turbo` | default | Fast, high-quality general generation | 16 / 18 GB | Apache-2.0 |
| `z-image` | standard | CFG, negative prompts, creative control | 16 / 18 GB | Apache-2.0 |
| `flux2-klein-4b` | recommended | Four-step interactive generation | 13 / 16 GB | Apache-2.0 |
| `sana-sprint-1.6b` | recommended | One-to-four-step drafts/backgrounds | 10 / 12 GB | Apache-2.0 + Gemma terms |
| `qwen-image-2512` | recommended | People, detail, composition, rendered text | 24 / 32 GB | Apache-2.0 |
| `qwen-image-2512-lightning` | recommended profile | Faster typography/cover drafts | 24 / 32 GB | Apache-2.0 |
| `sdxl`, `flux-schnell`, `flux-dev` | legacy | Existing ecosystem compatibility | varies | see `hftool info` |

The Qwen Lightning entry is a profile, not a separate base checkpoint. Downloading
it fetches `Qwen/Qwen-Image-2512` plus the exact declared adapter file from
`lightx2v/Qwen-Image-2512-Lightning`. Download state is complete only when both
assets are present.

## Authoritative defaults

- Z-Image Turbo: 8 steps, guidance 0, BF16, 1024×1024.
- Z-Image: 50 steps, guidance 4, `cfg_normalization=false`, BF16, 1024×1024.
- FLUX.2 Klein 4B: 4 steps, guidance 1, BF16, 1024×1024.
- SANA Sprint 1.6B: 2 steps, guidance 0, BF16, 1024×1024.
- Qwen 2512 Lightning: 4 steps, true CFG 1, the catalog scheduler, BF16.

Override inference settings after `--`; the output metadata records catalog
provenance and the effective generation values.

```bash
hftool -t t2i -m z-image -i "A misty coast" -o coast.png -- \
  --num_inference_steps 40 --guidance_scale 3.5 --seed 42
```

Native PyTorch SDPA is the default attention path. Experimental attention
backends and `torch.compile` are opt-in through environment configuration so a
failed optimization can fall back without changing model semantics.
