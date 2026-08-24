# Voiceover Pipeline

> [Back to README](../README.md) | [All Tasks](../README.md#supported-tasks)

Generate professional voiceovers from timed scripts — the full pipeline handles TTS generation per segment, audio merging with loudness normalization, and optional video muxing.

## Step-by-Step Guide

**Step 1: Install dependencies**

```bash
# Install the voiceover extras + your preferred TTS model
pip install "hftool[with_voiceover,with_tts_kokoro]"

# For voice cloning, also install Chatterbox:
pip install "hftool[with_voiceover,with_tts_chatterbox]"

# Ensure ffmpeg is installed (system package)
# Ubuntu/Debian: sudo apt install ffmpeg
# macOS: brew install ffmpeg
# Arch: sudo pacman -S ffmpeg
```

**Step 2: Create a timing script**

You need a script file that tells hftool *what to say* and *when to say it*. Two formats are supported:

**SRT format** (subtitle format — simple, widely supported):

```srt
1
00:00:02,000 --> 00:00:10,000
Welcome to the demo. In this walkthrough, we'll show you how the dashboard works.

2
00:00:13,000 --> 00:00:25,000
On the left sidebar, you can see the main navigation. Click on Analytics to view your reports.

3
00:00:26,000 --> 00:00:34,000
The analytics page shows your key metrics at a glance — revenue, users, and conversion rate.
```

**JSON format** (more metadata, supports emotion hints):

```json
{
  "metadata": {
    "title": "Dashboard Demo",
    "style": "tutorial"
  },
  "segments": [
    {
      "id": 1,
      "start": "00:00:02.000",
      "end": "00:00:10.000",
      "text": "Welcome to the demo. In this walkthrough, we'll show you how the dashboard works.",
      "emotion": "confident-friendly"
    },
    {
      "id": 2,
      "start": "00:00:13.000",
      "end": "00:00:25.000",
      "text": "On the left sidebar, you can see the main navigation.",
      "emotion": "instructional"
    }
  ]
}
```

**Step 3: Generate the voiceover**

```bash
# Audio-only voiceover (WAV output)
hftool voiceover --script timing.srt --output narration.wav

# Merge voiceover onto a video (replaces original audio)
hftool voiceover --script timing.srt --video input.mp4 --output final.mp4

# Keep original audio (ducked to 30%) + add voiceover on top
hftool voiceover --script timing.srt --video input.mp4 --output final.mp4 --keep-audio

# Use Chatterbox with voice cloning
hftool voiceover --script timing.srt --output narration.wav \
       --tts-model chatterbox --voice-ref my-voice-sample.wav

# Control Chatterbox emotion intensity
hftool voiceover --script timing.srt --output narration.wav \
       --tts-model chatterbox --voice-ref my-voice.wav --exaggeration 0.7
```

**Step 4: Resume interrupted generation**

The pipeline saves each segment as `seg_NNN.wav` in a segments directory. If generation is interrupted, re-running the same command skips already-generated segments:

```bash
# First run (generates all segments)
hftool voiceover --script timing.srt --output narration.wav --segments-dir ./segments

# Interrupted at segment 20/50? Just re-run — it picks up from segment 21
hftool voiceover --script timing.srt --output narration.wav --segments-dir ./segments

# Want to regenerate a specific segment? Delete it and re-run
rm segments/seg_015.wav
hftool voiceover --script timing.srt --output narration.wav --segments-dir ./segments
```

---

## Auto-Voiceover (Entry Point A)

Automatically analyze a video and generate narration — no script needed.

### How it works

1. **Scene Detection** — PySceneDetect finds scene boundaries in the video
2. **Keyframe Extraction** — FFmpeg grabs representative frames from each scene
3. **VLM Analysis** — Qwen 3.5 VLM analyzes each keyframe and describes what's happening
4. **Script Generation** — The VLM assembles descriptions into a timed narration script
5. **Review** — You can edit the generated script before proceeding
6. **TTS + Merge** — Kokoro/Chatterbox generates speech, merged onto the video

### Usage

```bash
# Full auto-voiceover (opens editor for review)
hftool voiceover --auto --video demo.mp4 --output final.mp4

# Skip editor review
hftool voiceover --auto --video demo.mp4 --output final.mp4 --no-edit

# Choose narration style
hftool voiceover --auto --video demo.mp4 --output final.mp4 --style presentation

# Save generated script for manual editing
hftool voiceover --auto --video demo.mp4 --output final.mp4 \
    --save-script script.json --no-edit
```

### Docker Two-Phase Workflow

Since Docker containers can't open a text editor, use the two-phase approach:

```bash
# Phase 1: Generate script (inside Docker)
hftool docker run -- voiceover --auto --video /workspace/demo.mp4 \
    --save-script /workspace/script.json --no-edit --output /workspace/final.mp4

# Edit script.json on your host machine with any editor

# Phase 2: Generate voiceover from edited script
hftool docker run -- voiceover --script /workspace/script.json \
    --video /workspace/demo.mp4 --output /workspace/final.mp4
```

### Narration Styles

| Style | Description |
|-------|-------------|
| `tutorial` (default) | Second-person guidance ("you"). Step by step. |
| `presentation` | Formal tone. Conference-style. Measured pace. |
| `demo` | Casual but professional. Feature-focused. Energetic. |
| `casual` | First-person. Conversational. Like explaining to a friend. |
| `formal` | Third-person. Documentation style. Precise. |

### VLM Model Options

| Model | VRAM | Default | Notes |
|-------|------|---------|-------|
| `qwen3.5-9b` | ~18 GB | Yes | Best balance of quality and speed |
| `qwen3.5-4b` | ~8 GB | | Lighter, for testing or low VRAM |
| `qwen3.5-27b` | ~54 GB | | Highest quality (needs FP8 or multi-GPU) |
| `internvl3.5-8b` | ~17 GB | | Non-Qwen alternative |

---

## Re-Voice (Entry Point B)

Replace existing narration in a video with a new TTS voice.

### How it works

1. **Extract Audio** — FFmpeg extracts the audio track from the video
2. **ASR Transcription** — Whisper transcribes the existing narration with timestamps
3. **Review** — Edit the transcript before re-generating
4. **TTS + Merge** — New voice generates speech at the same timestamps

### Usage

```bash
# Re-voice with default TTS (Kokoro)
hftool voiceover --revoice --video tutorial.mp4 --output revoiced.mp4

# Re-voice with Chatterbox voice cloning
hftool voiceover --revoice --video tutorial.mp4 --output revoiced.mp4 \
    --tts-model chatterbox --voice-ref my-voice.wav

# Keep original audio (ducked) + new voiceover
hftool voiceover --revoice --video tutorial.mp4 --output revoiced.mp4 --keep-audio
```

---

## Voiceover CLI Reference

```
hftool voiceover [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--auto` | | Auto-generate voiceover from video (Entry Point A) |
| `--revoice` | | Re-voice existing narration (Entry Point B) |
| `--script`, `-s` | | Path to SRT or JSON script file (Entry Point C) |
| `--output`, `-o` | (required) | Output file path |
| `--video`, `-v` | | Input video to merge voiceover onto |
| `--tts-model` | `kokoro` | TTS model: `kokoro` or `chatterbox` |
| `--voice-ref` | | Reference audio for voice cloning |
| `--exaggeration` | `0.4` | Emotion intensity for Chatterbox |
| `--keep-audio` | off | Duck original video audio |
| `--vlm-model` | `qwen3.5-9b` | VLM for auto mode frame analysis |
| `--style` | `tutorial` | Narration style for auto mode |
| `--scene-threshold` | `3.0` | Scene detection sensitivity |
| `--no-edit` | off | Skip editor review of generated script |
| `--save-script` | | Save generated script to file path |
| `--segments-dir` | alongside output | Segment WAV storage directory |
| `--device`, `-d` | `auto` | Device: auto, cuda, mps, cpu |
| `--dtype` | auto | Data type: bfloat16, float16, float32 |
