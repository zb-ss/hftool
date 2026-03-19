# Text-to-Speech

> [Back to README](../README.md) | [All Tasks](../README.md#supported-tasks)

```bash
# Quick TTS with Kokoro (default — fast, lightweight, CPU-capable)
hftool -t tts -i "Hello, this is a test of the text to speech system." -o hello.wav

# Chatterbox TTS (higher quality, voice cloning, emotion control)
hftool -t tts -m chatterbox -i "Welcome to hftool." -o welcome.wav

# Chatterbox with voice cloning from a reference recording
hftool -t tts -m chatterbox -i "This sounds like you." -o cloned.wav \
       -- --voice_ref my-voice-sample.wav

# Bark (legacy, multi-language, sound effects)
hftool -t tts -m suno/bark -i "Bark still works too." -o bark.wav

# Output as MP3 (requires ffmpeg)
hftool -t tts -i "This will be saved as MP3." -o output.mp3
```

**Supported models:**

| Model | Size | Speed | Voice Cloning | License | Best For |
|-------|------|-------|---------------|---------|----------|
| `kokoro` (default) | 0.3 GB | Very fast, runs on CPU | No | Apache 2.0 | Quick TTS, lightweight |
| `chatterbox` | 2 GB | Fast on GPU | Yes (zero-shot) | MIT | Professional voiceovers, cloning |
| `bark-small` | 1.5 GB | Moderate | No | — | Sound effects, multi-language |
| `bark` | 5 GB | Slower | No | — | Full Bark quality |
| `mms-tts-eng` | 0.3 GB | Fast | No | — | Ultra-lightweight English |

#### Voice Cloning with Chatterbox

Chatterbox can clone any voice from a short reference recording (5-30 seconds of clear speech):

```bash
# Record a voice sample (or use any WAV/MP3 with clear speech)
# Then use --voice-ref to clone that voice:

# Via the voiceover pipeline (recommended for multi-segment projects)
hftool voiceover --script narration.srt --output voiceover.wav \
       --tts-model chatterbox --voice-ref my-voice.wav

# Via direct TTS (single text input)
hftool -t tts -m chatterbox -i "Your text here" -o output.wav \
       -- --voice_ref my-voice.wav

# Control emotion intensity (0.0 = flat, 1.0 = very expressive)
hftool -t tts -m chatterbox -i "Exciting news!" -o excited.wav \
       -- --voice_ref my-voice.wav --exaggeration 0.7
```

**Tips for good voice cloning:**
- Use a clean recording with minimal background noise
- 10-30 seconds of natural speech works best
- Avoid music or other speakers in the reference
- WAV format at 24kHz+ recommended (MP3 also works)
