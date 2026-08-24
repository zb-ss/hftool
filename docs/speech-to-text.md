# Speech-to-Text (ASR)

> [Back to README](../README.md) | [All Tasks](../README.md#supported-tasks)

Transcribe audio with Whisper:

```bash
# Basic transcription
hftool -t asr -i recording.wav -o transcript.txt

# With specific model
hftool -t asr -m openai/whisper-large-v3 -i podcast.mp3 -o transcript.txt

# With timestamps (outputs JSON)
hftool -t asr -i interview.wav -o transcript.json \
       -- --return_timestamps true

# Generate SRT subtitles
hftool -t asr -i video_audio.wav -o subtitles.srt \
       -- --return_timestamps true --format srt
```

**Supported models:**
- `openai/whisper-large-v3` (best quality)
- `openai/whisper-medium`
- `openai/whisper-small` (fastest)
