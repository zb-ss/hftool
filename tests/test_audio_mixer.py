"""Tests for the audio mixer module."""

import os
import shutil
import struct
import tempfile

import pytest

from hftool.io.audio_mixer import SegmentAudio


def _ffmpeg_available() -> bool:
    """Check if ffmpeg is available."""
    return shutil.which("ffmpeg") is not None


def _write_silence_wav(path: str, duration_ms: int = 500, sample_rate: int = 24000) -> None:
    """Write a minimal silent WAV file."""
    num_samples = int(sample_rate * duration_ms / 1000)
    data_size = num_samples * 2  # 16-bit mono

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))  # chunk size
        f.write(struct.pack("<H", 1))   # PCM
        f.write(struct.pack("<H", 1))   # mono
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", sample_rate * 2))  # byte rate
        f.write(struct.pack("<H", 2))   # block align
        f.write(struct.pack("<H", 16))  # bits per sample
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(b"\x00" * data_size)


class TestSegmentAudio:
    """Tests for the SegmentAudio dataclass."""

    def test_fields(self):
        seg = SegmentAudio(path="/tmp/seg.wav", start_ms=1000, segment_id=1)
        assert seg.path == "/tmp/seg.wav"
        assert seg.start_ms == 1000
        assert seg.segment_id == 1


class TestMergeSegments:
    """Tests for merge_segments function."""

    def test_empty_segments_raises(self):
        from hftool.io.audio_mixer import merge_segments
        from hftool.utils.errors import HFToolError

        with pytest.raises(HFToolError, match="No audio segments"):
            merge_segments([], "/tmp/out.wav")

    def test_missing_segment_file_raises(self):
        from hftool.io.audio_mixer import merge_segments
        from hftool.utils.errors import HFToolError

        segments = [
            SegmentAudio(path="/nonexistent/seg_001.wav", start_ms=0, segment_id=1),
        ]
        with pytest.raises(HFToolError, match="not found"):
            merge_segments(segments, "/tmp/out.wav")

    @pytest.mark.skipif(
        not _ffmpeg_available(),
        reason="ffmpeg not available",
    )
    def test_merge_with_dummy_wav(self):
        """Integration test: merge actual WAV segments with ffmpeg."""
        from hftool.io.audio_mixer import merge_segments

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal WAV files
            for i in range(2):
                wav_path = os.path.join(tmpdir, f"seg_{i+1:03d}.wav")
                _write_silence_wav(wav_path, duration_ms=500, sample_rate=24000)

            segments = [
                SegmentAudio(path=os.path.join(tmpdir, "seg_001.wav"), start_ms=0, segment_id=1),
                SegmentAudio(path=os.path.join(tmpdir, "seg_002.wav"), start_ms=1000, segment_id=2),
            ]

            output_path = os.path.join(tmpdir, "merged.wav")
            result = merge_segments(segments, output_path)

            assert os.path.exists(result)
            assert os.path.getsize(result) > 0


class TestMergeWithVideo:
    """Tests for merge_with_video function."""

    def test_missing_video_raises(self):
        from hftool.io.audio_mixer import merge_with_video
        from hftool.utils.errors import HFToolError

        with pytest.raises(HFToolError, match="Video file not found"):
            merge_with_video("/nonexistent.mp4", "/tmp/audio.wav", "/tmp/out.mp4")

    def test_missing_audio_raises(self):
        from hftool.io.audio_mixer import merge_with_video
        from hftool.utils.errors import HFToolError

        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            with pytest.raises(HFToolError, match="Audio file not found"):
                merge_with_video(f.name, "/nonexistent.wav", "/tmp/out.mp4")
