"""Audio mixer for voiceover pipeline.

Handles placing TTS audio segments at timestamps and merging with video
using FFmpeg. Ported from the POC at local/poc/voiceover_generate.py.
"""

import os
import subprocess
from dataclasses import dataclass
from typing import List


@dataclass
class SegmentAudio:
    """A generated audio segment with its target timestamp."""

    path: str
    start_ms: int
    segment_id: int


def check_ffmpeg() -> bool:
    """Verify ffmpeg is available on the system.

    Returns:
        True if ffmpeg is available

    Raises:
        HFToolError: If ffmpeg is not found
    """
    from hftool.utils.deps import check_ffmpeg as _check_ffmpeg
    return _check_ffmpeg()


def merge_segments(
    segments: List[SegmentAudio],
    output_path: str,
    sample_rate: int = 24000,
    timeout: int = 300,
) -> str:
    """Place WAV segments at their timestamps and merge into one audio track.

    Uses FFmpeg adelay to position each segment, amix to combine them
    (normalize=0 to prevent volume reduction), and loudnorm for broadcast
    loudness normalization.

    Args:
        segments: List of SegmentAudio with paths and start timestamps
        output_path: Path for the merged WAV output
        sample_rate: Output sample rate (default 24000)
        timeout: FFmpeg timeout in seconds

    Returns:
        Path to the merged audio file

    Raises:
        HFToolError: If FFmpeg fails or no segments provided
    """
    from hftool.utils.errors import HFToolError

    if not segments:
        raise HFToolError("No audio segments to merge.")

    check_ffmpeg()

    # Verify all segment files exist
    for seg in segments:
        if not os.path.exists(seg.path):
            raise HFToolError(
                f"Segment file not found: {seg.path}",
                suggestion=f"Segment {seg.segment_id} was not generated. Re-run to generate missing segments.",
            )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Build FFmpeg filter complex
    inputs = []
    filter_parts = []

    for i, seg in enumerate(segments):
        inputs.extend(["-i", seg.path])
        filter_parts.append(f"[{i}]adelay={seg.start_ms}|{seg.start_ms}[d{i}]")

    mix_inputs = "".join(f"[d{i}]" for i in range(len(segments)))
    filter_parts.append(
        f"{mix_inputs}amix=inputs={len(segments)}:duration=longest:normalize=0[m]"
    )
    filter_parts.append("[m]loudnorm=I=-16:TP=-1.5:LRA=11[out]")

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", ";".join(filter_parts),
        "-map", "[out]",
        "-ar", str(sample_rate),
        "-ac", "1",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    if result.returncode != 0:
        stderr_tail = result.stderr[-500:] if result.stderr else "No error output"
        raise HFToolError(
            f"FFmpeg merge failed: {stderr_tail}",
            suggestion="Check that all segment WAV files are valid.",
        )

    return output_path


def merge_with_video(
    video_path: str,
    audio_path: str,
    output_path: str,
    keep_original: bool = False,
    timeout: int = 300,
) -> str:
    """Combine narration audio with a video file.

    Args:
        video_path: Path to input video
        audio_path: Path to narration audio (WAV or other)
        output_path: Path for output video
        keep_original: If True, duck original audio to 30% and mix.
            If False, strip original audio and use narration only.
        timeout: FFmpeg timeout in seconds

    Returns:
        Path to the output video

    Raises:
        HFToolError: If FFmpeg fails
    """
    from hftool.utils.errors import HFToolError

    check_ffmpeg()

    if not os.path.exists(video_path):
        raise HFToolError(f"Video file not found: {video_path}")
    if not os.path.exists(audio_path):
        raise HFToolError(f"Audio file not found: {audio_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if keep_original:
        # Mix: duck original to 30%, narration at full volume.
        # duration=first keeps the full original video audio length.
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-filter_complex",
            "[1:a]aresample=48000[narr];[0:a][narr]amix=inputs=2:duration=first:weights=0.3 1[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            output_path,
        ]
    else:
        # Replace: strip original audio, pad narration to match video length.
        # apad adds silence after the narration ends so the video is not
        # truncated.  -shortest then stops at the video stream end (not the
        # now-infinite audio pad).
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-filter_complex", "[1:a]apad[a]",
            "-map", "0:v", "-map", "[a]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            output_path,
        ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    if result.returncode != 0:
        stderr_tail = result.stderr[-500:] if result.stderr else "No error output"
        raise HFToolError(
            f"FFmpeg video merge failed: {stderr_tail}",
            suggestion="Check that video and audio files are valid.",
        )

    return output_path
