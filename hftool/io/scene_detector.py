"""Scene detection and keyframe extraction for voiceover pipeline.

Detects scene boundaries in video files using PySceneDetect and extracts
representative keyframes via FFmpeg for downstream captioning/description.
"""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SceneInfo:
    """A detected scene with its temporal boundaries and keyframe paths."""

    index: int
    start_ms: int
    end_ms: int
    keyframe_paths: List[str] = field(default_factory=list)


@dataclass
class SceneDetectionResult:
    """Full detection result for a video."""

    scenes: List[SceneInfo]
    video_duration_ms: int
    video_path: str
    keyframe_dir: str


def get_video_duration_ms(video_path: str) -> int:
    """Return video duration in milliseconds using FFprobe.

    Args:
        video_path: Path to the video file.

    Returns:
        Duration in milliseconds.

    Raises:
        HFToolError: If FFprobe is not available or fails.
    """
    from hftool.utils.errors import HFToolError

    if not os.path.exists(video_path):
        raise HFToolError(
            f"Video file not found: {video_path}",
            suggestion="Check the file path and try again.",
        )

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        video_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        raise HFToolError(
            "ffprobe not found. FFmpeg must be installed.",
            suggestion="Install FFmpeg: https://ffmpeg.org/download.html",
        )

    if result.returncode != 0:
        stderr_tail = result.stderr[-500:] if result.stderr else "No error output"
        raise HFToolError(
            f"FFprobe failed: {stderr_tail}",
            suggestion="Check that the video file is valid and not corrupted.",
        )

    try:
        data = json.loads(result.stdout)
        duration_s = float(data["format"]["duration"])
        return int(duration_s * 1000)
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise HFToolError(
            "Could not parse video duration from FFprobe output.",
            suggestion="Check that the file is a valid video.",
        ) from exc


def _fixed_interval_scenes(duration_ms: int, interval_s: float = 5.0) -> List[Tuple[int, int]]:
    """Generate (start_ms, end_ms) pairs at fixed intervals as a fallback."""
    interval_ms = int(interval_s * 1000)
    scenes: List[Tuple[int, int]] = []
    start = 0
    while start < duration_ms:
        end = min(start + interval_ms, duration_ms)
        scenes.append((start, end))
        start = end
    return scenes


def detect_scenes(
    video_path: str,
    threshold: float = 3.0,
    min_scene_len_s: float = 2.0,
) -> SceneDetectionResult:
    """Detect scene boundaries in a video file.

    Tries PySceneDetect AdaptiveDetector first, falls back to ContentDetector,
    then falls back to fixed 5-second intervals if both produce no scenes.

    Args:
        video_path: Path to the video file.
        threshold: Detection threshold (lower = more sensitive).
        min_scene_len_s: Minimum scene length in seconds.

    Returns:
        SceneDetectionResult with scenes populated but empty keyframe_paths.

    Raises:
        HFToolError: If the video cannot be read.
    """
    from hftool.utils.errors import HFToolError
    from hftool.utils.deps import check_dependency

    if not os.path.exists(video_path):
        raise HFToolError(
            f"Video file not found: {video_path}",
            suggestion="Check the file path and try again.",
        )

    duration_ms = get_video_duration_ms(video_path)
    keyframe_dir = tempfile.mkdtemp(prefix="hftool_keyframes_")

    scene_list_ms: List[Tuple[int, int]] = []

    try:
        check_dependency("scenedetect", extra="with_scene_detect", pip_name="scenedetect[opencv]")

        # Lazy import after dependency check
        from scenedetect import open_video, SceneManager  # type: ignore
        from scenedetect.detectors import AdaptiveDetector, ContentDetector  # type: ignore

        min_scene_frames = max(1, int(min_scene_len_s * 25))  # assume ~25fps default

        def _run_detector(detector) -> List[Tuple[int, int]]:
            video = open_video(video_path)
            manager = SceneManager()
            manager.add_detector(detector)
            manager.detect_scenes(video)
            raw = manager.get_scene_list()
            result: List[Tuple[int, int]] = []
            for start_tc, end_tc in raw:
                start_ms = int(start_tc.get_seconds() * 1000)
                end_ms = int(end_tc.get_seconds() * 1000)
                if (end_ms - start_ms) >= int(min_scene_len_s * 1000):
                    result.append((start_ms, end_ms))
            return result

        # Try AdaptiveDetector first
        try:
            scene_list_ms = _run_detector(
                AdaptiveDetector(adaptive_threshold=threshold, min_scene_len=min_scene_frames)
            )
        except Exception:
            scene_list_ms = []

        # Fall back to ContentDetector
        if not scene_list_ms:
            try:
                scene_list_ms = _run_detector(
                    ContentDetector(threshold=threshold * 10, min_scene_len=min_scene_frames)
                )
            except Exception:
                scene_list_ms = []

    except Exception:
        # scenedetect unavailable or failed entirely — handled below
        scene_list_ms = []

    # Final fallback: fixed intervals
    if not scene_list_ms:
        scene_list_ms = _fixed_interval_scenes(duration_ms, interval_s=5.0)

    scenes = [
        SceneInfo(index=i, start_ms=start, end_ms=end, keyframe_paths=[])
        for i, (start, end) in enumerate(scene_list_ms)
    ]

    return SceneDetectionResult(
        scenes=scenes,
        video_duration_ms=duration_ms,
        video_path=video_path,
        keyframe_dir=keyframe_dir,
    )


def extract_keyframes(
    video_path: str,
    scenes: SceneDetectionResult,
    output_dir: str,
    max_per_scene: int = 3,
    long_scene_interval_s: float = 5.0,
) -> SceneDetectionResult:
    """Extract keyframe PNGs from each scene using FFmpeg.

    For each scene, extracts a frame at the midpoint.  For scenes longer than
    10 seconds, additional frames are extracted every ``long_scene_interval_s``
    seconds.  The total number of keyframes across all scenes is capped at 100,
    sampled evenly if exceeded.

    Args:
        video_path: Path to the source video.
        scenes: SceneDetectionResult from :func:`detect_scenes`.
        output_dir: Directory where PNG frames will be written.
        max_per_scene: Maximum frames to extract per scene.
        long_scene_interval_s: Extra-frame interval (seconds) for long scenes.

    Returns:
        Updated SceneDetectionResult with keyframe_paths populated.

    Raises:
        HFToolError: If FFmpeg fails or the video cannot be read.
    """
    from hftool.utils.errors import HFToolError
    from hftool.utils.deps import check_ffmpeg

    check_ffmpeg()

    if not os.path.exists(video_path):
        raise HFToolError(
            f"Video file not found: {video_path}",
            suggestion="Check the file path and try again.",
        )

    os.makedirs(output_dir, exist_ok=True)

    long_scene_threshold_ms = 10_000  # 10 seconds

    # Build the list of (scene_index, timestamp_ms) pairs to extract
    extraction_plan: List[Tuple[int, int]] = []

    for scene in scenes.scenes:
        duration_ms = scene.end_ms - scene.start_ms
        timestamps: List[int] = []

        # Always include midpoint
        mid_ms = scene.start_ms + duration_ms // 2
        timestamps.append(mid_ms)

        # For long scenes, add frames at regular intervals
        if duration_ms > long_scene_threshold_ms:
            interval_ms = int(long_scene_interval_s * 1000)
            t = scene.start_ms + interval_ms
            while t < scene.end_ms - interval_ms // 2:
                if t != mid_ms:
                    timestamps.append(t)
                t += interval_ms

        # Sort and cap per scene
        timestamps.sort()
        timestamps = timestamps[:max_per_scene]

        for ts in timestamps:
            extraction_plan.append((scene.index, ts))

    # Cap total at 100, sampling evenly if needed
    max_total = 100
    if len(extraction_plan) > max_total:
        step = len(extraction_plan) / max_total
        extraction_plan = [
            extraction_plan[int(i * step)] for i in range(max_total)
        ]

    # Group by scene index for path assignment
    from collections import defaultdict
    scene_frames: dict = defaultdict(list)

    for scene_index, ts_ms in extraction_plan:
        ts_s = ts_ms / 1000.0
        frame_path = os.path.join(output_dir, f"scene{scene_index:04d}_t{ts_ms:08d}.png")

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{ts_s:.3f}",
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            frame_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            stderr_tail = result.stderr[-500:] if result.stderr else "No error output"
            raise HFToolError(
                f"FFmpeg keyframe extraction failed: {stderr_tail}",
                suggestion="Check that the video file is valid and FFmpeg is installed correctly.",
            )

        if os.path.exists(frame_path):
            scene_frames[scene_index].append(frame_path)

    # Build index for quick scene lookup
    scene_by_index = {s.index: s for s in scenes.scenes}

    for scene_index, paths in scene_frames.items():
        if scene_index in scene_by_index:
            scene_by_index[scene_index].keyframe_paths = sorted(paths)

    # Return updated result pointing at the output directory
    return SceneDetectionResult(
        scenes=scenes.scenes,
        video_duration_ms=scenes.video_duration_ms,
        video_path=scenes.video_path,
        keyframe_dir=output_dir,
    )
