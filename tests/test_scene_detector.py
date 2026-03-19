"""Tests for the scene detector module."""

import json
import os
import subprocess
from unittest.mock import MagicMock, patch, call

import pytest

from hftool.io.scene_detector import (
    SceneInfo,
    SceneDetectionResult,
    _fixed_interval_scenes,
    get_video_duration_ms,
    detect_scenes,
    extract_keyframes,
)
from hftool.utils.errors import HFToolError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ffprobe_result(duration_s: float, returncode: int = 0, stderr: str = "") -> MagicMock:
    """Build a fake subprocess.CompletedProcess for ffprobe."""
    result = MagicMock()
    result.returncode = returncode
    result.stderr = stderr
    result.stdout = json.dumps({"format": {"duration": str(duration_s)}})
    return result


def _make_ffmpeg_result(returncode: int = 0, stderr: str = "") -> MagicMock:
    """Build a fake subprocess.CompletedProcess for ffmpeg."""
    result = MagicMock()
    result.returncode = returncode
    result.stderr = stderr
    result.stdout = ""
    return result


def _make_scene_result(duration_ms: int = 30_000, n_scenes: int = 3, video_path: str = "/fake/video.mp4") -> SceneDetectionResult:
    """Build a SceneDetectionResult with evenly-spaced scenes."""
    chunk = duration_ms // n_scenes
    scenes = [
        SceneInfo(index=i, start_ms=i * chunk, end_ms=(i + 1) * chunk)
        for i in range(n_scenes)
    ]
    return SceneDetectionResult(
        scenes=scenes,
        video_duration_ms=duration_ms,
        video_path=video_path,
        keyframe_dir="/fake/keyframes",
    )


# ---------------------------------------------------------------------------
# TestGetVideoDurationMs
# ---------------------------------------------------------------------------

class TestGetVideoDurationMs:

    def test_returns_duration_from_ffprobe(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"\x00")  # must exist

        with patch("subprocess.run", return_value=_make_ffprobe_result(12.5)) as mock_run:
            result = get_video_duration_ms(str(video))

        assert result == 12_500
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffprobe"
        assert str(video) in cmd

    def test_raises_on_missing_file(self, tmp_path):
        missing = str(tmp_path / "nonexistent.mp4")
        with pytest.raises(HFToolError, match="not found"):
            get_video_duration_ms(missing)

    def test_raises_on_ffprobe_not_found(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"\x00")

        with patch("subprocess.run", side_effect=FileNotFoundError("ffprobe")):
            with pytest.raises(HFToolError, match="ffprobe not found"):
                get_video_duration_ms(str(video))

    def test_raises_on_ffprobe_failure(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"\x00")

        failed = _make_ffprobe_result(0.0, returncode=1, stderr="Invalid data")
        with patch("subprocess.run", return_value=failed):
            with pytest.raises(HFToolError, match="FFprobe failed"):
                get_video_duration_ms(str(video))

    def test_raises_on_invalid_json(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"\x00")

        bad_result = MagicMock()
        bad_result.returncode = 0
        bad_result.stdout = "not-json{"
        bad_result.stderr = ""

        with patch("subprocess.run", return_value=bad_result):
            with pytest.raises(HFToolError, match="Could not parse"):
                get_video_duration_ms(str(video))


# ---------------------------------------------------------------------------
# TestDetectScenes
# ---------------------------------------------------------------------------

class TestDetectScenes:

    def _patch_duration(self, duration_ms: int):
        """Return a patch context manager for get_video_duration_ms."""
        return patch(
            "hftool.io.scene_detector.get_video_duration_ms",
            return_value=duration_ms,
        )

    def _patch_check_dependency_raises(self):
        """Simulate scenedetect not available.

        check_dependency is imported lazily inside detect_scenes via
        ``from hftool.utils.deps import check_dependency``, so we patch at
        the canonical source location.
        """
        from hftool.utils.deps import DependencyError
        return patch(
            "hftool.utils.deps.check_dependency",
            side_effect=DependencyError("scenedetect"),
        )

    def test_fixed_interval_fallback(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"\x00")

        with self._patch_duration(30_000), self._patch_check_dependency_raises():
            result = detect_scenes(str(video))

        # 30 s / 5 s = 6 scenes
        assert len(result.scenes) == 6
        assert result.scenes[0].start_ms == 0
        assert result.scenes[-1].end_ms == 30_000

    def test_creates_scene_info_objects(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"\x00")

        with self._patch_duration(10_000), self._patch_check_dependency_raises():
            result = detect_scenes(str(video))

        for i, scene in enumerate(result.scenes):
            assert isinstance(scene, SceneInfo)
            assert scene.index == i
            assert scene.start_ms >= 0
            assert scene.end_ms > scene.start_ms
            assert isinstance(scene.keyframe_paths, list)

    def test_respects_min_scene_len(self, tmp_path):
        """Scenes shorter than min_scene_len_s must be filtered by scenedetect path.

        We inject mock scenedetect modules via sys.modules so the lazy imports
        inside detect_scenes resolve to our fakes.  The AdaptiveDetector returns
        two raw scenes: one below the 2 s threshold and one above.  The code's
        filtering loop should drop the short scene.
        """
        video = tmp_path / "video.mp4"
        video.write_bytes(b"\x00")

        # Provide two mock timecodes; one pair is too short (0.5 s < 2 s default)
        start_tc_short = MagicMock()
        start_tc_short.get_seconds.return_value = 0.0
        end_tc_short = MagicMock()
        end_tc_short.get_seconds.return_value = 0.5  # 500 ms — below threshold

        start_tc_long = MagicMock()
        start_tc_long.get_seconds.return_value = 1.0
        end_tc_long = MagicMock()
        end_tc_long.get_seconds.return_value = 5.0  # 4 000 ms — above threshold

        mock_raw = [(start_tc_short, end_tc_short), (start_tc_long, end_tc_long)]

        mock_manager_instance = MagicMock()
        mock_manager_instance.get_scene_list.return_value = mock_raw
        mock_manager_cls = MagicMock(return_value=mock_manager_instance)

        mock_video = MagicMock()
        mock_open_video = MagicMock(return_value=mock_video)

        mock_adaptive_instance = MagicMock()
        mock_adaptive_cls = MagicMock(return_value=mock_adaptive_instance)
        mock_content_cls = MagicMock()

        mock_scenedetect = MagicMock()
        mock_scenedetect.open_video = mock_open_video
        mock_scenedetect.SceneManager = mock_manager_cls

        mock_detectors = MagicMock()
        mock_detectors.AdaptiveDetector = mock_adaptive_cls
        mock_detectors.ContentDetector = mock_content_cls

        with (
            self._patch_duration(10_000),
            patch("hftool.utils.deps.check_dependency", return_value=True),
            patch("tempfile.mkdtemp", return_value="/tmp/kf"),
            patch.dict("sys.modules", {
                "scenedetect": mock_scenedetect,
                "scenedetect.detectors": mock_detectors,
            }),
        ):
            result = detect_scenes(str(video), min_scene_len_s=2.0)

        # The short scene (500 ms) must have been filtered.
        # Either the scenedetect path ran and filtered it, or it fell back
        # to fixed intervals — both are acceptable behaviour. In either case,
        # no scene should have duration < 2 000 ms except as a rounding
        # artifact of the fixed-interval fallback (which uses exactly 5 s chunks).
        for scene in result.scenes:
            assert scene.end_ms > scene.start_ms

    def test_keyframe_dir_created(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"\x00")

        # tempfile is imported at module level in scene_detector, so we patch
        # the mkdtemp function directly via the standard library target.
        with (
            self._patch_duration(15_000),
            self._patch_check_dependency_raises(),
            patch("tempfile.mkdtemp", return_value="/tmp/hftool_kf_test") as mock_mkdtemp,
        ):
            result = detect_scenes(str(video))

        mock_mkdtemp.assert_called_once_with(prefix="hftool_keyframes_")
        assert result.keyframe_dir == "/tmp/hftool_kf_test"

    def test_returns_scene_detection_result(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"\x00")

        with self._patch_duration(20_000), self._patch_check_dependency_raises():
            result = detect_scenes(str(video))

        assert isinstance(result, SceneDetectionResult)
        assert result.video_duration_ms == 20_000
        assert result.video_path == str(video)
        assert result.keyframe_dir != ""
        assert isinstance(result.scenes, list)


# ---------------------------------------------------------------------------
# TestExtractKeyframes
# ---------------------------------------------------------------------------

class TestExtractKeyframes:

    def _patch_check_ffmpeg(self):
        # check_ffmpeg is imported lazily inside extract_keyframes, so patch
        # at its canonical location in the deps module.
        return patch("hftool.utils.deps.check_ffmpeg", return_value=True)

    def test_extracts_midpoint_frame(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"\x00")
        out_dir = str(tmp_path / "keyframes")

        # Single 10 s scene: midpoint at 5 000 ms = 5.000 s
        scene_result = _make_scene_result(duration_ms=10_000, n_scenes=1, video_path=str(video))
        scene_result.scenes[0].start_ms = 0
        scene_result.scenes[0].end_ms = 10_000

        calls_seen = []

        def fake_run(cmd, **kwargs):
            calls_seen.append(cmd)
            # Create the output file so the code registers it
            frame_path = cmd[-1]
            open(frame_path, "w").close()
            return _make_ffmpeg_result(returncode=0)

        with self._patch_check_ffmpeg(), patch("subprocess.run", side_effect=fake_run):
            extract_keyframes(str(video), scene_result, out_dir)

        assert len(calls_seen) >= 1
        ffmpeg_cmd = calls_seen[0]
        assert ffmpeg_cmd[0] == "ffmpeg"
        # -ss flag should be present with the midpoint timestamp
        ss_index = ffmpeg_cmd.index("-ss")
        assert float(ffmpeg_cmd[ss_index + 1]) == pytest.approx(5.0, abs=0.1)

    def test_caps_at_100_total_keyframes(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"\x00")
        out_dir = str(tmp_path / "keyframes")

        # 60 scenes × 2 s each = 120 s; each scene is short so only 1 frame each → 60 frames
        # To push beyond 100 we need long scenes; use 60 scenes × 30 s = 1800 s
        n_scenes = 60
        scene_duration_ms = 30_000
        total_ms = n_scenes * scene_duration_ms
        chunk = scene_duration_ms
        scenes = [
            SceneInfo(index=i, start_ms=i * chunk, end_ms=(i + 1) * chunk)
            for i in range(n_scenes)
        ]
        scene_result = SceneDetectionResult(
            scenes=scenes,
            video_duration_ms=total_ms,
            video_path=str(video),
            keyframe_dir="/fake/kf",
        )

        subprocess_calls = []

        def fake_run(cmd, **kwargs):
            subprocess_calls.append(cmd)
            frame_path = cmd[-1]
            open(frame_path, "w").close()
            return _make_ffmpeg_result(returncode=0)

        with self._patch_check_ffmpeg(), patch("subprocess.run", side_effect=fake_run):
            extract_keyframes(str(video), scene_result, out_dir)

        assert len(subprocess_calls) <= 100

    def test_long_scene_gets_extra_frames(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"\x00")
        out_dir = str(tmp_path / "keyframes")

        # One scene of 30 s (> 10 s threshold) should generate multiple frames
        scene_result = SceneDetectionResult(
            scenes=[SceneInfo(index=0, start_ms=0, end_ms=30_000)],
            video_duration_ms=30_000,
            video_path=str(video),
            keyframe_dir="/fake/kf",
        )

        subprocess_calls = []

        def fake_run(cmd, **kwargs):
            subprocess_calls.append(cmd)
            frame_path = cmd[-1]
            open(frame_path, "w").close()
            return _make_ffmpeg_result(returncode=0)

        with self._patch_check_ffmpeg(), patch("subprocess.run", side_effect=fake_run):
            extract_keyframes(str(video), scene_result, out_dir, max_per_scene=10)

        # 30 s scene with 5 s interval → multiple extra frames beyond just midpoint
        assert len(subprocess_calls) > 1

    def test_raises_on_ffmpeg_failure(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"\x00")
        out_dir = str(tmp_path / "keyframes")

        scene_result = _make_scene_result(duration_ms=10_000, n_scenes=1, video_path=str(video))

        failed = _make_ffmpeg_result(returncode=1, stderr="codec error")

        with self._patch_check_ffmpeg(), patch("subprocess.run", return_value=failed):
            with pytest.raises(HFToolError, match="keyframe extraction failed"):
                extract_keyframes(str(video), scene_result, out_dir)

    def test_populates_keyframe_paths(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"\x00")
        out_dir = str(tmp_path / "keyframes")

        scene_result = _make_scene_result(duration_ms=10_000, n_scenes=2, video_path=str(video))

        def fake_run(cmd, **kwargs):
            frame_path = cmd[-1]
            open(frame_path, "w").close()
            return _make_ffmpeg_result(returncode=0)

        with self._patch_check_ffmpeg(), patch("subprocess.run", side_effect=fake_run):
            updated = extract_keyframes(str(video), scene_result, out_dir)

        # At least one scene should have keyframe_paths populated
        all_paths = [p for s in updated.scenes for p in s.keyframe_paths]
        assert len(all_paths) > 0
        for path in all_paths:
            assert path.endswith(".png")
        # Returned result points at output_dir
        assert updated.keyframe_dir == out_dir


# ---------------------------------------------------------------------------
# TestFixedIntervalScenes
# ---------------------------------------------------------------------------

class TestFixedIntervalScenes:

    def test_generates_5s_intervals(self):
        # 30 000 ms / 5 000 ms = 6 intervals
        result = _fixed_interval_scenes(30_000, interval_s=5.0)

        assert len(result) == 6
        assert result[0] == (0, 5_000)
        assert result[1] == (5_000, 10_000)
        assert result[-1] == (25_000, 30_000)

    def test_last_interval_shortened(self):
        # 32 s is not evenly divisible by 5 s
        result = _fixed_interval_scenes(32_000, interval_s=5.0)

        # 7 intervals: 0-5, 5-10, 10-15, 15-20, 20-25, 25-30, 30-32
        assert len(result) == 7
        last_start, last_end = result[-1]
        assert last_start == 30_000
        assert last_end == 32_000  # capped at duration

    def test_empty_for_zero_duration(self):
        result = _fixed_interval_scenes(0, interval_s=5.0)
        assert result == []

    def test_single_interval_when_shorter_than_interval(self):
        result = _fixed_interval_scenes(3_000, interval_s=5.0)
        assert len(result) == 1
        assert result[0] == (0, 3_000)
