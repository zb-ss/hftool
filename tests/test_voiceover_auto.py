"""Tests for VoiceoverTask auto-voiceover pipeline and re-voice entry points."""

import os
from unittest.mock import MagicMock, call, patch

import pytest


def _make_script():
    from hftool.io.script_parser import ScriptData, ScriptSegment

    return ScriptData(segments=[
        ScriptSegment(id=1, start_ms=0, end_ms=5000, text="Hello world"),
        ScriptSegment(id=2, start_ms=6000, end_ms=12000, text="Second segment"),
    ])


class TestVoiceoverTaskInit:
    """Tests for VoiceoverTask constructor defaults and custom values."""

    def test_new_params_defaults(self):
        from hftool.tasks.voiceover import VoiceoverTask

        task = VoiceoverTask()
        assert task.vlm_model == "qwen3-vl-8b"
        assert task.narration_style == "tutorial"
        assert task.scene_threshold == 3.0
        assert task.no_edit is False
        assert task.save_script is None

    def test_new_params_custom(self):
        from hftool.tasks.voiceover import VoiceoverTask

        task = VoiceoverTask(
            vlm_model="qwen3.5-4b",
            narration_style="documentary",
            scene_threshold=5.0,
            no_edit=True,
            save_script="/tmp/script.json",
        )
        assert task.vlm_model == "qwen3.5-4b"
        assert task.narration_style == "documentary"
        assert task.scene_threshold == 5.0
        assert task.no_edit is True
        assert task.save_script == "/tmp/script.json"


class TestVoiceoverResolveVLM:
    """Tests for VoiceoverTask._resolve_vlm_model."""

    def test_resolve_short_name(self):
        from hftool.tasks.voiceover import VoiceoverTask

        task = VoiceoverTask()
        result = task._resolve_vlm_model("qwen3-vl-8b")
        assert result == "Qwen/Qwen3-VL-8B-Instruct"

    def test_resolve_repo_id_passthrough(self):
        from hftool.tasks.voiceover import VoiceoverTask

        task = VoiceoverTask()
        result = task._resolve_vlm_model("Qwen/Qwen3-VL-8B-Instruct")
        assert result == "Qwen/Qwen3-VL-8B-Instruct"

    def test_resolve_unknown_passthrough(self):
        from hftool.tasks.voiceover import VoiceoverTask

        task = VoiceoverTask()
        result = task._resolve_vlm_model("custom/model")
        assert result == "custom/model"


class TestVoiceoverExtractAudio:
    """Tests for VoiceoverTask._extract_audio."""

    def test_calls_ffmpeg(self, tmp_path):
        from hftool.tasks.voiceover import VoiceoverTask

        task = VoiceoverTask()
        video = str(tmp_path / "video.mp4")
        audio_out = str(tmp_path / "audio.wav")

        # Create a dummy video file so makedirs doesn't fail
        open(video, "w").close()

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("hftool.utils.deps.check_ffmpeg"):
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                task._extract_audio(video, audio_out)

        mock_run.assert_called_once()
        cmd_used = mock_run.call_args[0][0]
        assert "ffmpeg" in cmd_used
        assert "-vn" in cmd_used
        assert audio_out in cmd_used

    def test_raises_on_failure(self, tmp_path):
        from hftool.tasks.voiceover import VoiceoverTask
        from hftool.utils.errors import HFToolError

        task = VoiceoverTask()
        video = str(tmp_path / "video.mp4")
        audio_out = str(tmp_path / "audio.wav")
        open(video, "w").close()

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "FFmpeg error: codec not found"

        with patch("hftool.utils.deps.check_ffmpeg"):
            with patch("subprocess.run", return_value=mock_result):
                with pytest.raises(HFToolError):
                    task._extract_audio(video, audio_out)


class TestVoiceoverRunAutoSequencing:
    """Tests that VLM is unloaded before TTS is loaded in run_auto."""

    def test_vlm_unloaded_before_tts(self, tmp_path):
        from hftool.tasks.voiceover import VoiceoverTask

        video_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "out" / "output.mp4")
        open(video_path, "w").close()

        script = _make_script()
        call_order = []

        # Fake scene/keyframe objects
        fake_scene = MagicMock()
        fake_scene.scenes = [MagicMock(keyframe_paths=["frame1.jpg"])]
        fake_scene.video_duration_ms = 12000

        def fake_load_vlm():
            call_order.append("load_vlm")

        def fake_unload_vlm():
            call_order.append("unload_vlm")

        def fake_load_tts():
            call_order.append("load_tts")

        def fake_generate_and_merge(sc, out, vid=None, keep=False):
            call_order.append("generate_and_merge")
            return out

        with patch("hftool.utils.deps.check_ffmpeg"):
            with patch("os.path.exists", return_value=True):
                with patch("os.makedirs"):
                    with patch("hftool.io.scene_detector.detect_scenes", return_value=fake_scene):
                        with patch("hftool.io.scene_detector.extract_keyframes", return_value=fake_scene):
                            with patch("hftool.io.script_generator.analyze_frames", return_value=["analysis1"]):
                                with patch("hftool.io.script_generator.generate_script", return_value=script):
                                    with patch("hftool.io.script_review.review_script", return_value=script):
                                        task = VoiceoverTask(no_edit=True)
                                        task._load_vlm = fake_load_vlm
                                        task._unload_vlm = fake_unload_vlm
                                        task._load_tts = fake_load_tts
                                        task._generate_and_merge = fake_generate_and_merge

                                        task.run_auto(video_path, output_path)

        assert "load_vlm" in call_order
        assert "unload_vlm" in call_order
        assert "generate_and_merge" in call_order

        vlm_unload_idx = call_order.index("unload_vlm")
        generate_idx = call_order.index("generate_and_merge")
        assert vlm_unload_idx < generate_idx, (
            f"VLM unload (idx {vlm_unload_idx}) must happen before TTS/merge "
            f"(idx {generate_idx}). Order was: {call_order}"
        )


class TestVoiceoverRunRevoice:
    """Tests for VoiceoverTask.run_revoice."""

    def test_extracts_then_transcribes(self, tmp_path):
        from hftool.tasks.voiceover import VoiceoverTask

        video_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "out" / "output.mp4")
        open(video_path, "w").close()

        script = _make_script()

        extract_calls = []
        transcribe_calls = []

        def fake_extract(vid, out):
            extract_calls.append((vid, out))
            return out

        def fake_transcribe(audio):
            transcribe_calls.append(audio)
            return script

        with patch("hftool.utils.deps.check_ffmpeg"):
            with patch("os.path.exists", return_value=True):
                with patch("os.makedirs"):
                    with patch("hftool.io.script_review.review_script", return_value=script):
                        task = VoiceoverTask(no_edit=True)
                        task._extract_audio = fake_extract
                        task._transcribe_to_script = fake_transcribe
                        task._generate_and_merge = MagicMock(return_value=output_path)

                        task.run_revoice(video_path, output_path)

        assert len(extract_calls) == 1, "Expected _extract_audio to be called once"
        assert len(transcribe_calls) == 1, "Expected _transcribe_to_script to be called once"

        extracted_audio = extract_calls[0][1]
        assert transcribe_calls[0] == extracted_audio, (
            "Audio path passed to _transcribe_to_script must match output of _extract_audio"
        )
