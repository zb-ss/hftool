"""Tests for the voiceover pipeline task."""

import pytest


class TestVoiceoverTaskRegistry:
    """Tests for voiceover task registration."""

    def test_voiceover_task_registered(self):
        from hftool.core.registry import TASK_REGISTRY
        assert "voiceover" in TASK_REGISTRY

    def test_voiceover_alias_vo(self):
        from hftool.core.registry import TASK_ALIASES
        assert "vo" in TASK_ALIASES
        assert TASK_ALIASES["vo"] == "voiceover"

    def test_voiceover_requires_ffmpeg(self):
        from hftool.core.registry import TASK_REGISTRY
        config = TASK_REGISTRY["voiceover"]
        assert config.requires_ffmpeg is True

    def test_voiceover_default_models(self):
        from hftool.core.registry import TASK_REGISTRY
        config = TASK_REGISTRY["voiceover"]
        assert "hexgrad/Kokoro-82M" in config.default_models
        assert "ResembleAI/chatterbox" in config.default_models

    def test_voiceover_handler_path(self):
        from hftool.core.registry import TASK_REGISTRY
        config = TASK_REGISTRY["voiceover"]
        assert config.handler == "hftool.tasks.voiceover"


class TestVoiceoverTaskFactory:
    """Tests for VoiceoverTask creation."""

    def test_create_task(self):
        from hftool.tasks.voiceover import create_task
        task = create_task()
        assert task is not None

    def test_create_task_with_options(self):
        from hftool.tasks.voiceover import VoiceoverTask
        task = VoiceoverTask(
            device="cpu",
            tts_model="chatterbox",
            voice_ref="/tmp/ref.wav",
            exaggeration=0.6,
            segments_dir="/tmp/segs",
        )
        assert task.device == "cpu"
        assert task.tts_model == "chatterbox"
        assert task.voice_ref == "/tmp/ref.wav"
        assert task.exaggeration == 0.6
        assert task.segments_dir == "/tmp/segs"


class TestVoiceoverModelResolution:
    """Tests for TTS model resolution in VoiceoverTask."""

    def test_resolve_kokoro(self):
        from hftool.tasks.voiceover import VoiceoverTask
        task = VoiceoverTask()
        resolved = task._resolve_tts_model("kokoro")
        assert resolved == "hexgrad/Kokoro-82M"

    def test_resolve_chatterbox(self):
        from hftool.tasks.voiceover import VoiceoverTask
        task = VoiceoverTask()
        resolved = task._resolve_tts_model("chatterbox")
        assert resolved == "ResembleAI/chatterbox"

    def test_resolve_full_repo_id(self):
        from hftool.tasks.voiceover import VoiceoverTask
        task = VoiceoverTask()
        resolved = task._resolve_tts_model("hexgrad/Kokoro-82M")
        assert resolved == "hexgrad/Kokoro-82M"

    def test_resolve_unknown_passes_through(self):
        from hftool.tasks.voiceover import VoiceoverTask
        task = VoiceoverTask()
        resolved = task._resolve_tts_model("some-custom/model")
        assert resolved == "some-custom/model"
