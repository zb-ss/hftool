"""Tests for TTS model registration and configuration."""

import pytest


class TestTTSModelRegistry:
    """Tests for TTS entries in MODEL_REGISTRY."""

    def test_kokoro_registered(self):
        from hftool.core.models import MODEL_REGISTRY
        assert "kokoro" in MODEL_REGISTRY["text-to-speech"]

    def test_chatterbox_registered(self):
        from hftool.core.models import MODEL_REGISTRY
        assert "chatterbox" in MODEL_REGISTRY["text-to-speech"]

    def test_bark_still_registered(self):
        from hftool.core.models import MODEL_REGISTRY
        assert "bark-small" in MODEL_REGISTRY["text-to-speech"]
        assert "bark" in MODEL_REGISTRY["text-to-speech"]

    def test_kokoro_is_default(self):
        from hftool.core.models import MODEL_REGISTRY
        kokoro = MODEL_REGISTRY["text-to-speech"]["kokoro"]
        assert kokoro.is_default is True

    def test_bark_small_no_longer_default(self):
        from hftool.core.models import MODEL_REGISTRY
        bark = MODEL_REGISTRY["text-to-speech"]["bark-small"]
        assert bark.is_default is False

    def test_kokoro_model_type_custom(self):
        from hftool.core.models import MODEL_REGISTRY, ModelType
        kokoro = MODEL_REGISTRY["text-to-speech"]["kokoro"]
        assert kokoro.model_type == ModelType.CUSTOM

    def test_chatterbox_model_type_custom(self):
        from hftool.core.models import MODEL_REGISTRY, ModelType
        chatterbox = MODEL_REGISTRY["text-to-speech"]["chatterbox"]
        assert chatterbox.model_type == ModelType.CUSTOM

    def test_kokoro_repo_id(self):
        from hftool.core.models import MODEL_REGISTRY
        kokoro = MODEL_REGISTRY["text-to-speech"]["kokoro"]
        assert kokoro.repo_id == "hexgrad/Kokoro-82M"

    def test_chatterbox_repo_id(self):
        from hftool.core.models import MODEL_REGISTRY
        chatterbox = MODEL_REGISTRY["text-to-speech"]["chatterbox"]
        assert chatterbox.repo_id == "ResembleAI/chatterbox"

    def test_chatterbox_has_pip_dependencies(self):
        from hftool.core.models import MODEL_REGISTRY
        chatterbox = MODEL_REGISTRY["text-to-speech"]["chatterbox"]
        assert "chatterbox-tts" in chatterbox.pip_dependencies

    def test_kokoro_has_pip_dependencies(self):
        from hftool.core.models import MODEL_REGISTRY
        kokoro = MODEL_REGISTRY["text-to-speech"]["kokoro"]
        assert any("kokoro" in dep for dep in kokoro.pip_dependencies)

    def test_chatterbox_metadata_has_defaults(self):
        from hftool.core.models import MODEL_REGISTRY
        chatterbox = MODEL_REGISTRY["text-to-speech"]["chatterbox"]
        assert chatterbox.metadata.get("exaggeration") == 0.4
        assert chatterbox.metadata.get("cfg_weight") == 0.5
        assert chatterbox.metadata.get("temperature") == 0.7

    def test_get_default_model_returns_kokoro(self):
        from hftool.core.models import get_default_model_info
        default = get_default_model_info("text-to-speech")
        assert default.repo_id == "hexgrad/Kokoro-82M"

    def test_get_model_info_by_short_name(self):
        from hftool.core.models import get_model_info
        info = get_model_info("text-to-speech", "chatterbox")
        assert info.repo_id == "ResembleAI/chatterbox"


class TestTTSTaskConfig:
    """Tests for TTS task in TASK_REGISTRY."""

    def test_tts_task_registered(self):
        from hftool.core.registry import TASK_REGISTRY
        assert "text-to-speech" in TASK_REGISTRY

    def test_tts_default_models_includes_kokoro(self):
        from hftool.core.registry import TASK_REGISTRY
        config = TASK_REGISTRY["text-to-speech"]
        assert "hexgrad/Kokoro-82M" in config.default_models

    def test_tts_default_models_includes_chatterbox(self):
        from hftool.core.registry import TASK_REGISTRY
        config = TASK_REGISTRY["text-to-speech"]
        assert "ResembleAI/chatterbox" in config.default_models

    def test_tts_library_is_custom(self):
        from hftool.core.registry import TASK_REGISTRY
        config = TASK_REGISTRY["text-to-speech"]
        assert config.library == "custom"


class TestTTSTaskFactory:
    """Tests for TextToSpeechTask creation and config."""

    def test_create_task_returns_instance(self):
        from hftool.tasks.text_to_speech import create_task
        task = create_task()
        assert task is not None
        assert task.device == "auto"

    def test_create_task_with_device(self):
        from hftool.tasks.text_to_speech import create_task
        task = create_task(device="cpu")
        assert task.device == "cpu"

    def test_model_config_kokoro(self):
        from hftool.tasks.text_to_speech import TextToSpeechTask
        task = TextToSpeechTask()
        config = task._get_model_config("hexgrad/Kokoro-82M")
        assert config["sample_rate"] == 24000

    def test_model_config_chatterbox(self):
        from hftool.tasks.text_to_speech import TextToSpeechTask
        task = TextToSpeechTask()
        config = task._get_model_config("ResembleAI/chatterbox")
        assert config["sample_rate"] == 24000
        assert config["exaggeration"] == 0.4
        assert config["cfg_weight"] == 0.5

    def test_detect_model_type_kokoro(self):
        from hftool.tasks.text_to_speech import TextToSpeechTask
        task = TextToSpeechTask()
        assert task._detect_model_type("hexgrad/Kokoro-82M") == "kokoro"

    def test_detect_model_type_chatterbox(self):
        from hftool.tasks.text_to_speech import TextToSpeechTask
        task = TextToSpeechTask()
        assert task._detect_model_type("ResembleAI/chatterbox") == "chatterbox"

    def test_detect_model_type_bark(self):
        from hftool.tasks.text_to_speech import TextToSpeechTask
        task = TextToSpeechTask()
        assert task._detect_model_type("suno/bark-small") == "bark"

    def test_detect_model_type_mms(self):
        from hftool.tasks.text_to_speech import TextToSpeechTask
        task = TextToSpeechTask()
        assert task._detect_model_type("facebook/mms-tts-eng") == "mms-tts"
