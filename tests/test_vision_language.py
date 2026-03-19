"""Tests for the vision-language task handler."""

import gc
import os
import tempfile
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


class TestVisionLanguageTaskFactory:
    """Tests for the create_task() factory function."""

    def test_create_task_returns_instance(self):
        from hftool.tasks.vision_language import VisionLanguageTask, create_task

        task = create_task()
        assert isinstance(task, VisionLanguageTask)

    def test_create_task_with_device(self):
        from hftool.tasks.vision_language import create_task

        task = create_task(device="cpu")
        assert task.device == "cpu"

    def test_create_task_default_device(self):
        from hftool.tasks.vision_language import create_task

        task = create_task()
        assert task.device == "auto"

    def test_create_task_with_dtype(self):
        from hftool.tasks.vision_language import create_task

        task = create_task(dtype="float16")
        assert task.dtype == "float16"

    def test_initial_state(self):
        from hftool.tasks.vision_language import VisionLanguageTask

        task = VisionLanguageTask()
        assert task._model is None
        assert task._processor is None
        assert task._pipeline is None


class TestVisionLanguageTaskRegistry:
    """Tests for vision-language entries in TASK_REGISTRY and MODEL_REGISTRY."""

    def test_vlm_task_registered(self):
        from hftool.core.registry import TASK_REGISTRY

        assert "vision-language" in TASK_REGISTRY

    def test_vlm_alias(self):
        from hftool.core.registry import TASK_ALIASES

        assert "vlm" in TASK_ALIASES
        assert TASK_ALIASES["vlm"] == "vision-language"

    def test_vlm_models_registered(self):
        from hftool.core.models import MODEL_REGISTRY

        assert "vision-language" in MODEL_REGISTRY

    def test_qwen35_9b_registered(self):
        from hftool.core.models import MODEL_REGISTRY

        assert "qwen3.5-9b" in MODEL_REGISTRY["vision-language"]

    def test_qwen35_9b_is_default(self):
        from hftool.core.models import MODEL_REGISTRY

        model = MODEL_REGISTRY["vision-language"]["qwen3.5-9b"]
        assert model.is_default is True

    def test_only_one_default_model(self):
        from hftool.core.models import MODEL_REGISTRY

        defaults = [
            name
            for name, info in MODEL_REGISTRY["vision-language"].items()
            if info.is_default
        ]
        assert len(defaults) == 1

    def test_qwen35_models_have_pip_deps(self):
        from hftool.core.models import MODEL_REGISTRY

        for key in ("qwen3.5-9b", "qwen3.5-4b", "qwen3.5-27b"):
            model = MODEL_REGISTRY["vision-language"][key]
            dep_str = " ".join(model.pip_dependencies)
            assert "qwen-vl-utils" in dep_str, f"{key} missing qwen-vl-utils dependency"

    def test_vlm_task_library_is_transformers(self):
        from hftool.core.registry import TASK_REGISTRY

        config = TASK_REGISTRY["vision-language"]
        assert config.library == "transformers"

    def test_vlm_task_input_type(self):
        from hftool.core.registry import TASK_REGISTRY

        config = TASK_REGISTRY["vision-language"]
        assert config.input_type == "image"

    def test_vlm_task_output_type(self):
        from hftool.core.registry import TASK_REGISTRY

        config = TASK_REGISTRY["vision-language"]
        assert config.output_type == "text"

    def test_vlm_task_handler_path(self):
        from hftool.core.registry import TASK_REGISTRY

        config = TASK_REGISTRY["vision-language"]
        assert config.handler == "hftool.tasks.vision_language"


class TestVisionLanguageAnalyzeFrame:
    """Tests for the analyze_frame() convenience method."""

    def _make_task_with_pipeline(self):
        from hftool.tasks.vision_language import VisionLanguageTask

        task = VisionLanguageTask(device="cpu")
        task._pipeline = {
            "type": "qwen3.5",
            "model": MagicMock(),
            "processor": MagicMock(),
            "device": "cpu",
        }
        return task

    def test_substitutes_previous_description(self):
        task = self._make_task_with_pipeline()

        captured_input = {}

        def fake_run_inference(pipeline, input_data, **kwargs):
            captured_input.update(input_data)
            return {"text": "a scene"}

        task.run_inference = fake_run_inference

        prompt_template = "Describe what you see. Previous: {previous_description}"
        task.analyze_frame("/fake/frame.jpg", prompt_template, previous_context="a cat")

        assert "previous_description" not in captured_input["prompt"]
        assert "a cat" in captured_input["prompt"]

    def test_no_previous_context_leaves_placeholder_empty(self):
        task = self._make_task_with_pipeline()

        captured_input = {}

        def fake_run_inference(pipeline, input_data, **kwargs):
            captured_input.update(input_data)
            return {"text": "first frame"}

        task.run_inference = fake_run_inference

        task.analyze_frame("/fake/frame.jpg", "Describe: {previous_description}", previous_context="")

        assert captured_input["prompt"] == "Describe: "

    def test_returns_string(self):
        task = self._make_task_with_pipeline()

        task.run_inference = MagicMock(return_value={"text": "a bright sunny day"})

        result = task.analyze_frame("/fake/frame.jpg", "Describe this image.")

        assert isinstance(result, str)
        assert result == "a bright sunny day"

    def test_passes_image_path_to_run_inference(self):
        task = self._make_task_with_pipeline()

        captured_input = {}

        def fake_run_inference(pipeline, input_data, **kwargs):
            captured_input.update(input_data)
            return {"text": "ok"}

        task.run_inference = fake_run_inference

        task.analyze_frame("/path/to/image.png", "What is this?")

        assert captured_input["image_path"] == "/path/to/image.png"

    def test_prompt_without_placeholder_unchanged(self):
        task = self._make_task_with_pipeline()

        captured_input = {}

        def fake_run_inference(pipeline, input_data, **kwargs):
            captured_input.update(input_data)
            return {"text": "result"}

        task.run_inference = fake_run_inference

        task.analyze_frame("/img.jpg", "Simple prompt.", previous_context="ignored")

        assert captured_input["prompt"] == "Simple prompt."


class TestVisionLanguageCleanup:
    """Tests for the cleanup() method."""

    def _make_loaded_task(self):
        from hftool.tasks.vision_language import VisionLanguageTask

        task = VisionLanguageTask(device="cpu")
        task._model = MagicMock(name="model")
        task._processor = MagicMock(name="processor")
        task._pipeline = {"type": "qwen3.5", "model": task._model}
        return task

    def test_cleanup_sets_model_to_none(self):
        task = self._make_loaded_task()
        task.cleanup()
        assert task._model is None

    def test_cleanup_sets_processor_to_none(self):
        task = self._make_loaded_task()
        task.cleanup()
        assert task._processor is None

    def test_cleanup_sets_pipeline_to_none(self):
        task = self._make_loaded_task()
        task.cleanup()
        assert task._pipeline is None

    def test_cleanup_sets_none(self):
        task = self._make_loaded_task()
        task.cleanup()
        assert task._model is None
        assert task._processor is None
        assert task._pipeline is None

    def test_cleanup_calls_gc_collect(self):
        task = self._make_loaded_task()

        with patch("gc.collect") as mock_gc:
            task.cleanup()
            mock_gc.assert_called_once()

    def test_cleanup_calls_cuda_empty_cache(self):
        task = self._make_loaded_task()

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            task.cleanup()

        mock_torch.cuda.empty_cache.assert_called_once()

    def test_cleanup_skips_cuda_cache_when_cuda_unavailable(self):
        task = self._make_loaded_task()

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            task.cleanup()

        mock_torch.cuda.empty_cache.assert_not_called()

    def test_cleanup_handles_no_torch(self):
        task = self._make_loaded_task()

        with patch.dict("sys.modules", {"torch": None}):
            # Should not raise even if torch import fails
            try:
                task.cleanup()
            except ImportError:
                pytest.fail("cleanup() raised ImportError when torch is unavailable")

        assert task._model is None
        assert task._processor is None
        assert task._pipeline is None

    def test_cleanup_idempotent(self):
        task = self._make_loaded_task()
        task.cleanup()
        # Second call should not raise
        task.cleanup()
        assert task._model is None

    def test_cleanup_with_none_model_does_not_raise(self):
        from hftool.tasks.vision_language import VisionLanguageTask

        task = VisionLanguageTask(device="cpu")
        # _model and _processor are already None
        task.cleanup()  # must not raise
        assert task._model is None
        assert task._processor is None


class TestVisionLanguageSaveOutput:
    """Tests for save_output()."""

    def test_saves_text_to_file(self):
        from hftool.tasks.vision_language import VisionLanguageTask

        task = VisionLanguageTask(device="cpu")

        with tempfile.NamedTemporaryFile(mode="r", suffix=".txt", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = task.save_output({"text": "The image shows a mountain."}, tmp_path)

            with open(tmp_path, "r", encoding="utf-8") as fh:
                contents = fh.read()

            assert contents == "The image shows a mountain."
            assert result == tmp_path
        finally:
            os.unlink(tmp_path)

    def test_save_output_returns_path(self):
        from hftool.tasks.vision_language import VisionLanguageTask

        task = VisionLanguageTask(device="cpu")

        with tempfile.NamedTemporaryFile(mode="r", suffix=".txt", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            returned = task.save_output({"text": "hello"}, tmp_path)
            assert returned == tmp_path
        finally:
            os.unlink(tmp_path)

    def test_save_output_utf8_content(self):
        from hftool.tasks.vision_language import VisionLanguageTask

        task = VisionLanguageTask(device="cpu")
        text = "Résumé: 图像描述 – Ñoño"

        with tempfile.NamedTemporaryFile(mode="r", suffix=".txt", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            task.save_output({"text": text}, tmp_path)

            with open(tmp_path, "r", encoding="utf-8") as fh:
                contents = fh.read()

            assert contents == text
        finally:
            os.unlink(tmp_path)

    def test_save_output_empty_text(self):
        from hftool.tasks.vision_language import VisionLanguageTask

        task = VisionLanguageTask(device="cpu")

        with tempfile.NamedTemporaryFile(mode="r", suffix=".txt", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            task.save_output({"text": ""}, tmp_path)

            with open(tmp_path, "r", encoding="utf-8") as fh:
                contents = fh.read()

            assert contents == ""
        finally:
            os.unlink(tmp_path)
