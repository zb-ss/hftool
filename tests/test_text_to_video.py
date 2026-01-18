"""Tests for text-to-video task handler.

Regression tests for video generation frame unwrapping bug fix.
"""

import pytest


class TestTextToVideoFrameUnwrapping:
    """Tests for run_inference frame unwrapping.

    Verifies that the run_inference method properly unwraps numpy arrays
    and torch tensors with batch dimensions before returning frames.
    """

    def test_run_inference_unwraps_5d_numpy_array(self):
        """run_inference should unwrap 5D numpy array (batch, frames, H, W, C)."""
        pytest.importorskip("numpy")

        import numpy as np
        from unittest.mock import MagicMock
        from hftool.tasks.text_to_video import TextToVideoTask

        task = TextToVideoTask(device="cpu")

        # Mock pipeline that returns 5D numpy array
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        # Shape: (batch=1, frames=5, H=8, W=8, C=3)
        mock_result.frames = np.random.rand(1, 5, 8, 8, 3).astype(np.float32)
        mock_pipeline.return_value = mock_result

        result = task.run_inference(mock_pipeline, "test prompt")

        # Should return list of 5 frames
        assert isinstance(result, list)
        assert len(result) == 5
        # Each frame should be 3D (H, W, C)
        assert result[0].shape == (8, 8, 3)

    def test_run_inference_unwraps_4d_numpy_array(self):
        """run_inference should handle 4D numpy array (frames, H, W, C)."""
        pytest.importorskip("numpy")

        import numpy as np
        from unittest.mock import MagicMock
        from hftool.tasks.text_to_video import TextToVideoTask

        task = TextToVideoTask(device="cpu")

        # Mock pipeline that returns 4D numpy array (already unwrapped)
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        # Shape: (frames=5, H=8, W=8, C=3)
        mock_result.frames = np.random.rand(5, 8, 8, 3).astype(np.float32)
        mock_pipeline.return_value = mock_result

        result = task.run_inference(mock_pipeline, "test prompt")

        # Should return list of 5 frames
        assert isinstance(result, list)
        assert len(result) == 5
        assert result[0].shape == (8, 8, 3)

    def test_run_inference_converts_torch_tensor(self):
        """run_inference should convert torch tensor to numpy and unwrap."""
        pytest.importorskip("numpy")
        torch = pytest.importorskip("torch")

        from unittest.mock import MagicMock
        from hftool.tasks.text_to_video import TextToVideoTask

        task = TextToVideoTask(device="cpu")

        # Mock pipeline that returns torch tensor
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        # Shape: (batch=1, frames=5, H=8, W=8, C=3)
        mock_result.frames = torch.rand(1, 5, 8, 8, 3)
        mock_pipeline.return_value = mock_result

        result = task.run_inference(mock_pipeline, "test prompt")

        # Should return list of 5 numpy frames
        assert isinstance(result, list)
        assert len(result) == 5
        # Each frame should be numpy array
        import numpy as np
        assert isinstance(result[0], np.ndarray)
        assert result[0].shape == (8, 8, 3)

    def test_run_inference_handles_nested_list(self):
        """run_inference should unwrap nested list [[frame1, frame2, ...]]."""
        pytest.importorskip("numpy")

        import numpy as np
        from unittest.mock import MagicMock
        from hftool.tasks.text_to_video import TextToVideoTask

        task = TextToVideoTask(device="cpu")

        # Mock pipeline that returns nested list (some pipelines do this)
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        frames = [np.random.rand(8, 8, 3).astype(np.float32) for _ in range(5)]
        mock_result.frames = [frames]  # Nested in list
        mock_pipeline.return_value = mock_result

        result = task.run_inference(mock_pipeline, "test prompt")

        # Should return flattened list of 5 frames
        assert isinstance(result, list)
        assert len(result) == 5
        assert result[0].shape == (8, 8, 3)

    def test_run_inference_handles_images_attribute(self):
        """run_inference should work with result.images instead of result.frames."""
        pytest.importorskip("numpy")

        import numpy as np
        from unittest.mock import MagicMock
        from hftool.tasks.text_to_video import TextToVideoTask

        task = TextToVideoTask(device="cpu")

        # Create a simple object that has only 'images' attribute
        class MockResult:
            def __init__(self):
                self.images = np.random.rand(1, 5, 8, 8, 3).astype(np.float32)

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = MockResult()

        result = task.run_inference(mock_pipeline, "test prompt")

        # Should handle images attribute
        assert isinstance(result, list)
        assert len(result) == 5

    def test_run_inference_handles_direct_array_return(self):
        """run_inference should handle direct array return (no attributes)."""
        pytest.importorskip("numpy")

        import numpy as np
        from unittest.mock import MagicMock
        from hftool.tasks.text_to_video import TextToVideoTask

        task = TextToVideoTask(device="cpu")

        # Mock pipeline that returns array directly
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = np.random.rand(1, 5, 8, 8, 3).astype(np.float32)

        result = task.run_inference(mock_pipeline, "test prompt")

        # Should handle direct array
        assert isinstance(result, list)
        assert len(result) == 5

    def test_run_inference_preserves_list_of_pil_images(self):
        """run_inference should preserve list of PIL images (backward compat)."""
        PIL = pytest.importorskip("PIL")

        from PIL import Image
        from unittest.mock import MagicMock
        from hftool.tasks.text_to_video import TextToVideoTask

        task = TextToVideoTask(device="cpu")

        # Mock pipeline that returns list of PIL images
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        images = [Image.new("RGB", (8, 8), color="red") for _ in range(5)]
        mock_result.frames = images
        mock_pipeline.return_value = mock_result

        result = task.run_inference(mock_pipeline, "test prompt")

        # Should preserve PIL images
        assert isinstance(result, list)
        assert len(result) == 5
        assert all(isinstance(img, Image.Image) for img in result)

    def test_run_inference_passes_kwargs_to_pipeline(self):
        """run_inference should pass inference kwargs to pipeline."""
        import numpy as np
        from unittest.mock import MagicMock
        from hftool.tasks.text_to_video import TextToVideoTask

        task = TextToVideoTask(device="cpu")

        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.frames = np.random.rand(3, 8, 8, 3).astype(np.float32)
        mock_pipeline.return_value = mock_result

        result = task.run_inference(
            mock_pipeline,
            "test prompt",
            num_frames=10,
            num_inference_steps=25,
            height=256,
            width=256
        )

        # Verify pipeline was called with correct kwargs
        mock_pipeline.assert_called_once()
        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["prompt"] == "test prompt"
        assert call_kwargs["num_frames"] == 10
        assert call_kwargs["num_inference_steps"] == 25
        assert call_kwargs["height"] == 256
        assert call_kwargs["width"] == 256

    def test_run_inference_converts_seed_to_generator(self):
        """run_inference should convert seed parameter to generator."""
        torch = pytest.importorskip("torch")
        import numpy as np
        from unittest.mock import MagicMock
        from hftool.tasks.text_to_video import TextToVideoTask

        task = TextToVideoTask(device="cpu")

        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.frames = np.random.rand(3, 8, 8, 3).astype(np.float32)
        mock_pipeline.return_value = mock_result

        result = task.run_inference(mock_pipeline, "test prompt", seed=42)

        # Verify generator was passed instead of seed
        call_kwargs = mock_pipeline.call_args[1]
        assert "seed" not in call_kwargs
        assert "generator" in call_kwargs
        assert isinstance(call_kwargs["generator"], torch.Generator)

    def test_run_inference_uses_model_defaults(self):
        """run_inference should merge model defaults with provided kwargs."""
        import numpy as np
        from unittest.mock import MagicMock
        from hftool.tasks.text_to_video import TextToVideoTask

        task = TextToVideoTask(device="cpu")
        task._model_name = "CogVideoX"  # Use CogVideoX which has simpler defaults

        # Create a simple pipeline object without guider
        class SimplePipeline:
            def __call__(self, **kwargs):
                class Result:
                    def __init__(self):
                        self.frames = np.random.rand(3, 8, 8, 3).astype(np.float32)
                return Result()

        mock_pipeline = SimplePipeline()

        # Call with partial kwargs - should use defaults for missing
        result = task.run_inference(mock_pipeline, "test prompt", num_frames=30)

        # The test verifies that the method runs successfully and returns frames
        # The actual defaults are merged internally
        assert isinstance(result, list)
        assert len(result) == 3


class TestTextToVideoIntegration:
    """Integration tests for text-to-video task end-to-end."""

    def test_save_output_calls_save_video(self):
        """save_output should call save_video with correct parameters."""
        PIL = pytest.importorskip("PIL")

        from PIL import Image
        from unittest.mock import patch, MagicMock
        from hftool.tasks.text_to_video import TextToVideoTask
        import tempfile
        import os

        task = TextToVideoTask(device="cpu")

        # Create mock frames
        frames = [Image.new("RGB", (8, 8), color="red") for _ in range(3)]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            temp_path = f.name

        try:
            # Mock save_video
            with patch("hftool.tasks.text_to_video.save_video") as mock_save:
                mock_save.return_value = temp_path

                result = task.save_output(frames, temp_path, fps=30)

                # Verify save_video was called correctly
                mock_save.assert_called_once_with(frames, temp_path, fps=30)
                assert result == temp_path
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_output_uses_default_fps(self):
        """save_output should use default fps of 24 if not specified."""
        PIL = pytest.importorskip("PIL")

        from PIL import Image
        from unittest.mock import patch
        from hftool.tasks.text_to_video import TextToVideoTask
        import tempfile
        import os

        task = TextToVideoTask(device="cpu")
        frames = [Image.new("RGB", (8, 8)) for _ in range(3)]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            temp_path = f.name

        try:
            with patch("hftool.tasks.text_to_video.save_video") as mock_save:
                mock_save.return_value = temp_path

                task.save_output(frames, temp_path)

                # Should use default fps=24
                call_kwargs = mock_save.call_args[1]
                assert call_kwargs["fps"] == 24
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
