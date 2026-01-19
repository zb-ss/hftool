"""Tests for I/O modules."""

import os
import tempfile
import pytest


class TestInputLoader:
    """Tests for hftool.io.input_loader module."""
    
    def test_detect_input_type_text(self):
        """detect_input_type should detect plain text."""
        from hftool.io.input_loader import detect_input_type, InputType
        
        result = detect_input_type("Hello, this is some text")
        assert result == InputType.TEXT
    
    def test_detect_input_type_image_url(self):
        """detect_input_type should detect image URLs."""
        from hftool.io.input_loader import detect_input_type, InputType
        
        result = detect_input_type("https://example.com/image.jpg")
        assert result == InputType.IMAGE
        
        result = detect_input_type("https://example.com/image.png")
        assert result == InputType.IMAGE
    
    def test_detect_input_type_audio_file(self):
        """detect_input_type should detect audio files."""
        from hftool.io.input_loader import detect_input_type, InputType
        
        # Create temp audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake audio data")
            temp_path = f.name
        
        try:
            result = detect_input_type(temp_path)
            assert result == InputType.AUDIO
        finally:
            os.unlink(temp_path)
    
    def test_load_text_from_string(self):
        """load_text should return text as-is."""
        from hftool.io.input_loader import load_text
        
        text = "Hello, world!"
        result = load_text(text)
        assert result == text
    
    def test_load_text_from_file(self):
        """load_text should read from file."""
        from hftool.io.input_loader import load_text
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("File content")
            temp_path = f.name
        
        try:
            result = load_text(temp_path)
            assert result == "File content"
        finally:
            os.unlink(temp_path)
    
    def test_load_input_auto_detect(self):
        """load_input should auto-detect input type."""
        from hftool.io.input_loader import load_input, InputType
        
        # Text input
        result = load_input("Just some text", InputType.AUTO)
        assert result == "Just some text"
    
    def test_load_audio_file_not_found(self):
        """load_audio should raise for non-existent files."""
        from hftool.io.input_loader import load_audio
        
        with pytest.raises(ValueError, match="not found"):
            load_audio("/non/existent/file.wav")
    
    def test_load_video_file_not_found(self):
        """load_video should raise for non-existent files."""
        from hftool.io.input_loader import load_video
        
        with pytest.raises(ValueError, match="not found"):
            load_video("/non/existent/file.mp4")


class TestOutputHandler:
    """Tests for hftool.io.output_handler module."""
    
    def test_save_text(self):
        """save_text should write text to file."""
        from hftool.io.output_handler import save_text
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            temp_path = f.name
        
        try:
            save_text("Hello, world!", temp_path)
            
            with open(temp_path, "r") as f:
                content = f.read()
            
            assert content == "Hello, world!"
        finally:
            os.unlink(temp_path)
    
    def test_get_output_path_image(self):
        """get_output_path should generate image path."""
        from hftool.io.output_handler import get_output_path, OutputType
        
        path = get_output_path(None, OutputType.IMAGE)
        assert path.endswith(".png")
    
    def test_get_output_path_audio(self):
        """get_output_path should generate audio path."""
        from hftool.io.output_handler import get_output_path, OutputType
        
        path = get_output_path(None, OutputType.AUDIO)
        assert path.endswith(".wav")
    
    def test_get_output_path_video(self):
        """get_output_path should generate video path."""
        from hftool.io.output_handler import get_output_path, OutputType
        
        path = get_output_path(None, OutputType.VIDEO)
        assert path.endswith(".mp4")
    
    def test_get_output_path_from_input(self):
        """get_output_path should use input filename as base."""
        from hftool.io.output_handler import get_output_path, OutputType

        path = get_output_path("/path/to/input.txt", OutputType.TEXT)
        assert "input" in path


class TestVideoOutputBugFix:
    """Regression tests for video output bug fix.

    Tests verify that numpy arrays with batch dimension (1, N, H, W, C)
    are properly unwrapped before saving, preventing empty video files.
    """

    def test_convert_frame_to_pil_normal_3d_frame(self):
        """_convert_frame_to_pil should convert normal 3D frame (H, W, C)."""
        pytest.importorskip("PIL")
        pytest.importorskip("numpy")

        import numpy as np
        from hftool.io.output_handler import _convert_frame_to_pil
        from PIL import Image

        # Create 8x8 RGB frame with float values in [0, 1]
        frame = np.random.rand(8, 8, 3).astype(np.float32)

        result = _convert_frame_to_pil(frame)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (8, 8)

    def test_convert_frame_to_pil_uint8_frame(self):
        """_convert_frame_to_pil should handle uint8 frames."""
        pytest.importorskip("PIL")
        pytest.importorskip("numpy")

        import numpy as np
        from hftool.io.output_handler import _convert_frame_to_pil
        from PIL import Image

        # Create 8x8 RGB frame with uint8 values
        frame = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)

        result = _convert_frame_to_pil(frame)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_convert_frame_to_pil_channels_first(self):
        """_convert_frame_to_pil should handle channels-first format (C, H, W)."""
        pytest.importorskip("PIL")
        pytest.importorskip("numpy")

        import numpy as np
        from hftool.io.output_handler import _convert_frame_to_pil
        from PIL import Image

        # Create 3x8x8 channels-first frame
        frame = np.random.rand(3, 8, 8).astype(np.float32)

        result = _convert_frame_to_pil(frame)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (8, 8)

    def test_convert_frame_to_pil_single_batch_dimension(self):
        """_convert_frame_to_pil should unwrap single frame with batch dimension (1, H, W, C)."""
        pytest.importorskip("PIL")
        pytest.importorskip("numpy")

        import numpy as np
        from hftool.io.output_handler import _convert_frame_to_pil
        from PIL import Image

        # Create 1x8x8x3 frame (batch of 1)
        frame = np.random.rand(1, 8, 8, 3).astype(np.float32)

        result = _convert_frame_to_pil(frame)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (8, 8)

    def test_convert_frame_to_pil_rejects_5d_array(self):
        """_convert_frame_to_pil should raise ValueError for 5D arrays."""
        pytest.importorskip("numpy")

        import numpy as np
        from hftool.io.output_handler import _convert_frame_to_pil

        # Create 5D array (batch, frames, H, W, C)
        frame = np.random.rand(1, 3, 8, 8, 3).astype(np.float32)

        with pytest.raises(ValueError, match="5 dimensions.*batched video"):
            _convert_frame_to_pil(frame, frame_index=0)

    def test_convert_frame_to_pil_rejects_large_4d_array(self):
        """_convert_frame_to_pil should raise ValueError for 4D arrays with many frames."""
        pytest.importorskip("numpy")

        import numpy as np
        from hftool.io.output_handler import _convert_frame_to_pil

        # Create 4D array that looks like multiple frames (10, H, W, C)
        frame = np.random.rand(10, 8, 8, 3).astype(np.float32)

        with pytest.raises(ValueError, match="appears to be multiple frames"):
            _convert_frame_to_pil(frame, frame_index=0)

    def test_convert_frame_to_pil_handles_pil_image(self):
        """_convert_frame_to_pil should pass through PIL Images."""
        pytest.importorskip("PIL")

        from PIL import Image
        from hftool.io.output_handler import _convert_frame_to_pil

        # Create a PIL Image
        img = Image.new("RGB", (8, 8), color="red")

        result = _convert_frame_to_pil(img)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_convert_frame_to_pil_converts_rgba_to_rgb(self):
        """_convert_frame_to_pil should convert RGBA to RGB."""
        pytest.importorskip("PIL")

        from PIL import Image
        from hftool.io.output_handler import _convert_frame_to_pil

        # Create RGBA image
        img = Image.new("RGBA", (8, 8), color=(255, 0, 0, 128))

        result = _convert_frame_to_pil(img)

        assert result.mode == "RGB"

    def test_convert_frame_to_pil_handles_torch_tensor(self):
        """_convert_frame_to_pil should convert torch tensors to numpy."""
        pytest.importorskip("PIL")
        pytest.importorskip("numpy")
        torch = pytest.importorskip("torch")

        from hftool.io.output_handler import _convert_frame_to_pil
        from PIL import Image

        # Create torch tensor (H, W, C)
        tensor = torch.rand(8, 8, 3)

        result = _convert_frame_to_pil(tensor)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_save_video_with_5d_numpy_array(self):
        """save_video should unwrap 5D numpy array (1, N, H, W, C) to list of frames."""
        pytest.importorskip("PIL")
        pytest.importorskip("numpy")

        import numpy as np
        from unittest.mock import patch, MagicMock
        from hftool.io.output_handler import save_video

        # Create 5D array (batch=1, frames=3, H=8, W=8, C=3)
        frames = np.random.rand(1, 3, 8, 8, 3).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            temp_path = f.name

        try:
            # Mock ffmpeg availability and subprocess call
            with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(returncode=0, stderr="")

                    result = save_video(frames, temp_path, fps=24)

                    assert result == temp_path
                    assert mock_run.called

                    # Verify ffmpeg was called with correct parameters
                    call_args = mock_run.call_args[0][0]
                    assert "ffmpeg" in call_args
                    assert "-framerate" in call_args
                    assert "24" in call_args
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_video_with_4d_numpy_array(self):
        """save_video should convert 4D numpy array (N, H, W, C) to list of frames."""
        pytest.importorskip("PIL")
        pytest.importorskip("numpy")

        import numpy as np
        from unittest.mock import patch, MagicMock
        from hftool.io.output_handler import save_video

        # Create 4D array (frames=3, H=8, W=8, C=3)
        frames = np.random.rand(3, 8, 8, 3).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            temp_path = f.name

        try:
            # Mock ffmpeg availability and subprocess call
            with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(returncode=0, stderr="")

                    result = save_video(frames, temp_path, fps=24)

                    assert result == temp_path
                    assert mock_run.called
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_video_with_list_of_pil_images(self):
        """save_video should work with list of PIL images (backward compatibility)."""
        pytest.importorskip("PIL")

        from PIL import Image
        from unittest.mock import patch, MagicMock
        from hftool.io.output_handler import save_video

        # Create list of PIL images
        frames = [Image.new("RGB", (8, 8), color="red") for _ in range(3)]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            temp_path = f.name

        try:
            # Mock ffmpeg availability and subprocess call
            with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(returncode=0, stderr="")

                    result = save_video(frames, temp_path, fps=24)

                    assert result == temp_path
                    assert mock_run.called
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_video_with_torch_tensor(self):
        """save_video should convert torch tensor and unwrap batch dimension."""
        pytest.importorskip("PIL")
        torch = pytest.importorskip("torch")

        from unittest.mock import patch, MagicMock
        from hftool.io.output_handler import save_video

        # Create torch tensor (batch=1, frames=3, H=8, W=8, C=3)
        frames = torch.rand(1, 3, 8, 8, 3)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            temp_path = f.name

        try:
            # Mock ffmpeg availability and subprocess call
            with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(returncode=0, stderr="")

                    result = save_video(frames, temp_path, fps=24)

                    assert result == temp_path
                    assert mock_run.called
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_video_raises_on_empty_frames(self):
        """save_video should raise ValueError for empty frames."""
        from unittest.mock import patch
        from hftool.io.output_handler import save_video

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            temp_path = f.name

        try:
            # Mock ffmpeg availability so we can test the frame validation
            with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
                with pytest.raises(ValueError, match="No frames to save"):
                    save_video([], temp_path, fps=24)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_video_raises_on_none_frames(self):
        """save_video should raise ValueError for None frames."""
        from unittest.mock import patch
        from hftool.io.output_handler import save_video

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            temp_path = f.name

        try:
            # Mock ffmpeg availability so we can test the frame validation
            with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
                with pytest.raises(ValueError, match="No frames to save"):
                    save_video(None, temp_path, fps=24)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_video_respects_fps_parameter(self):
        """save_video should use custom fps parameter."""
        pytest.importorskip("PIL")
        pytest.importorskip("numpy")

        import numpy as np
        from unittest.mock import patch, MagicMock
        from hftool.io.output_handler import save_video

        frames = np.random.rand(3, 8, 8, 3).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            temp_path = f.name

        try:
            # Mock ffmpeg availability and subprocess call
            with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(returncode=0, stderr="")

                    save_video(frames, temp_path, fps=30)

                    call_args = mock_run.call_args[0][0]
                    fps_index = call_args.index("-framerate")
                    assert call_args[fps_index + 1] == "30"
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_video_respects_codec_parameter(self):
        """save_video should use custom codec parameter."""
        pytest.importorskip("PIL")
        pytest.importorskip("numpy")

        import numpy as np
        from unittest.mock import patch, MagicMock
        from hftool.io.output_handler import save_video

        frames = np.random.rand(3, 8, 8, 3).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            temp_path = f.name

        try:
            # Mock ffmpeg availability and subprocess call
            with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(returncode=0, stderr="")

                    save_video(frames, temp_path, codec="libx265")

                    call_args = mock_run.call_args[0][0]
                    codec_index = call_args.index("-c:v")
                    assert call_args[codec_index + 1] == "libx265"
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_video_raises_on_ffmpeg_failure(self):
        """save_video should raise RuntimeError when ffmpeg fails."""
        pytest.importorskip("PIL")
        pytest.importorskip("numpy")

        import numpy as np
        from unittest.mock import patch
        from subprocess import CalledProcessError
        from hftool.io.output_handler import save_video

        frames = np.random.rand(3, 8, 8, 3).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            temp_path = f.name

        try:
            # Mock ffmpeg availability and subprocess call to fail
            with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
                with patch("subprocess.run") as mock_run:
                    mock_run.side_effect = CalledProcessError(1, "ffmpeg", stderr="encoding error")

                    with pytest.raises(RuntimeError, match="ffmpeg encoding failed"):
                        save_video(frames, temp_path, fps=24)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
