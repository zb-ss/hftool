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
