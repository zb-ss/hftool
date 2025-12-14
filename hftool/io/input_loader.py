"""Input loading utilities for hftool.

Handles loading various input types: text, images, audio, video, URLs.
"""

import os
from enum import Enum, auto
from typing import Any, Optional, Union
from pathlib import Path


class InputType(Enum):
    """Types of input data."""
    TEXT = auto()
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()
    AUTO = auto()  # Auto-detect based on content


def detect_input_type(input_str: str) -> InputType:
    """Detect the type of input based on the string content.
    
    Args:
        input_str: Input string (could be text, file path, or URL)
    
    Returns:
        Detected InputType
    """
    # Check if it's a file path
    if os.path.exists(input_str):
        ext = os.path.splitext(input_str)[1].lower()
        
        # Image extensions
        if ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}:
            return InputType.IMAGE
        
        # Audio extensions
        if ext in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm", ".aac"}:
            return InputType.AUDIO
        
        # Video extensions
        if ext in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}:
            return InputType.VIDEO
        
        # Text files
        if ext in {".txt", ".md", ".json", ".csv"}:
            return InputType.TEXT
    
    # Check if it's a URL
    if input_str.startswith(("http://", "https://")):
        # Try to detect type from URL extension
        url_path = input_str.split("?")[0]  # Remove query params
        ext = os.path.splitext(url_path)[1].lower()
        
        if ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}:
            return InputType.IMAGE
        if ext in {".wav", ".mp3", ".flac", ".ogg"}:
            return InputType.AUDIO
        if ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            return InputType.VIDEO
        
        # Default URLs to images (common case for HF datasets)
        return InputType.IMAGE
    
    # Default to text
    return InputType.TEXT


def load_input(
    input_data: str,
    input_type: InputType = InputType.AUTO,
    **kwargs
) -> Any:
    """Load input data from various sources.
    
    Args:
        input_data: Input string (text, file path, or URL)
        input_type: Expected input type (or AUTO to detect)
        **kwargs: Additional arguments for specific loaders
    
    Returns:
        Loaded data (str, PIL.Image, audio array, etc.)
    
    Raises:
        ValueError: If input cannot be loaded
    """
    if input_type == InputType.AUTO:
        input_type = detect_input_type(input_data)
    
    if input_type == InputType.TEXT:
        return load_text(input_data)
    elif input_type == InputType.IMAGE:
        return load_image(input_data, **kwargs)
    elif input_type == InputType.AUDIO:
        return load_audio(input_data, **kwargs)
    elif input_type == InputType.VIDEO:
        return load_video(input_data, **kwargs)
    else:
        raise ValueError(f"Unknown input type: {input_type}")


def load_text(input_data: str) -> str:
    """Load text input.
    
    Args:
        input_data: Text string or path to text file
    
    Returns:
        Text content
    """
    # If it's a file, read it
    if os.path.isfile(input_data):
        with open(input_data, "r", encoding="utf-8") as f:
            return f.read()
    
    # Otherwise, treat as raw text
    return input_data


def load_image(input_data: str, **kwargs) -> Any:
    """Load image from file path or URL.
    
    Args:
        input_data: Image file path or URL
        **kwargs: Additional arguments (e.g., mode for conversion)
    
    Returns:
        PIL.Image object
    
    Raises:
        ImportError: If PIL is not available
        ValueError: If image cannot be loaded
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL is required for image loading. Install with: pip install Pillow")
    
    # URL
    if input_data.startswith(("http://", "https://")):
        try:
            import requests
            from io import BytesIO
            
            response = requests.get(input_data, timeout=30)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        except ImportError:
            raise ImportError("requests is required for URL loading. Install with: pip install requests")
        except Exception as e:
            raise ValueError(f"Failed to load image from URL: {e}")
    
    # Local file
    elif os.path.isfile(input_data):
        try:
            image = Image.open(input_data)
        except Exception as e:
            raise ValueError(f"Failed to load image from file: {e}")
    
    else:
        raise ValueError(f"Image not found: {input_data}")
    
    # Convert mode if specified
    mode = kwargs.get("mode")
    if mode and image.mode != mode:
        image = image.convert(mode)
    
    return image


def load_audio(
    input_data: str,
    sample_rate: Optional[int] = None,
    **kwargs
) -> Any:
    """Load audio from file path.
    
    Args:
        input_data: Audio file path
        sample_rate: Target sample rate (if resampling is needed)
        **kwargs: Additional arguments
    
    Returns:
        Tuple of (audio_array, sample_rate) or just the file path
        depending on what's needed by the downstream task
    """
    if not os.path.isfile(input_data):
        raise ValueError(f"Audio file not found: {input_data}")
    
    # For most transformers pipelines, just return the path
    # The pipeline will handle loading
    return input_data


def load_audio_array(
    input_data: str,
    sample_rate: int = 16000,
) -> tuple:
    """Load audio as numpy array with resampling.
    
    Args:
        input_data: Audio file path
        sample_rate: Target sample rate
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("soundfile is required for audio loading. Install with: pip install soundfile")
    
    audio, sr = sf.read(input_data)
    
    # Resample if needed
    if sr != sample_rate:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        except ImportError:
            # If librosa not available, warn but continue
            import warnings
            warnings.warn(
                f"Audio sample rate is {sr}Hz but {sample_rate}Hz is expected. "
                "Install librosa for automatic resampling."
            )
    
    return audio, sample_rate


def load_video(input_data: str, **kwargs) -> str:
    """Load video file path.
    
    Args:
        input_data: Video file path
        **kwargs: Additional arguments
    
    Returns:
        Video file path (video processing is done by downstream tasks)
    """
    if not os.path.isfile(input_data):
        raise ValueError(f"Video file not found: {input_data}")
    
    return input_data
