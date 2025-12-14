"""Output handling utilities for hftool.

Handles saving various output types: images, video, audio.
"""

import os
import subprocess
import tempfile
from enum import Enum, auto
from typing import Any, List, Optional, Union
from pathlib import Path


class OutputType(Enum):
    """Types of output data."""
    TEXT = auto()
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()


def save_output(
    result: Any,
    output_path: str,
    output_type: OutputType,
    **kwargs
) -> str:
    """Save output data to file.
    
    Args:
        result: Data to save
        output_path: Path to save to
        output_type: Type of output
        **kwargs: Additional arguments for specific savers
    
    Returns:
        Path to saved file
    """
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    if output_type == OutputType.TEXT:
        return save_text(result, output_path)
    elif output_type == OutputType.IMAGE:
        return save_image(result, output_path, **kwargs)
    elif output_type == OutputType.AUDIO:
        return save_audio(result, output_path, **kwargs)
    elif output_type == OutputType.VIDEO:
        return save_video(result, output_path, **kwargs)
    else:
        raise ValueError(f"Unknown output type: {output_type}")


def save_text(text: str, output_path: str) -> str:
    """Save text to file.
    
    Args:
        text: Text content
        output_path: Path to save to
    
    Returns:
        Path to saved file
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return output_path


def save_image(image: Any, output_path: str, **kwargs) -> str:
    """Save image to file.
    
    Args:
        image: PIL.Image object or numpy array
        output_path: Path to save to (extension determines format)
        **kwargs: Additional arguments (quality, etc.)
    
    Returns:
        Path to saved file
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL is required for image saving. Install with: pip install Pillow")
    
    # Convert numpy array to PIL Image if needed
    if not isinstance(image, Image.Image):
        try:
            import numpy as np
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            else:
                raise ValueError(f"Cannot save image of type {type(image).__name__}")
        except ImportError:
            raise ValueError(f"Cannot save image of type {type(image).__name__}")
    
    # Get save kwargs
    save_kwargs = {}
    if "quality" in kwargs:
        save_kwargs["quality"] = kwargs["quality"]
    
    # Determine format from extension
    ext = os.path.splitext(output_path)[1].lower()
    if ext in {".jpg", ".jpeg"}:
        # Convert RGBA to RGB for JPEG
        if image.mode == "RGBA":
            image = image.convert("RGB")
        save_kwargs.setdefault("quality", 95)
    
    image.save(output_path, **save_kwargs)
    return output_path


def save_audio(
    audio: Any,
    output_path: str,
    sample_rate: int = 24000,
    **kwargs
) -> str:
    """Save audio to file.
    
    Args:
        audio: Audio data (numpy array or dict with 'audio' and 'sampling_rate')
        output_path: Path to save to (extension determines format)
        sample_rate: Sample rate (if not provided in audio dict)
        **kwargs: Additional arguments
    
    Returns:
        Path to saved file
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("soundfile is required for audio saving. Install with: pip install soundfile")
    
    import numpy as np
    
    # Handle different audio formats
    if isinstance(audio, dict):
        # Common format from TTS pipelines: {"audio": array, "sampling_rate": int}
        audio_data = audio.get("audio", audio.get("waveform"))
        sample_rate = audio.get("sampling_rate", audio.get("sample_rate", sample_rate))
    elif isinstance(audio, (list, tuple)) and len(audio) == 2:
        # Tuple of (audio_array, sample_rate)
        audio_data, sample_rate = audio
    else:
        audio_data = audio
    
    # Convert to numpy if needed
    if hasattr(audio_data, "cpu"):  # torch tensor
        audio_data = audio_data.cpu().numpy()
    
    # Ensure proper shape
    audio_data = np.asarray(audio_data)
    if audio_data.ndim > 1:
        audio_data = audio_data.squeeze()
    
    # Normalize if needed (some models output in different ranges)
    if audio_data.max() > 1.0 or audio_data.min() < -1.0:
        audio_data = audio_data / max(abs(audio_data.max()), abs(audio_data.min()))
    
    ext = os.path.splitext(output_path)[1].lower()
    
    if ext == ".mp3":
        # Save as WAV first, then convert to MP3 using ffmpeg
        wav_path = output_path.replace(".mp3", ".wav")
        sf.write(wav_path, audio_data, sample_rate)
        
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", wav_path, "-b:a", "192k", output_path],
                check=True,
                capture_output=True,
            )
            os.remove(wav_path)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            # If ffmpeg fails, keep the WAV file
            import warnings
            warnings.warn(f"Failed to convert to MP3 (ffmpeg error), keeping WAV: {wav_path}")
            return wav_path
    else:
        # Save directly (WAV, FLAC, OGG, etc.)
        sf.write(output_path, audio_data, sample_rate)
    
    return output_path


def save_video(
    frames: List[Any],
    output_path: str,
    fps: int = 24,
    **kwargs
) -> str:
    """Save video frames to MP4 file using ffmpeg.
    
    Args:
        frames: List of PIL.Image objects or numpy arrays
        output_path: Path to save to
        fps: Frames per second
        **kwargs: Additional arguments (quality, codec, etc.)
    
    Returns:
        Path to saved file
    
    Raises:
        RuntimeError: If ffmpeg is not available
    """
    import shutil
    
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is required for video saving but was not found. "
            "Please install ffmpeg: https://ffmpeg.org/download.html"
        )
    
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL is required for video saving. Install with: pip install Pillow")
    
    # Get encoding settings
    crf = kwargs.get("crf", 23)  # Quality (lower = better, 18-28 is reasonable)
    codec = kwargs.get("codec", "libx264")
    pix_fmt = kwargs.get("pix_fmt", "yuv420p")
    
    # Create temporary directory for frames
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save frames as images
        for i, frame in enumerate(frames):
            # Convert to PIL Image if needed
            if not isinstance(frame, Image.Image):
                try:
                    import numpy as np
                    if isinstance(frame, np.ndarray):
                        frame = Image.fromarray(frame)
                    elif hasattr(frame, "cpu"):  # torch tensor
                        frame = Image.fromarray(frame.cpu().numpy())
                except Exception as e:
                    raise ValueError(f"Cannot convert frame {i} to image: {e}")
            
            # Ensure RGB mode
            if frame.mode != "RGB":
                frame = frame.convert("RGB")
            
            frame.save(os.path.join(tmpdir, f"frame_{i:06d}.png"))
        
        # Encode with ffmpeg
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmpdir, "frame_%06d.png"),
            "-c:v", codec,
            "-crf", str(crf),
            "-pix_fmt", pix_fmt,
            output_path,
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg encoding failed: {e.stderr}")
    
    return output_path


def get_output_path(
    input_path: Optional[str],
    output_type: OutputType,
    output_dir: str = ".",
    suffix: str = "_output"
) -> str:
    """Generate an output path based on input path and output type.
    
    Args:
        input_path: Optional input file path
        output_type: Type of output
        output_dir: Directory for output
        suffix: Suffix to add before extension
    
    Returns:
        Generated output path
    """
    # Default extensions for each type
    extensions = {
        OutputType.TEXT: ".txt",
        OutputType.IMAGE: ".png",
        OutputType.AUDIO: ".wav",
        OutputType.VIDEO: ".mp4",
    }
    
    ext = extensions.get(output_type, ".bin")
    
    if input_path:
        base = os.path.splitext(os.path.basename(input_path))[0]
        return os.path.join(output_dir, f"{base}{suffix}{ext}")
    else:
        import time
        timestamp = int(time.time())
        return os.path.join(output_dir, f"output_{timestamp}{ext}")
