"""Metadata embedding for hftool output files.

Embeds generation parameters in output files for reproducibility and reference.
- PNG: PIL tEXt chunks (lossless, preserves metadata)
- JPEG: EXIF UserComment
- Audio/Video: Sidecar .json files
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List
import click


def get_hftool_version() -> str:
    """Get hftool version.
    
    Returns:
        Version string
    """
    try:
        from importlib.metadata import version
        return version("hftool")
    except Exception:
        return "unknown"


def create_metadata(
    task: str,
    model: str,
    prompt: Optional[str] = None,
    seed: Optional[int] = None,
    extra_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create metadata dictionary for embedding.
    
    Args:
        task: Task name
        model: Model name or repo_id
        prompt: Input prompt/text
        seed: Random seed
        extra_params: Additional generation parameters
    
    Returns:
        Metadata dictionary
    """
    metadata: Dict[str, Any] = {
        "hftool_version": get_hftool_version(),
        "task": task,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    
    if prompt is not None:
        metadata["prompt"] = prompt
    
    if seed is not None:
        metadata["seed"] = seed
    
    if extra_params:
        # Flatten common generation parameters
        for key in ["num_inference_steps", "guidance_scale", "width", "height", 
                    "negative_prompt", "strength", "steps", "cfg_scale"]:
            if key in extra_params:
                value = extra_params[key]
                # Convert to JSON-serializable types
                if isinstance(value, (str, int, float, bool)) or value is None:
                    metadata[key] = value
                else:
                    metadata[key] = str(value)
        
        # Store all extra params as well (convert non-serializable types)
        serializable_params = {}
        for k, v in extra_params.items():
            if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                serializable_params[k] = v
            else:
                serializable_params[k] = str(v)
        metadata["generation_params"] = serializable_params
    
    return metadata


def embed_metadata_png(file_path: str, metadata: Dict[str, Any], verbose: bool = False) -> bool:
    """Embed metadata in PNG file using tEXt chunks.
    
    Args:
        file_path: Path to PNG file
        metadata: Metadata dictionary
        verbose: Whether to print verbose messages
    
    Returns:
        True if successful, False otherwise
    """
    try:
        from PIL import Image, PngImagePlugin
        
        # Load image
        img = Image.open(file_path)
        
        # Create PNG info object
        png_info = PngImagePlugin.PngInfo()
        
        # Add hftool metadata as JSON in a custom chunk
        png_info.add_text("hftool", json.dumps(metadata))
        
        # Also add individual common fields for compatibility
        png_info.add_text("Software", f"hftool {metadata.get('hftool_version', 'unknown')}")
        
        if "prompt" in metadata:
            png_info.add_text("Description", metadata["prompt"])
        
        if "model" in metadata:
            png_info.add_text("Model", metadata["model"])
        
        if "seed" in metadata:
            png_info.add_text("Seed", str(metadata["seed"]))
        
        # Save with metadata
        img.save(file_path, pnginfo=png_info)
        
        if verbose:
            click.echo(f"  Embedded metadata in PNG: {file_path}")
        
        return True
        
    except Exception as e:
        if verbose:
            click.echo(f"  Warning: Failed to embed PNG metadata: {e}", err=True)
        return False


def embed_metadata_jpeg(file_path: str, metadata: Dict[str, Any], verbose: bool = False) -> bool:
    """Embed metadata in JPEG file using EXIF UserComment.
    
    Args:
        file_path: Path to JPEG file
        metadata: Metadata dictionary
        verbose: Whether to print verbose messages
    
    Returns:
        True if successful, False otherwise
    """
    try:
        from PIL import Image
        import piexif
        
        # Load image
        img = Image.open(file_path)
        
        # Get existing EXIF data or create new
        try:
            exif_dict = piexif.load(img.info.get("exif", b""))
        except Exception:
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}
        
        # Add metadata as JSON in UserComment
        metadata_json = json.dumps(metadata)
        exif_dict["Exif"][piexif.ExifIFD.UserComment] = metadata_json.encode("utf-8")
        
        # Add Software tag
        exif_dict["0th"][piexif.ImageIFD.Software] = f"hftool {metadata.get('hftool_version', 'unknown')}".encode("utf-8")
        
        # Add ImageDescription (prompt)
        if "prompt" in metadata:
            exif_dict["0th"][piexif.ImageIFD.ImageDescription] = metadata["prompt"][:255].encode("utf-8")
        
        # Convert to bytes
        exif_bytes = piexif.dump(exif_dict)
        
        # Save with EXIF
        img.save(file_path, exif=exif_bytes, quality=95)
        
        if verbose:
            click.echo(f"  Embedded metadata in JPEG: {file_path}")
        
        return True
        
    except ImportError:
        if verbose:
            click.echo("  Warning: piexif not installed. JPEG metadata not embedded.", err=True)
            click.echo("  Install with: pip install piexif", err=True)
        return False
    except Exception as e:
        if verbose:
            click.echo(f"  Warning: Failed to embed JPEG metadata: {e}", err=True)
        return False


def embed_metadata_sidecar(file_path: str, metadata: Dict[str, Any], verbose: bool = False) -> bool:
    """Create a sidecar .json file with metadata.
    
    Used for audio/video files and as fallback for other formats.
    
    Args:
        file_path: Path to media file
        metadata: Metadata dictionary
        verbose: Whether to print verbose messages
    
    Returns:
        True if successful, False otherwise
    """
    try:
        sidecar_path = Path(file_path).with_suffix(Path(file_path).suffix + ".json")
        
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        if verbose:
            click.echo(f"  Created metadata sidecar: {sidecar_path}")
        
        return True
        
    except Exception as e:
        if verbose:
            click.echo(f"  Warning: Failed to create sidecar file: {e}", err=True)
        return False


def embed_metadata(
    file_path: str,
    task: str,
    model: str,
    prompt: Optional[str] = None,
    seed: Optional[int] = None,
    extra_params: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> bool:
    """Embed metadata in output file.
    
    Automatically determines the best method based on file extension.
    
    Args:
        file_path: Path to output file
        task: Task name
        model: Model name or repo_id
        prompt: Input prompt/text
        seed: Random seed
        extra_params: Additional generation parameters
        verbose: Whether to print verbose messages
    
    Returns:
        True if successful, False otherwise
    """
    # Security: Validate file path exists and is a regular file (M-3)
    if not os.path.exists(file_path):
        if verbose:
            click.echo(f"  Warning: File not found for metadata embedding: {file_path}", err=True)
        return False
    
    if not os.path.isfile(file_path):
        if verbose:
            click.echo(f"  Warning: Not a regular file: {file_path}", err=True)
        return False
    
    # Create metadata
    metadata = create_metadata(task, model, prompt, seed, extra_params)
    
    # Determine file type
    ext = Path(file_path).suffix.lower()
    
    if ext == ".png":
        return embed_metadata_png(file_path, metadata, verbose)
    elif ext in {".jpg", ".jpeg"}:
        return embed_metadata_jpeg(file_path, metadata, verbose)
    elif ext in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        # Audio/video - use sidecar
        return embed_metadata_sidecar(file_path, metadata, verbose)
    else:
        # Unknown format - try sidecar as fallback
        if verbose:
            click.echo(f"  Unknown file format '{ext}', using sidecar file", err=True)
        return embed_metadata_sidecar(file_path, metadata, verbose)


def read_metadata_png(file_path: str) -> Optional[Dict[str, Any]]:
    """Read metadata from PNG file.
    
    Args:
        file_path: Path to PNG file
    
    Returns:
        Metadata dictionary or None if not found
    """
    try:
        from PIL import Image
        
        img = Image.open(file_path)
        
        # Try to get hftool metadata
        if "hftool" in img.info:
            return json.loads(img.info["hftool"])
        
        # Fall back to individual fields
        metadata = {}
        for key in ["Software", "Description", "Model", "Seed"]:
            if key in img.info:
                metadata[key.lower()] = img.info[key]
        
        return metadata if metadata else None
        
    except Exception:
        return None


def read_metadata_jpeg(file_path: str) -> Optional[Dict[str, Any]]:
    """Read metadata from JPEG file.
    
    Args:
        file_path: Path to JPEG file
    
    Returns:
        Metadata dictionary or None if not found
    """
    try:
        from PIL import Image
        import piexif
        
        img = Image.open(file_path)
        exif_dict = piexif.load(img.info.get("exif", b""))
        
        # Try to get UserComment
        if piexif.ExifIFD.UserComment in exif_dict.get("Exif", {}):
            comment = exif_dict["Exif"][piexif.ExifIFD.UserComment]
            if isinstance(comment, bytes):
                comment = comment.decode("utf-8")
            return json.loads(comment)
        
        return None
        
    except Exception:
        return None


def read_metadata_sidecar(file_path: str) -> Optional[Dict[str, Any]]:
    """Read metadata from sidecar .json file.
    
    Args:
        file_path: Path to media file (not the sidecar itself)
    
    Returns:
        Metadata dictionary or None if not found
    """
    try:
        sidecar_path = Path(file_path).with_suffix(Path(file_path).suffix + ".json")
        
        if not sidecar_path.exists():
            return None
        
        with open(sidecar_path, "r", encoding="utf-8") as f:
            return json.load(f)
        
    except Exception:
        return None


def read_metadata(file_path: str) -> Optional[Dict[str, Any]]:
    """Read metadata from file.
    
    Automatically determines the best method based on file extension.
    
    Args:
        file_path: Path to file
    
    Returns:
        Metadata dictionary or None if not found
    """
    if not os.path.exists(file_path):
        return None
    
    ext = Path(file_path).suffix.lower()
    
    if ext == ".png":
        return read_metadata_png(file_path)
    elif ext in {".jpg", ".jpeg"}:
        return read_metadata_jpeg(file_path)
    else:
        # Try sidecar
        return read_metadata_sidecar(file_path)
