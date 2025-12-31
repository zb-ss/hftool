"""Tests for metadata embedding functionality."""

import os
import json
import tempfile
from pathlib import Path

import pytest


class TestMetadata:
    """Tests for hftool.core.metadata module."""
    
    def test_create_metadata(self):
        """create_metadata should return a valid metadata dictionary."""
        from hftool.core.metadata import create_metadata
        
        metadata = create_metadata(
            task="text-to-image",
            model="test-model",
            prompt="A test prompt",
            seed=42,
            extra_params={"num_inference_steps": 20, "guidance_scale": 7.5},
        )
        
        assert metadata["task"] == "text-to-image"
        assert metadata["model"] == "test-model"
        assert metadata["prompt"] == "A test prompt"
        assert metadata["seed"] == 42
        assert metadata["num_inference_steps"] == 20
        assert metadata["guidance_scale"] == 7.5
        assert "hftool_version" in metadata
        assert "timestamp" in metadata
    
    def test_create_metadata_minimal(self):
        """create_metadata should work with minimal parameters."""
        from hftool.core.metadata import create_metadata
        
        metadata = create_metadata(
            task="text-to-speech",
            model="bark-small",
        )
        
        assert metadata["task"] == "text-to-speech"
        assert metadata["model"] == "bark-small"
        assert "hftool_version" in metadata
        assert "timestamp" in metadata
    
    def test_embed_metadata_sidecar(self):
        """embed_metadata_sidecar should create a JSON file."""
        from hftool.core.metadata import embed_metadata_sidecar, create_metadata
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_file = tmp.name
        
        try:
            # Write some dummy content
            with open(tmp_file, "w") as f:
                f.write("dummy audio data")
            
            metadata = create_metadata(
                task="text-to-speech",
                model="bark-small",
                prompt="Hello world",
                seed=42,
            )
            
            result = embed_metadata_sidecar(tmp_file, metadata, verbose=False)
            
            assert result is True
            
            # Check sidecar file exists
            sidecar_file = tmp_file + ".json"
            assert os.path.exists(sidecar_file)
            
            # Read and verify
            with open(sidecar_file, "r") as f:
                loaded = json.load(f)
            
            assert loaded["task"] == "text-to-speech"
            assert loaded["model"] == "bark-small"
            assert loaded["prompt"] == "Hello world"
            assert loaded["seed"] == 42
            
            # Cleanup
            os.unlink(sidecar_file)
        
        finally:
            if os.path.exists(tmp_file):
                os.unlink(tmp_file)
    
    def test_read_metadata_sidecar(self):
        """read_metadata_sidecar should read metadata from JSON file."""
        from hftool.core.metadata import read_metadata_sidecar, create_metadata, embed_metadata_sidecar
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_file = tmp.name
        
        try:
            # Write dummy file
            with open(tmp_file, "w") as f:
                f.write("dummy video data")
            
            # Embed metadata
            metadata = create_metadata(
                task="text-to-video",
                model="hunyuanvideo",
                prompt="A flying bird",
                seed=123,
            )
            embed_metadata_sidecar(tmp_file, metadata, verbose=False)
            
            # Read metadata
            loaded = read_metadata_sidecar(tmp_file)
            
            assert loaded is not None
            assert loaded["task"] == "text-to-video"
            assert loaded["model"] == "hunyuanvideo"
            assert loaded["prompt"] == "A flying bird"
            assert loaded["seed"] == 123
            
            # Cleanup
            sidecar_file = tmp_file + ".json"
            if os.path.exists(sidecar_file):
                os.unlink(sidecar_file)
        
        finally:
            if os.path.exists(tmp_file):
                os.unlink(tmp_file)
    
    def test_read_metadata_nonexistent(self):
        """read_metadata should return None for nonexistent files."""
        from hftool.core.metadata import read_metadata
        
        result = read_metadata("/nonexistent/file.png")
        assert result is None
    
    def test_embed_metadata_validates_path(self):
        """embed_metadata should validate file path."""
        from hftool.core.metadata import embed_metadata
        
        # Non-existent file
        result = embed_metadata(
            file_path="/nonexistent/file.png",
            task="text-to-image",
            model="test",
            verbose=False,
        )
        
        assert result is False
