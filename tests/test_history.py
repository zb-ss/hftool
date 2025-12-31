"""Tests for command history functionality."""

import json
import tempfile
import time
from pathlib import Path
import pytest

from hftool.core.history import History, HistoryEntry


class TestHistory:
    """Tests for History class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Reset singleton before each test
        History.reset()
        
        # Use temporary directory for test history
        self.temp_dir = tempfile.mkdtemp()
        self.history_file = Path(self.temp_dir) / ".hftool" / "history.json"
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Patch the history file location
        self._original_init = History.__init__
        
        def patched_init(self_inner):
            self._original_init(self_inner)
            self_inner._history_file = self.history_file
        
        History.__init__ = patched_init
    
    def teardown_method(self):
        """Clean up after test."""
        # Restore original __init__
        History.__init__ = self._original_init
        
        # Clean up temp directory
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_singleton_pattern(self):
        """Test that History follows singleton pattern."""
        history1 = History.get()
        history2 = History.get()
        assert history1 is history2
    
    def test_add_entry(self):
        """Test adding a history entry."""
        history = History.get()
        
        entry_id = history.add(
            task="text-to-image",
            model="z-image",
            input_data="A cat in space",
            output_file="cat.png",
            device="cuda",
            dtype="float16",
            seed=42,
            extra_args={"num_inference_steps": 20},
            success=True,
        )
        
        assert entry_id == 1
        assert len(history.get_all()) == 1
    
    def test_get_by_id(self):
        """Test retrieving entry by ID."""
        history = History.get()
        
        entry_id = history.add(
            task="text-to-image",
            model="z-image",
            input_data="A cat",
            output_file="cat.png",
            device="cuda",
            dtype=None,
            seed=42,
            extra_args={},
            success=True,
        )
        
        entry = history.get_by_id(entry_id)
        assert entry is not None
        assert entry.task == "text-to-image"
        assert entry.seed == 42
    
    def test_get_by_id_not_found(self):
        """Test getting non-existent entry."""
        history = History.get()
        entry = history.get_by_id(999)
        assert entry is None
    
    def test_get_recent(self):
        """Test getting recent entries."""
        history = History.get()
        
        # Add multiple entries
        for i in range(5):
            history.add(
                task="text-to-image",
                model="z-image",
                input_data=f"Test {i}",
                output_file=f"test{i}.png",
                device="cuda",
                dtype=None,
                seed=i,
                extra_args={},
                success=True,
            )
        
        recent = history.get_recent(limit=3)
        assert len(recent) == 3
        # Should be in reverse order (newest first)
        assert recent[0].seed == 4
        assert recent[1].seed == 3
        assert recent[2].seed == 2
    
    def test_clear_history(self):
        """Test clearing all history."""
        history = History.get()
        
        # Add some entries
        for i in range(3):
            history.add(
                task="text-to-image",
                model="z-image",
                input_data=f"Test {i}",
                output_file=f"test{i}.png",
                device="cuda",
                dtype=None,
                seed=i,
                extra_args={},
                success=True,
            )
        
        assert len(history.get_all()) == 3
        
        history.clear()
        assert len(history.get_all()) == 0
    
    def test_persistence(self):
        """Test that history persists to disk."""
        history = History.get()
        
        history.add(
            task="text-to-image",
            model="z-image",
            input_data="Test",
            output_file="test.png",
            device="cuda",
            dtype=None,
            seed=42,
            extra_args={},
            success=True,
        )
        
        # Check file was created
        assert self.history_file.exists()
        
        # Load and verify content
        with open(self.history_file, "r") as f:
            data = json.load(f)
        
        assert "entries" in data
        assert len(data["entries"]) == 1
        assert data["entries"][0]["task"] == "text-to-image"
    
    def test_history_entry_to_command(self):
        """Test command reconstruction from entry."""
        entry = HistoryEntry(
            id=1,
            timestamp=time.time(),
            task="text-to-image",
            model="z-image",
            input_data="A cat in space",
            output_file="cat.png",
            device="cuda",
            dtype="float16",
            seed=42,
            extra_args={"num_inference_steps": 20},
            success=True,
        )
        
        command = entry.to_command()
        assert "hftool" in command
        assert "-t text-to-image" in command
        assert "-m z-image" in command
        assert "--seed 42" in command
        assert "--dtype float16" in command
        assert "-- --num-inference-steps 20" in command
    
    def test_failed_entry_recorded(self):
        """Test that failed commands are recorded."""
        history = History.get()
        
        entry_id = history.add(
            task="text-to-image",
            model="z-image",
            input_data="Test",
            output_file="test.png",
            device="cuda",
            dtype=None,
            seed=42,
            extra_args={},
            success=False,
            error_message="CUDA out of memory",
        )
        
        entry = history.get_by_id(entry_id)
        assert entry.success is False
        assert entry.error_message == "CUDA out of memory"
    
    def test_truncation_on_max_entries(self):
        """Test that history is truncated when exceeding MAX_ENTRIES."""
        history = History.get()
        
        # Temporarily set a low limit
        original_max = History.MAX_ENTRIES
        History.MAX_ENTRIES = 10
        
        try:
            # Add more than max entries
            for i in range(15):
                history.add(
                    task="text-to-image",
                    model="z-image",
                    input_data=f"Test {i}",
                    output_file=f"test{i}.png",
                    device="cuda",
                    dtype=None,
                    seed=i,
                    extra_args={},
                    success=True,
                )
            
            # Should be truncated to max
            all_entries = history.get_all()
            assert len(all_entries) <= History.MAX_ENTRIES
            
            # Should keep the most recent ones
            assert all_entries[-1].seed == 14
        
        finally:
            History.MAX_ENTRIES = original_max
    
    def test_get_recent_files(self):
        """Test getting recent files from history."""
        history = History.get()
        
        # Add entries with various file types
        history.add(
            task="text-to-image",
            model="z-image",
            input_data="Test",
            output_file="test1.png",
            device="cuda",
            dtype=None,
            seed=1,
            extra_args={},
            success=True,
        )
        
        history.add(
            task="text-to-speech",
            model="bark",
            input_data="Test",
            output_file="test2.wav",
            device="cuda",
            dtype=None,
            seed=2,
            extra_args={},
            success=True,
        )
        
        # Get all files
        all_files = history.get_recent_files(file_type=None, limit=10)
        assert len(all_files) == 2
        
        # Get only images
        image_files = history.get_recent_files(file_type="image", limit=10)
        assert len(image_files) == 1
        assert image_files[0] == "test1.png"
        
        # Get only audio
        audio_files = history.get_recent_files(file_type="audio", limit=10)
        assert len(audio_files) == 1
        assert audio_files[0] == "test2.wav"
