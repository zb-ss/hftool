"""Tests for file picker functionality."""

import tempfile
from pathlib import Path
import pytest

from hftool.io.file_picker import FilePicker, FileType, resolve_file_reference


class TestFilePicker:
    """Tests for FilePicker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory with test files
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test files
        (self.temp_path / "test1.png").touch()
        (self.temp_path / "test2.jpg").touch()
        (self.temp_path / "test3.wav").touch()
        (self.temp_path / "test4.mp4").touch()
        (self.temp_path / "test5.txt").touch()
        
        # Create subdirectory
        subdir = self.temp_path / "subdir"
        subdir.mkdir()
        (subdir / "sub1.png").touch()
        (subdir / "sub2.wav").touch()
    
    def teardown_method(self):
        """Clean up after test."""
        import shutil
        if self.temp_path.exists():
            shutil.rmtree(self.temp_path)
    
    def test_file_type_detection(self):
        """Test file type matching."""
        picker = FilePicker(FileType.IMAGE)
        
        assert picker._matches_file_type(Path("test.png"))
        assert picker._matches_file_type(Path("test.jpg"))
        assert picker._matches_file_type(Path("test.jpeg"))
        assert not picker._matches_file_type(Path("test.wav"))
        assert not picker._matches_file_type(Path("test.mp4"))
    
    def test_audio_file_type(self):
        """Test audio file type filtering."""
        picker = FilePicker(FileType.AUDIO)
        
        assert picker._matches_file_type(Path("test.wav"))
        assert picker._matches_file_type(Path("test.mp3"))
        assert picker._matches_file_type(Path("test.flac"))
        assert not picker._matches_file_type(Path("test.png"))
    
    def test_video_file_type(self):
        """Test video file type filtering."""
        picker = FilePicker(FileType.VIDEO)
        
        assert picker._matches_file_type(Path("test.mp4"))
        assert picker._matches_file_type(Path("test.avi"))
        assert picker._matches_file_type(Path("test.mkv"))
        assert not picker._matches_file_type(Path("test.png"))
    
    def test_all_file_type(self):
        """Test ALL file type accepts everything."""
        picker = FilePicker(FileType.ALL)
        
        assert picker._matches_file_type(Path("test.png"))
        assert picker._matches_file_type(Path("test.wav"))
        assert picker._matches_file_type(Path("test.mp4"))
        assert picker._matches_file_type(Path("test.txt"))
    
    def test_find_files_non_recursive(self):
        """Test finding files non-recursively."""
        picker = FilePicker(FileType.ALL)
        files = picker._find_files(self.temp_path, recursive=False)
        
        # Should find files in root but not subdirectories
        assert len(files) == 5
        assert any("test1.png" in f for f in files)
        assert not any("sub1.png" in f for f in files)
    
    def test_find_files_recursive(self):
        """Test finding files recursively."""
        picker = FilePicker(FileType.ALL)
        files = picker._find_files(self.temp_path, recursive=True, depth=0)
        
        # Should find files in root and subdirectories
        assert len(files) == 7
        assert any("test1.png" in f for f in files)
        assert any("sub1.png" in f for f in files)
    
    def test_find_files_with_filter(self):
        """Test finding files with type filter."""
        picker = FilePicker(FileType.IMAGE)
        files = picker._find_files(self.temp_path, recursive=True, depth=0)
        
        # Should only find image files
        assert len(files) == 3  # test1.png, test2.jpg, sub1.png
        assert all(Path(f).suffix.lower() in {".png", ".jpg", ".jpeg"} for f in files)
    
    def test_max_depth_limit(self):
        """Test recursion depth limit."""
        # Create deep directory structure
        deep_path = self.temp_path / "a" / "b" / "c" / "d"
        deep_path.mkdir(parents=True)
        (deep_path / "deep.png").touch()
        
        picker = FilePicker(FileType.IMAGE)
        # MAX_DEPTH is 3, so depth 0,1,2,3 are allowed (4 levels)
        # Start at temp_path (depth 0)
        # a (depth 1), b (depth 2), c (depth 3), d (depth 4 - not scanned)
        files = picker._find_files(self.temp_path, recursive=True, depth=0)
        
        # Should not find the deep file (depth > MAX_DEPTH)
        assert not any("deep.png" in f for f in files)
    
    def test_path_validation_security(self):
        """Test path validation for security."""
        picker = FilePicker(FileType.ALL)
        
        # Allow paths within temp directory
        picker._validate_path(self.temp_path)
        
        # Should raise for paths outside allowed directories
        with pytest.raises(ValueError, match="outside allowed directories"):
            picker._validate_path(Path("/etc"))
    
    def test_file_type_from_task(self):
        """Test inferring file type from task name."""
        picker = FilePicker(FileType.ALL)
        
        assert picker._file_type_from_task("text-to-image") == FileType.IMAGE
        assert picker._file_type_from_task("t2i") == FileType.IMAGE
        assert picker._file_type_from_task("image-to-image") == FileType.IMAGE
        
        assert picker._file_type_from_task("text-to-speech") == FileType.AUDIO
        assert picker._file_type_from_task("tts") == FileType.AUDIO
        assert picker._file_type_from_task("automatic-speech-recognition") == FileType.AUDIO
        
        assert picker._file_type_from_task("text-to-video") == FileType.VIDEO
        assert picker._file_type_from_task("t2v") == FileType.VIDEO
        
        assert picker._file_type_from_task("unknown-task") == FileType.ALL
    
    def test_non_reference_passthrough(self):
        """Test that non-@ strings are passed through unchanged."""
        picker = FilePicker(FileType.ALL)
        
        # Regular file path should be returned as-is
        result = picker.resolve_reference("test.png")
        assert result == "test.png"
        
        # Text input should be returned as-is
        result = picker.resolve_reference("Some text input")
        assert result == "Some text input"
    
    def test_filter_by_type(self):
        """Test filtering file list by type."""
        picker = FilePicker(FileType.IMAGE)
        
        files = [
            "test1.png",
            "test2.jpg",
            "test3.wav",
            "test4.mp4",
        ]
        
        filtered = picker._filter_by_type(files)
        assert len(filtered) == 2
        assert "test1.png" in filtered
        assert "test2.jpg" in filtered
        assert "test3.wav" not in filtered
    
    def test_max_files_limit(self):
        """Test maximum files limit."""
        # Create many files
        for i in range(FilePicker.MAX_FILES + 10):
            (self.temp_path / f"file{i}.png").touch()
        
        picker = FilePicker(FileType.IMAGE)
        files = picker._find_files(self.temp_path, recursive=False)
        
        # Should be limited to MAX_FILES
        assert len(files) <= FilePicker.MAX_FILES


class TestResolveFileReference:
    """Tests for resolve_file_reference convenience function."""
    
    def test_passthrough_non_reference(self):
        """Test that non-references pass through."""
        result = resolve_file_reference("test.png")
        assert result == "test.png"
    
    def test_invalid_reference_raises(self):
        """Test that invalid references raise ValueError."""
        with pytest.raises(ValueError, match="Unknown file reference format"):
            resolve_file_reference("@unknown_format")
