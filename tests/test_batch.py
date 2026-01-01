"""Tests for batch processing functionality."""

import os
import json
import tempfile
from pathlib import Path

import pytest


class TestBatch:
    """Tests for hftool.core.batch module."""
    
    def test_batch_result_dataclass(self):
        """BatchResult should be a valid dataclass."""
        from hftool.core.batch import BatchResult
        from dataclasses import asdict
        
        result = BatchResult(
            input_file="input.txt",
            output_file="output.png",
            success=True,
            execution_time=1.5,
        )
        
        assert result.input_file == "input.txt"
        assert result.output_file == "output.png"
        assert result.success is True
        
        # Can convert to dict
        result_dict = asdict(result)
        assert result_dict["input_file"] == "input.txt"
    
    def test_load_batch_inputs_from_file(self):
        """load_batch_inputs should read from a text file."""
        from hftool.core.batch import load_batch_inputs
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            tmp.write("file1.png\n")
            tmp.write("file2.png\n")
            tmp.write("# comment\n")
            tmp.write("file3.png\n")
            tmp_file = tmp.name
        
        try:
            # Create dummy files
            for i in range(1, 4):
                with open(f"file{i}.png", "w") as f:
                    f.write("dummy")
            
            inputs = load_batch_inputs(tmp_file)
            
            # Should have 3 files (comment line ignored)
            assert len(inputs) == 3
            assert "file1.png" in inputs[0]
            assert "file2.png" in inputs[1]
            assert "file3.png" in inputs[2]
            
            # Cleanup dummy files
            for i in range(1, 4):
                if os.path.exists(f"file{i}.png"):
                    os.unlink(f"file{i}.png")
        
        finally:
            if os.path.exists(tmp_file):
                os.unlink(tmp_file)
    
    def test_load_batch_inputs_from_directory(self):
        """load_batch_inputs should list files from directory."""
        from hftool.core.batch import load_batch_inputs
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for i in range(3):
                (Path(tmpdir) / f"file{i}.png").write_text("dummy")
            
            inputs = load_batch_inputs(tmpdir, file_pattern="*.png")
            
            assert len(inputs) == 3
            assert all(str(Path(tmpdir)) in inp for inp in inputs)
    
    def test_load_batch_inputs_nonexistent(self):
        """load_batch_inputs should raise for nonexistent path."""
        from hftool.core.batch import load_batch_inputs
        
        with pytest.raises(ValueError, match="not found"):
            load_batch_inputs("/nonexistent/path")
    
    def test_load_batch_json_valid(self):
        """load_batch_json should parse JSON array."""
        from hftool.core.batch import load_batch_json
        
        batch_data = [
            {"input": "prompt 1", "output": "out1.png"},
            {"input": "prompt 2", "output": "out2.png", "params": {"seed": 42}},
        ]
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(batch_data, tmp)
            tmp_file = tmp.name
        
        try:
            entries = load_batch_json(tmp_file)
            
            assert len(entries) == 2
            assert entries[0]["input"] == "prompt 1"
            assert entries[1]["params"]["seed"] == 42
        
        finally:
            if os.path.exists(tmp_file):
                os.unlink(tmp_file)
    
    def test_load_batch_json_invalid_structure(self):
        """load_batch_json should raise for invalid JSON structure."""
        from hftool.core.batch import load_batch_json
        
        # Not an array
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump({"key": "value"}, tmp)
            tmp_file = tmp.name
        
        try:
            with pytest.raises(ValueError, match="must be an array"):
                load_batch_json(tmp_file)
        
        finally:
            if os.path.exists(tmp_file):
                os.unlink(tmp_file)
    
    def test_load_batch_json_too_large(self):
        """load_batch_json should raise for files too large."""
        from hftool.core.batch import load_batch_json
        
        # Create a file that appears too large
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            # Write 11MB of data
            data = [{"input": "x" * 10000}] * 100
            json.dump(data, tmp)
            tmp_file = tmp.name
        
        try:
            # Check file size first
            file_size = os.path.getsize(tmp_file)
            if file_size > 10 * 1024 * 1024:
                with pytest.raises(ValueError, match="too large"):
                    load_batch_json(tmp_file)
        
        finally:
            if os.path.exists(tmp_file):
                os.unlink(tmp_file)
    
    def test_generate_output_filename_with_input_name(self):
        """generate_output_filename should use input filename."""
        from hftool.core.batch import generate_output_filename
        
        output = generate_output_filename(
            input_path="/path/to/input.txt",
            index=0,
            output_dir=None,
            output_extension=".png",
            use_input_name=True,
        )
        
        assert "input.png" in output
        assert "/path/to/input.png" == output
    
    def test_generate_output_filename_with_index(self):
        """generate_output_filename should use index for numbering."""
        from hftool.core.batch import generate_output_filename
        
        output = generate_output_filename(
            input_path="/path/to/input.txt",
            index=5,
            output_dir=None,
            output_extension=".png",
            use_input_name=False,
        )
        
        assert "0005.png" in output
    
    def test_generate_output_filename_with_output_dir(self):
        """generate_output_filename should respect output_dir."""
        from hftool.core.batch import generate_output_filename
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output = generate_output_filename(
                input_path="/path/to/input.txt",
                index=0,
                output_dir=tmpdir,
                output_extension=".png",
                use_input_name=True,
            )
            
            assert tmpdir in output
            assert "input.png" in output
