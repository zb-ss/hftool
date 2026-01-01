"""Tests for benchmark functionality."""

import os
import tempfile
from pathlib import Path

import pytest


class TestBenchmark:
    """Tests for hftool.core.benchmark module."""
    
    def test_benchmark_result_dataclass(self):
        """BenchmarkResult should be a valid dataclass."""
        from hftool.core.benchmark import BenchmarkResult
        from dataclasses import asdict
        
        result = BenchmarkResult(
            task="text-to-image",
            model="test-model",
            repo_id="test/model",
            timestamp="2025-01-01T00:00:00Z",
            device="cpu",
            dtype="float32",
            load_time=1.5,
            inference_time=2.3,
            total_time=3.8,
            vram_peak=None,
            vram_allocated=None,
            test_prompt="A test prompt",
            test_params={"width": 512},
            success=True,
        )
        
        assert result.task == "text-to-image"
        assert result.success is True
        assert result.total_time == 3.8
        
        # Can convert to dict
        result_dict = asdict(result)
        assert result_dict["task"] == "text-to-image"
    
    def test_get_benchmarks_file(self):
        """get_benchmarks_file should return a Path."""
        from hftool.core.benchmark import get_benchmarks_file
        
        benchmarks_file = get_benchmarks_file()
        assert isinstance(benchmarks_file, Path)
        assert str(benchmarks_file).endswith("benchmarks.json")
    
    def test_save_and_load_benchmarks(self):
        """save_benchmark and load_benchmarks should work."""
        from hftool.core.benchmark import BenchmarkResult, save_benchmark, load_benchmarks, get_benchmarks_file
        
        # Create a temporary benchmarks file
        with tempfile.TemporaryDirectory() as tmpdir:
            # Temporarily override the benchmarks file location
            import hftool.core.benchmark as benchmark_module
            original_func = benchmark_module.get_benchmarks_file
            
            def temp_get_benchmarks_file():
                return Path(tmpdir) / "benchmarks.json"
            
            benchmark_module.get_benchmarks_file = temp_get_benchmarks_file
            
            try:
                result = BenchmarkResult(
                    task="text-to-speech",
                    model="bark-small",
                    repo_id="suno/bark-small",
                    timestamp="2025-01-01T00:00:00Z",
                    device="cpu",
                    dtype="float32",
                    load_time=5.0,
                    inference_time=3.0,
                    total_time=8.0,
                    vram_peak=None,
                    vram_allocated=None,
                    test_prompt="Test prompt",
                    test_params={},
                    success=True,
                )
                
                # Save
                save_benchmark(result)
                
                # Load
                loaded = load_benchmarks()
                
                assert len(loaded) == 1
                assert loaded[0].task == "text-to-speech"
                assert loaded[0].model == "bark-small"
                assert loaded[0].total_time == 8.0
            
            finally:
                # Restore original function
                benchmark_module.get_benchmarks_file = original_func
    
    def test_measure_vram_returns_tuple_or_none(self):
        """measure_vram should return tuple or None."""
        from hftool.core.benchmark import measure_vram
        
        result = measure_vram()
        
        # Result is either None or a tuple of two floats
        assert result is None or (isinstance(result, tuple) and len(result) == 2)
    
    def test_get_benchmark_history_filtering(self):
        """get_benchmark_history should filter results."""
        from hftool.core.benchmark import BenchmarkResult, save_benchmark, get_benchmark_history, get_benchmarks_file
        
        with tempfile.TemporaryDirectory() as tmpdir:
            import hftool.core.benchmark as benchmark_module
            original_func = benchmark_module.get_benchmarks_file
            
            def temp_get_benchmarks_file():
                return Path(tmpdir) / "benchmarks.json"
            
            benchmark_module.get_benchmarks_file = temp_get_benchmarks_file
            
            try:
                # Save multiple benchmarks
                for i, task in enumerate(["text-to-image", "text-to-speech"]):
                    result = BenchmarkResult(
                        task=task,
                        model=f"model-{i}",
                        repo_id=f"test/model-{i}",
                        timestamp=f"2025-01-01T00:00:0{i}Z",
                        device="cpu",
                        dtype="float32",
                        load_time=1.0,
                        inference_time=1.0,
                        total_time=2.0,
                        vram_peak=None,
                        vram_allocated=None,
                        test_prompt="Test",
                        test_params={},
                        success=True,
                    )
                    save_benchmark(result)
                
                # Filter by task
                t2i_results = get_benchmark_history(task="text-to-image")
                assert len(t2i_results) == 1
                assert t2i_results[0].task == "text-to-image"
                
                # All results
                all_results = get_benchmark_history()
                assert len(all_results) == 2
            
            finally:
                benchmark_module.get_benchmarks_file = original_func
