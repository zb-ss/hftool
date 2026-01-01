"""Tests for progress tracking."""

import pytest
from hftool.utils.progress import ProgressTracker, RICH_AVAILABLE


class TestProgressTracker:
    """Test progress tracking functionality."""
    
    def test_create_tracker(self):
        """Test creating a progress tracker."""
        stages = ["Stage 1", "Stage 2", "Stage 3"]
        tracker = ProgressTracker(stages, verbose=True)
        
        assert tracker.stages == stages
        assert tracker.verbose is True
        assert tracker.current_stage_idx == -1
    
    def test_silent_mode(self):
        """Test that silent mode doesn't output."""
        stages = ["Stage 1"]
        tracker = ProgressTracker(stages, verbose=False)
        
        # These should not raise errors but also not output
        tracker.start()
        tracker.start_stage("Stage 1")
        tracker.update(50, 100)
        tracker.complete_stage()
        tracker.finish()
    
    def test_start_stage(self):
        """Test starting a stage."""
        stages = ["Stage 1", "Stage 2"]
        # Note: We need verbose=True for state to update, but we can test silently
        tracker = ProgressTracker(stages, verbose=True)
        
        tracker.start_stage("Stage 1", total=100)
        assert tracker.current_stage_idx == 0
        assert tracker.current_stage_name == "Stage 1"
        assert tracker._total_steps == 100
    
    def test_update_progress(self):
        """Test updating progress."""
        stages = ["Stage 1"]
        tracker = ProgressTracker(stages, verbose=True)
        
        tracker.start_stage("Stage 1", total=100)
        tracker.update(50, 100)
        
        assert tracker._current_step == 50
        assert tracker._total_steps == 100
    
    def test_multiple_stages(self):
        """Test progressing through multiple stages."""
        stages = ["Loading", "Processing", "Saving"]
        tracker = ProgressTracker(stages, verbose=True)
        
        tracker.start()
        
        tracker.start_stage("Loading", total=10)
        assert tracker.current_stage_idx == 0
        
        tracker.start_stage("Processing", total=50)
        assert tracker.current_stage_idx == 1
        
        tracker.start_stage("Saving", total=5)
        assert tracker.current_stage_idx == 2
        
        tracker.finish()
    
    def test_context_manager(self):
        """Test using tracker as context manager."""
        stages = ["Stage 1"]
        
        with ProgressTracker(stages, verbose=False) as tracker:
            tracker.start_stage("Stage 1")
            tracker.update(50, 100)
        
        # Should not raise
    
    def test_diffusers_callback(self):
        """Test diffusers callback integration."""
        stages = ["Inference"]
        tracker = ProgressTracker(stages, verbose=True)
        
        # Mock pipeline object
        class MockPipeline:
            num_inference_steps = 30
        
        pipe = MockPipeline()
        tracker.start_stage("Inference", total=30)
        
        # Simulate diffusers callback
        callback_kwargs = {"test": "value"}
        result = tracker.diffusers_callback(pipe, step=10, timestep=None, callback_kwargs=callback_kwargs)
        
        # Should return callback_kwargs unmodified
        assert result == callback_kwargs
        assert tracker._current_step == 11  # step + 1
    
    def test_rich_availability(self):
        """Test that RICH_AVAILABLE is a boolean."""
        assert isinstance(RICH_AVAILABLE, bool)
    
    def test_use_rich_fallback(self):
        """Test fallback to simple progress when rich disabled."""
        stages = ["Stage 1"]
        tracker = ProgressTracker(stages, verbose=True, use_rich=False)
        
        # Should use simple text output
        assert tracker._use_rich is False
