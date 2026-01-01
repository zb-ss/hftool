"""Progress tracking utilities for hftool.

Provides rich progress bars with fallback to simple text output.
"""

import time
import click
from typing import Optional, Callable, Any, List

# Try to import rich for fancy progress bars
try:
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeRemainingColumn,
        TimeElapsedColumn,
    )
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ProgressTracker:
    """Track progress across multiple stages with optional step-by-step updates.
    
    Supports both rich progress bars (if available) and simple text fallback.
    """
    
    def __init__(
        self,
        stages: List[str],
        verbose: bool = True,
        use_rich: bool = True,
    ):
        """Initialize progress tracker.
        
        Args:
            stages: List of stage names to track
            verbose: Whether to show progress output
            use_rich: Whether to use rich progress bars (if available)
        """
        self.stages = stages
        self.verbose = verbose
        self.current_stage_idx = -1
        self.current_stage_name = ""
        
        # Rich progress components
        self._use_rich = use_rich and RICH_AVAILABLE
        self._progress: Optional[Any] = None
        self._task_id: Optional[Any] = None
        self._console: Optional[Any] = None
        
        # Simple progress state
        self._start_time: Optional[float] = None
        self._last_update_time: Optional[float] = None
        
        # Inference step tracking
        self._current_step = 0
        self._total_steps = 0
    
    def start(self) -> None:
        """Start the progress tracker."""
        if not self.verbose:
            return
        
        self._start_time = time.time()
        
        if self._use_rich:
            self._console = Console()
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self._console,
            )
            self._progress.__enter__()
    
    def start_stage(self, stage_name: str, total: Optional[int] = None) -> None:
        """Start a new stage.
        
        Args:
            stage_name: Name of the stage
            total: Total number of steps in this stage (for progress bar)
        """
        if not self.verbose:
            return
        
        self.current_stage_idx += 1
        self.current_stage_name = stage_name
        self._current_step = 0
        self._total_steps = total or 0
        
        if self._use_rich and self._progress:
            # Update or create task
            description = f"[{self.current_stage_idx + 1}/{len(self.stages)}] {stage_name}"
            
            if self._task_id is not None:
                self._progress.update(
                    self._task_id,
                    description=description,
                    total=total,
                    completed=0,
                )
            else:
                self._task_id = self._progress.add_task(
                    description,
                    total=total,
                )
        else:
            # Simple text output
            prefix = f"[{self.current_stage_idx + 1}/{len(self.stages)}]"
            click.echo(f"{prefix} {stage_name}...")
            self._last_update_time = time.time()
    
    def update(self, current: int, total: Optional[int] = None, message: str = "") -> None:
        """Update progress within current stage.
        
        Args:
            current: Current step number
            total: Total steps (optional, uses value from start_stage if None)
            message: Optional message to display
        """
        if not self.verbose:
            return
        
        self._current_step = current
        if total is not None:
            self._total_steps = total
        
        if self._use_rich and self._progress and self._task_id is not None:
            # Update rich progress bar
            self._progress.update(
                self._task_id,
                completed=current,
                total=self._total_steps,
            )
        else:
            # Simple text output - only update every second to avoid spam
            current_time = time.time()
            if self._last_update_time is None or current_time - self._last_update_time >= 1.0:
                if self._total_steps > 0:
                    percent = (current / self._total_steps) * 100
                    eta = self._calculate_eta(current, self._total_steps)
                    click.echo(f"  Step {current}/{self._total_steps} ({percent:.0f}%) | ETA: {eta}")
                else:
                    click.echo(f"  Step {current}...")
                self._last_update_time = current_time
    
    def _calculate_eta(self, current: int, total: int) -> str:
        """Calculate estimated time remaining.
        
        Args:
            current: Current step
            total: Total steps
        
        Returns:
            Formatted ETA string
        """
        if current == 0 or self._start_time is None:
            return "calculating..."
        
        elapsed = time.time() - self._start_time
        rate = current / elapsed
        remaining = (total - current) / rate
        
        if remaining < 60:
            return f"{remaining:.0f}s"
        elif remaining < 3600:
            return f"{remaining/60:.1f}m"
        else:
            return f"{remaining/3600:.1f}h"
    
    def complete_stage(self, success: bool = True) -> None:
        """Mark current stage as complete.
        
        Args:
            success: Whether the stage completed successfully
        """
        if not self.verbose:
            return
        
        if self._use_rich and self._progress and self._task_id is not None:
            # Mark task as complete
            self._progress.update(
                self._task_id,
                completed=self._total_steps,
            )
        else:
            # Simple text output
            if success:
                click.echo(f"  ✓ {self.current_stage_name} complete")
            else:
                click.echo(f"  ✗ {self.current_stage_name} failed", err=True)
    
    def finish(self) -> None:
        """Finish progress tracking and cleanup."""
        if not self.verbose:
            return
        
        if self._use_rich and self._progress:
            self._progress.__exit__(None, None, None)
            self._progress = None
            self._task_id = None
    
    def diffusers_callback(self, pipe: Any, step: int, timestep: Any, callback_kwargs: dict) -> dict:
        """Callback for diffusers pipelines.
        
        This is called during inference to update progress.
        
        Args:
            pipe: Pipeline object
            step: Current step number
            timestep: Current timestep (unused)
            callback_kwargs: Callback arguments from diffusers
        
        Returns:
            callback_kwargs (unmodified)
        """
        # Get total steps from pipeline
        total_steps = getattr(pipe, "num_inference_steps", 0)
        
        # Update progress
        self.update(step + 1, total_steps)
        
        return callback_kwargs
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()
        return False


def create_simple_progress(description: str, total: Optional[int] = None, verbose: bool = True) -> Optional[Any]:
    """Create a simple progress context for one-off operations.
    
    Args:
        description: Description of the operation
        total: Total steps (if known)
        verbose: Whether to show progress
    
    Returns:
        Progress context manager or None if verbose=False
    """
    if not verbose:
        return None
    
    if RICH_AVAILABLE:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        )
        progress.__enter__()
        progress.add_task(description, total=total)
        return progress
    else:
        click.echo(f"{description}...")
        return None
