"""TUIProgressBridge — adapts progress callbacks to Textual messages."""

import threading
from dataclasses import dataclass
from typing import Callable, Optional

from textual.message import Message


@dataclass
class ProgressUpdate(Message):
    """Message posted to a screen when progress changes."""
    stage: str = ""
    current: int = 0
    total: int = 0
    message: str = ""


@dataclass
class StageStarted(Message):
    """Message posted when a new stage begins."""
    name: str = ""
    total: int = 0


@dataclass
class TaskComplete(Message):
    """Message posted when the task finishes."""
    success: bool = True
    output_path: Optional[str] = None
    error: Optional[str] = None
    elapsed_s: float = 0.0


class TUIProgressBridge:
    """Converts progress callbacks into Textual messages.

    Used by TaskRunner in a worker thread to communicate progress
    back to the UI thread safely.
    """

    def __init__(self, post_message: Callable):
        self._post_message = post_message
        self._cancelled = threading.Event()
        self._current_stage = ""

    def start_stage(self, name: str, total: int = 0):
        """Signal start of a new processing stage."""
        self._current_stage = name
        try:
            self._post_message(StageStarted(name=name, total=total))
        except Exception:
            pass  # UI might be shutting down

    def update(self, current: int, total: int, message: str = ""):
        """Update progress for current stage."""
        if self._cancelled.is_set():
            raise KeyboardInterrupt("Cancelled by user")
        try:
            self._post_message(ProgressUpdate(
                stage=self._current_stage,
                current=current,
                total=total,
                message=message,
            ))
        except Exception:
            pass

    def diffusers_callback(self, pipe, step, timestep, kwargs):
        """Drop-in for diffusers callback_on_step_end."""
        if self._cancelled.is_set():
            raise KeyboardInterrupt("Cancelled by user")
        num_steps = getattr(pipe, "num_inference_steps", 0)
        self.update(step + 1, num_steps)
        return kwargs

    def cancel(self):
        """Request cancellation from UI thread."""
        self._cancelled.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()
