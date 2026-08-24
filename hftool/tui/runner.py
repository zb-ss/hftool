"""TaskRunner — wraps core/executor.py for Textual worker threads."""

from typing import Callable, Optional

from hftool.core.executor import TaskRequest, TaskResult, execute_task
from hftool.tui.bridge import TUIProgressBridge


class TaskRunner:
    """Run a task in a worker thread with TUI progress bridging."""

    def __init__(self, request: TaskRequest, post_message: Optional[Callable] = None):
        self.request = request
        self.bridge: Optional[TUIProgressBridge] = None
        if post_message:
            self.bridge = TUIProgressBridge(post_message)
            self.request.progress_callback = self.bridge.update

    def execute(self) -> TaskResult:
        """Execute the task. Call from a @work(thread=True) worker."""
        return execute_task(self.request)

    def cancel(self):
        """Request cancellation from the UI thread."""
        if self.bridge:
            self.bridge.cancel()
