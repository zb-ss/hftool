"""GenerationScreen — progress, log, and result display."""

import os

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Label, ProgressBar, RichLog, Static

from hftool.core.executor import TaskRequest, TaskResult
from hftool.tui.bridge import ProgressUpdate, StageStarted
from hftool.tui.runner import TaskRunner


class GenerationScreen(Screen):
    """Shows progress while a task runs, then displays the result."""

    BINDINGS = [
        Binding("escape", "cancel_or_back", "Cancel/Back"),
    ]

    DEFAULT_CSS = """
    GenerationScreen {
        layout: vertical;
    }
    #gen-content {
        padding: 1 2;
        height: 1fr;
    }
    #stage-label {
        margin: 1 0;
        text-style: bold;
    }
    #result-panel {
        border: round $success 60%;
        border-title-color: $text-muted;
        border-title-style: bold;
        padding: 1 2;
        margin: 1 0;
        height: auto;
    }
    #cancel-btn {
        margin: 1 0;
    }
    """

    def __init__(self, request: TaskRequest):
        super().__init__()
        self._request = request
        self._runner: TaskRunner | None = None
        self._running = False
        self._result: TaskResult | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="gen-content"):
            yield Label(
                f"[bold]{self._request.task_name}[/bold] — {self._request.model or 'default'}",
                id="task-info",
            )
            yield Label("Preparing...", id="stage-label")
            yield ProgressBar(total=100, show_eta=True, id="progress")
            yield RichLog(highlight=True, markup=True, id="log")
            yield Static(id="result-panel")
            yield Button("Cancel", variant="error", id="cancel-btn")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#result-panel").display = False
        self.query_one("#result-panel").border_title = "Result"
        self._start_task()

    def _start_task(self) -> None:
        self._runner = TaskRunner(self._request, post_message=self.call_from_thread)
        self._running = True
        self._log("Starting task...")
        self._run_worker()

    @work(thread=True)
    def _run_worker(self) -> None:
        result = self._runner.execute()
        self.call_from_thread(self._on_task_done, result)

    def _on_task_done(self, result: TaskResult) -> None:
        self._running = False
        self._result = result

        stage_label = self.query_one("#stage-label", Label)
        progress_bar = self.query_one("#progress", ProgressBar)
        cancel_btn = self.query_one("#cancel-btn", Button)
        result_panel = self.query_one("#result-panel", Static)

        cancel_btn.label = "Back"
        cancel_btn.variant = "primary"

        if result.success:
            stage_label.update("[green]Complete![/green]")
            progress_bar.update(total=100, progress=100)
            self._log(f"[green]Task completed in {result.elapsed_s:.1f}s[/green]")

            lines = []
            if result.output_path and os.path.exists(result.output_path):
                size = os.path.getsize(result.output_path)
                size_str = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / (1024*1024):.1f} MB"
                lines.append(f"[bold]Output:[/bold] {result.output_path} ({size_str})")
            if result.seed is not None:
                lines.append(f"[bold]Seed:[/bold] {result.seed}")
            if result.elapsed_s:
                lines.append(f"[bold]Time:[/bold] {result.elapsed_s:.1f}s")

            if isinstance(result.result_data, str):
                lines.append(f"\n[bold]Result:[/bold]\n{result.result_data[:500]}")
            elif isinstance(result.result_data, dict) and "text" in result.result_data:
                lines.append(f"\n[bold]Result:[/bold]\n{result.result_data['text'][:500]}")

            result_panel.update("\n".join(lines) if lines else "Done")
            result_panel.display = True
        else:
            stage_label.update("[red]Failed[/red]")
            self._log(f"[red]Error: {result.error}[/red]")
            result_panel.update(f"[red]{result.error}[/red]")
            result_panel.styles.border = ("round", "red 60%")
            result_panel.border_title = "Error"
            result_panel.display = True

    def on_progress_update(self, message: ProgressUpdate) -> None:
        progress_bar = self.query_one("#progress", ProgressBar)
        if message.total > 0:
            progress_bar.update(total=message.total, progress=message.current)
        if message.message:
            self._log(message.message)

    def on_stage_started(self, message: StageStarted) -> None:
        self.query_one("#stage-label", Label).update(f"[bold]{message.name}[/bold]")
        if message.total > 0:
            self.query_one("#progress", ProgressBar).update(total=message.total, progress=0)
        self._log(f"Stage: {message.name}")

    def _log(self, text: str) -> None:
        try:
            self.query_one("#log", RichLog).write(text)
        except Exception:
            pass

    def action_cancel_or_back(self) -> None:
        if self._running and self._runner:
            self._runner.cancel()
            self._log("[yellow]Cancelling...[/yellow]")
        else:
            self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-btn":
            self.action_cancel_or_back()
