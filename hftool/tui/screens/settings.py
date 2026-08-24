"""SettingsScreen — device, GPU, paths configuration."""

import os

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, Label, Select, Static


class SettingsScreen(Screen):
    """View and edit hftool settings."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
    ]

    DEFAULT_CSS = """
    SettingsScreen {
        layout: vertical;
    }
    #settings-content {
        padding: 1 2;
        height: 1fr;
    }
    #settings-section {
        border: round $accent 40%;
        border-title-color: $text-muted;
        border-title-style: bold;
        padding: 1 2;
        height: auto;
    }
    #settings-section:focus-within {
        border: round $accent 100%;
        border-title-color: $text;
    }
    .settings-note {
        color: $text-muted;
        margin: 1 0;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="settings-content"):
            with Vertical(id="settings-section"):
                yield Label("Device:", classes="field-label")
                yield Select(
                    [("auto", "auto"), ("cuda", "cuda"), ("mps", "mps"), ("cpu", "cpu")],
                    value="auto",
                    id="device-select",
                )
                yield Label("GPU acceleration backend", classes="field-help")

                yield Label("Default dtype:", classes="field-label")
                yield Select(
                    [("auto", "auto"), ("bfloat16", "bfloat16"), ("float16", "float16"), ("float32", "float32")],
                    value="auto",
                    id="dtype-select",
                )
                yield Label("Data type for model weights", classes="field-help")

                yield Label("Models directory:", classes="field-label")
                yield Input(id="models-dir")
                yield Label("Where downloaded models are stored", classes="field-help")

                yield Label("Auto-download:", classes="field-label")
                yield Select(
                    [("ask", "ask"), ("yes", "yes"), ("no", "no")],
                    value="ask",
                    id="auto-download-select",
                )
                yield Label("Whether to auto-download models", classes="field-help")

            yield Static(
                "  Settings are read from config and environment variables.\n"
                "  Edit [bold]~/.hftool/config.toml[/bold] or set [bold]HFTOOL_*[/bold] env vars.",
                classes="settings-note",
            )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#settings-section").border_title = "Settings"
        self._load_settings()

    def _load_settings(self) -> None:
        try:
            from hftool.core.download import get_models_dir
            models_dir = str(get_models_dir())
        except Exception:
            models_dir = "~/.hftool/models"

        self.query_one("#models-dir", Input).value = models_dir

        auto_dl = os.environ.get("HFTOOL_AUTO_DOWNLOAD", "").lower()
        if auto_dl in ("1", "true", "yes"):
            self.query_one("#auto-download-select", Select).value = "yes"

    def action_back(self) -> None:
        self.app.pop_screen()
