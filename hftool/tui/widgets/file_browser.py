"""FilePickerScreen — modal file browser for selecting files."""

import os
from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Input, Label, Static


def get_browse_root() -> str:
    """Return the best root directory for browsing.

    Inside Docker: /home/host (user's home mounted)
    Native: user's home directory
    """
    if os.environ.get("HFTOOL_IN_DOCKER"):
        host_home = "/home/host"
        if os.path.isdir(host_home):
            return host_home
        # Fall back to workspace
        return "/workspace"
    return os.path.expanduser("~")


class FilePickerScreen(ModalScreen[str]):
    """Modal screen for browsing and selecting a file.

    Returns the selected file path via dismiss(), or empty string if cancelled.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    DEFAULT_CSS = """
    FilePickerScreen {
        align: center middle;
    }
    #file-picker-dialog {
        width: 80;
        height: 30;
        border: round $accent;
        border-title-color: $text;
        border-title-style: bold;
        background: $surface;
        padding: 1 2;
    }
    #file-picker-dialog:focus-within {
        border: round $accent 100%;
    }
    #fp-path-input {
        margin: 0 0 1 0;
    }
    #fp-tree {
        height: 1fr;
        border: round $surface-lighten-2;
    }
    #fp-tree:focus {
        border: round $accent;
    }
    #fp-buttons {
        height: auto;
        margin: 1 0 0 0;
        align: right middle;
    }
    #fp-buttons > Button {
        margin: 0 0 0 1;
    }
    #fp-current {
        color: $text-muted;
        height: 1;
        margin: 0 0 1 0;
    }
    """

    def __init__(
        self,
        title: str = "Select File",
        root: Optional[str] = None,
        extensions: Optional[list[str]] = None,
    ):
        super().__init__()
        self._title = title
        self._root = root or get_browse_root()
        self._extensions = extensions

    def compose(self) -> ComposeResult:
        with Vertical(id="file-picker-dialog"):
            yield Input(
                placeholder="Type path or browse below...",
                id="fp-path-input",
            )
            yield Static(f"Browsing: {self._root}", id="fp-current")
            yield DirectoryTree(self._root, id="fp-tree")
            with Horizontal(id="fp-buttons"):
                yield Button("Cancel", variant="default", id="fp-cancel")
                yield Button("Select", variant="success", id="fp-select")

    def on_mount(self) -> None:
        dialog = self.query_one("#file-picker-dialog")
        dialog.border_title = self._title

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        path = str(event.path)
        if self._extensions:
            suffix = Path(path).suffix.lower()
            if suffix not in self._extensions:
                return
        self.query_one("#fp-path-input", Input).value = path

    def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        self.query_one("#fp-current", Static).update(f"Browsing: {event.path}")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "fp-path-input" and event.value:
            self.dismiss(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "fp-select":
            path = self.query_one("#fp-path-input", Input).value.strip()
            if path:
                self.dismiss(path)
        elif event.button.id == "fp-cancel":
            self.dismiss("")

    def action_cancel(self) -> None:
        self.dismiss("")
