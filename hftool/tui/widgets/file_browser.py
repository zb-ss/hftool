"""FileBrowser widget — scoped directory tree for file selection."""

from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import DirectoryTree, Input, Label


class FileSelected(Message):
    """Posted when a file is selected."""
    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__()


class FileBrowser(Vertical):
    """File browser with path input and directory tree."""

    DEFAULT_CSS = """
    FileBrowser {
        height: auto;
        max-height: 20;
        border: solid $accent;
        padding: 0 1;
    }
    FileBrowser > Label {
        margin: 1 0 0 0;
    }
    FileBrowser > Input {
        margin: 0 0 1 0;
    }
    """

    def __init__(
        self,
        root: str = ".",
        label: str = "Select file:",
        extensions: Optional[list[str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._root = root
        self._label = label
        self._extensions = extensions or []

    def compose(self) -> ComposeResult:
        yield Label(self._label)
        yield Input(placeholder="Type path or browse below...", id="file-input")
        yield DirectoryTree(self._root, id="file-tree")

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        path = str(event.path)
        if self._extensions:
            suffix = Path(path).suffix.lower()
            if suffix not in self._extensions:
                return
        self.query_one("#file-input", Input).value = path
        self.post_message(FileSelected(path))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "file-input" and event.value:
            self.post_message(FileSelected(event.value))

    @property
    def selected_path(self) -> str:
        return self.query_one("#file-input", Input).value
