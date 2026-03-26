"""FilePickerScreen — modal file browser with search for selecting files.

Centralized file picker used by all TUI screens (Task, Voiceover, etc.).
Provides both directory tree browsing and real-time search.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button, DirectoryTree, Input, Label, ListItem, ListView, Static,
)


def get_browse_root() -> str:
    """Return the best root directory for browsing.

    Inside Docker: /home/host (user's home mounted)
    Native: user's home directory
    """
    if os.environ.get("HFTOOL_IN_DOCKER"):
        host_home = "/home/host"
        if os.path.isdir(host_home):
            return host_home
        return "/workspace"
    return os.path.expanduser("~")


# Directories to always skip during search
def _search_files(
    root: str,
    query: str,
    extensions: list[str] | None = None,
    max_results: int = 40,
    max_depth: int = 6,
) -> list[str]:
    """Search for files using `find` (fast) with fallback to Python os.walk.

    Supports:
    - Substring match: "servo" finds servonaut-1.mp4
    - Glob wildcards: "*.mp4", "servo*.mp4"
    - Extension filtering via the extensions parameter

    Uses the system `find` command which is much faster than Python os.walk
    on large directory trees.
    """
    import shutil
    import subprocess

    if shutil.which("find"):
        return _search_with_find(root, query, extensions, max_results, max_depth)
    return _search_with_walk(root, query, extensions, max_results, max_depth)


def _search_with_find(
    root: str,
    query: str,
    extensions: list[str] | None,
    max_results: int,
    max_depth: int,
) -> list[str]:
    """Fast file search using system `find` command."""
    import subprocess

    # Build find command with pruning for hidden/noisy dirs
    cmd = ["find", root, "-maxdepth", str(max_depth)]

    # Prune hidden dirs and common noise — must come before the match
    prune_names = [".*", "__pycache__", "node_modules", "site-packages"]
    prune_parts: list[str] = []
    for i, name in enumerate(prune_names):
        if i > 0:
            prune_parts.append("-o")
        prune_parts.extend(["-name", name])
    cmd.extend(["("] + prune_parts + [")", "-prune", "-o"])

    # Build name match — support glob wildcards
    has_wildcard = any(c in query for c in "*?[]")

    if has_wildcard:
        cmd.extend(["-type", "f", "-iname", query, "-print"])
    elif extensions:
        cmd.append("(")
        for i, ext in enumerate(extensions):
            if i > 0:
                cmd.append("-o")
            cmd.extend(["-type", "f", "-iname", f"*{query}*{ext}"])
        cmd.extend([")", "-print"])
    else:
        cmd.extend(["-type", "f", "-iname", f"*{query}*", "-print"])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        lines = [l for l in result.stdout.strip().split("\n") if l]
        return lines[:max_results]
    except (subprocess.TimeoutExpired, Exception):
        return []


def _search_with_walk(
    root: str,
    query: str,
    extensions: list[str] | None,
    max_results: int,
    max_depth: int,
) -> list[str]:
    """Fallback file search using Python os.walk."""
    from fnmatch import fnmatch

    has_wildcard = any(c in query for c in "*?[]")
    query_lower = query.lower()
    results: list[str] = []
    root_depth = root.rstrip("/").count("/")
    skip_dirs = {".git", "__pycache__", "node_modules", "site-packages", ".cache"}

    for dirpath, dirnames, filenames in os.walk(root):
        current_depth = dirpath.rstrip("/").count("/") - root_depth
        if current_depth >= max_depth:
            dirnames.clear()
            continue

        dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]

        for filename in filenames:
            if len(results) >= max_results:
                return results

            if has_wildcard:
                if not fnmatch(filename.lower(), query.lower()):
                    continue
            else:
                if extensions:
                    suffix = Path(filename).suffix.lower()
                    if suffix not in extensions:
                        continue
                if query_lower not in filename.lower():
                    continue

            results.append(os.path.join(dirpath, filename))

    return results


class FilePickerScreen(ModalScreen[str]):
    """Modal file browser with search and directory tree.

    Usage from any screen::

        def _on_file_selected(path: str) -> None:
            if path:
                self.query_one("#my-input", Input).value = path

        self.app.push_screen(
            FilePickerScreen(title="Select Video", extensions=[".mp4", ".mkv"]),
            callback=_on_file_selected,
        )

    Returns the selected file path via dismiss(), or empty string if cancelled.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    DEFAULT_CSS = """
    FilePickerScreen {
        align: center middle;
    }
    #fp-dialog {
        width: 90;
        height: 35;
        border: round $accent;
        border-title-color: $text;
        border-title-style: bold;
        background: $surface;
        padding: 1 2;
    }
    #fp-dialog:focus-within {
        border: round $accent 100%;
    }
    #fp-search {
        margin: 0 0 1 0;
    }
    #fp-path-input {
        margin: 0 0 1 0;
    }
    #fp-body {
        height: 1fr;
    }
    #fp-tree {
        height: 1fr;
        border: round $surface-lighten-2;
    }
    #fp-tree:focus {
        border: round $accent;
    }
    #fp-results {
        height: 1fr;
        border: round $surface-lighten-2;
        border-title-color: $text-muted;
        border-title-style: bold;
    }
    #fp-results:focus {
        border: round $accent;
    }
    #fp-status {
        color: $text-muted;
        height: 1;
    }
    #fp-buttons {
        height: auto;
        margin: 1 0 0 0;
        align: right middle;
    }
    #fp-buttons > Button {
        margin: 0 0 0 1;
    }
    """

    def __init__(
        self,
        title: str = "Select File",
        root: str | None = None,
        extensions: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._title = title
        self._root = root or get_browse_root()
        self._extensions = extensions
        self._search_mode = False

    def compose(self) -> ComposeResult:
        with Vertical(id="fp-dialog"):
            yield Input(
                placeholder="Search: name, *.mp4, servo*.mp4 (2+ chars)...",
                id="fp-search",
            )
            yield Input(
                placeholder="Or type a full path and press Enter",
                id="fp-path-input",
            )
            with Vertical(id="fp-body"):
                yield DirectoryTree(self._root, id="fp-tree")
                yield ListView(id="fp-results")
            yield Static(f"Browsing: {self._root}", id="fp-status")
            with Horizontal(id="fp-buttons"):
                yield Button("Cancel", variant="default", id="fp-cancel")
                yield Button("Select", variant="success", id="fp-select")

    def on_mount(self) -> None:
        self.query_one("#fp-dialog").border_title = self._title
        self.query_one("#fp-results").display = False
        self.query_one("#fp-results").border_title = "Search Results"
        self.query_one("#fp-search", Input).focus()

    # ── Search ────────────────────────────────────────────────

    def on_input_changed(self, event: Input.Changed) -> None:
        # Use event.control (works across all Textual versions) to identify the input
        try:
            input_id = event.control.id
        except AttributeError:
            # Fallback for older Textual
            input_id = getattr(event, "input", event).id if hasattr(event, "input") else None

        if input_id != "fp-search":
            return

        query = event.value.strip()
        if len(query) < 2:
            self._show_tree_mode()
            return

        self._show_search_mode()
        self._update_status(f"Searching for '{query}'...")
        self._run_search(query)

    @work(thread=True, exclusive=True)
    def _run_search(self, query: str) -> None:
        """Search files in background thread. exclusive=True cancels previous search."""
        results = _search_files(
            self._root, query,
            extensions=self._extensions,
            max_results=40,
            max_depth=5,
        )
        self.app.call_from_thread(self._display_results, results, query)

    def _display_results(self, results: list[str], query: str) -> None:
        """Update the search results list on the UI thread."""
        result_list = self.query_one("#fp-results", ListView)
        result_list.clear()

        if not results:
            result_list.append(ListItem(Label("[dim]No matches found[/dim]")))
            self._update_status(f"Search: '{query}' — 0 results")
            return

        for path in results:
            try:
                display = os.path.relpath(path, self._root)
            except ValueError:
                display = path
            item = ListItem(Label(f"  {display}"))
            item._file_path = path
            result_list.append(item)

        self._update_status(f"Search: '{query}' — {len(results)} result{'s' if len(results) != 1 else ''}")

    def _show_search_mode(self) -> None:
        if not self._search_mode:
            self._search_mode = True
            self.query_one("#fp-tree").display = False
            self.query_one("#fp-results").display = True

    def _show_tree_mode(self) -> None:
        if self._search_mode:
            self._search_mode = False
            self.query_one("#fp-tree").display = True
            self.query_one("#fp-results").display = False
            self._update_status(f"Browsing: {self._root}")

    # ── Tree browsing ─────────────────────────────────────────

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        path = str(event.path)
        if self._extensions:
            suffix = Path(path).suffix.lower()
            if suffix not in self._extensions:
                return
        self.query_one("#fp-path-input", Input).value = path

    def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        self._update_status(f"Browsing: {event.path}")

    # ── Search result selection ───────────────────────────────

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        path = getattr(event.item, "_file_path", None)
        if path:
            self.query_one("#fp-path-input", Input).value = path

    # ── Actions ───────────────────────────────────────────────

    def on_input_submitted(self, event: Input.Submitted) -> None:
        try:
            input_id = event.control.id
        except AttributeError:
            input_id = getattr(event, "input", event).id if hasattr(event, "input") else None

        if input_id == "fp-path-input":
            value = event.value.strip()
            if value:
                self.dismiss(value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "fp-select":
            path = self.query_one("#fp-path-input", Input).value.strip()
            if path:
                self.dismiss(path)
        elif event.button.id == "fp-cancel":
            self.dismiss("")

    def action_cancel(self) -> None:
        self.dismiss("")

    # ── Helpers ───────────────────────────────────────────────

    def _update_status(self, text: str) -> None:
        self.query_one("#fp-status", Static).update(text)
