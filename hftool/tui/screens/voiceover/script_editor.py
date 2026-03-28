"""Inline script editor mixin for the voiceover TUI screen.

Manages the TextArea-based JSON script editor that appears between
VLM script generation and TTS synthesis, allowing users to review
and edit the narration script.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Optional

from textual.widgets import RichLog, TextArea

from hftool.tui.screens.voiceover.path_utils import container_to_host_path


class ScriptEditorMixin:
    """Mixin that manages the inline script editor lifecycle.

    Expects the host Screen to contain these widget IDs:
    - ``#editor-section``, ``#script-editor``, ``#log``

    State:
        _script_ready: Event signalled when the user clicks Continue.
        _edited_script: The raw JSON string from the TextArea after editing.
        _keyframe_dir: Path to the keyframe directory for the current run.
    """

    _script_ready: threading.Event
    _edited_script: Optional[str]
    _keyframe_dir: Optional[str]

    def _init_script_editor(self) -> None:
        """Initialise script editor state. Call from ``__init__``."""
        self._script_ready = threading.Event()
        self._edited_script = None
        self._keyframe_dir = None

    # ------------------------------------------------------------------
    # Show / populate
    # ------------------------------------------------------------------

    def _show_script_editor(self, script) -> None:
        """Show the inline script editor with clickable keyframe links."""
        self._update_stage("Step 6/7 — Review and edit script")
        self._log("[green]Script ready for editing. Modify below and click Continue.[/green]")
        self._log("[dim]Tip: Delete segments for silence. Edit timestamps as M:SS.[/dim]")
        if os.environ.get("HFTOOL_IN_DOCKER"):
            self._log("[dim]Copy: Shift+mouse select to copy text to host clipboard.[/dim]")

        self._show_keyframe_links(script)

        try:
            script_text = script.to_editor_json()
        except Exception:
            script_text = json.dumps(
                [{"start_ms": s.start_ms, "end_ms": s.end_ms, "text": s.text}
                 for s in script.segments],
                indent=2,
            )

        editor = self.query_one("#script-editor", TextArea)
        self.query_one("#editor-section").display = True
        try:
            editor.load_text(script_text)
        except Exception:
            editor.text = script_text
        self.call_after_refresh(self._reveal_editor)
        self.notify(
            "Script ready for review. Use Continue or Cancel.",
            severity="information",
            timeout=5,
        )

    def _show_keyframe_links(self, script) -> None:
        """Show keyframe paths in the log above the editor."""
        from hftool.io.script_parser import _ms_to_short_timestamp

        contexts = script.metadata.get("scene_contexts", [])
        if not contexts:
            return

        log = self.query_one("#log", RichLog)
        in_docker = bool(os.environ.get("HFTOOL_IN_DOCKER"))

        shown_paths: set = set()
        for seg in script.segments:
            mid = (seg.start_ms + seg.end_ms) // 2
            best_ctx, best_dist = None, float("inf")
            for ctx in contexts:
                dist = abs(ctx.get("timestamp_ms", 0) - mid)
                if dist < best_dist:
                    best_dist = dist
                    best_ctx = ctx

            if not best_ctx:
                continue

            img_path = best_ctx.get("image_path", "")
            if not img_path or not os.path.isfile(img_path) or img_path in shown_paths:
                continue
            shown_paths.add(img_path)

            host_path = container_to_host_path(img_path) if in_docker else img_path
            ts = _ms_to_short_timestamp(seg.start_ms)
            desc = best_ctx.get("description", "")[:60]
            log.write(
                f"  [bold cyan]Seg {seg.id}[/bold cyan] [{ts}] "
                f"[bold]{host_path}[/bold]  "
                f"[dim]{desc}[/dim]"
            )

    # ------------------------------------------------------------------
    # Hide / reveal / continue
    # ------------------------------------------------------------------

    def _hide_editor(self) -> None:
        self.query_one("#editor-section").display = False

    def _reveal_editor(self) -> None:
        """Scroll editor into view and focus it after layout refresh."""
        try:
            editor_section = self.query_one("#editor-section")
            editor_section.scroll_visible(top=True)
        except Exception:
            pass

        try:
            self.query_one("#script-editor", TextArea).focus()
        except Exception:
            pass

    def _continue_after_edit(self) -> None:
        """User clicked Continue — unblock the worker thread."""
        editor = self.query_one("#script-editor", TextArea)
        self._edited_script = editor.text
        self._script_ready.set()
