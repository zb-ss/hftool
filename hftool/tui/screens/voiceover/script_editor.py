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

    # ------------------------------------------------------------------
    # Save / Load script
    # ------------------------------------------------------------------

    def _save_script(self) -> None:
        """Save the current editor content to a JSON file."""
        editor = self.query_one("#script-editor", TextArea)
        script_text = editor.text.strip()
        if not script_text:
            self.notify("Nothing to save — script is empty.", severity="warning")
            return

        # Derive save path next to the output file, or fall back to home/cwd
        output_val = ""
        try:
            from textual.widgets import Input
            output_val = self.query_one("#output-input", Input).value.strip()
        except Exception:
            pass

        if output_val:
            save_dir = os.path.dirname(os.path.abspath(output_val))
            base = os.path.splitext(os.path.basename(output_val))[0]
            save_path = os.path.join(save_dir, f"{base}_script.json")
        else:
            save_path = os.path.join(os.getcwd(), "voiceover_script.json")

        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(script_text)

            display_path = save_path
            if os.environ.get("HFTOOL_IN_DOCKER"):
                display_path = container_to_host_path(save_path)

            self.notify(f"Script saved to {display_path}", severity="information", timeout=8)
            log = self.query_one("#log", RichLog)
            log.write(f"[green]Script saved:[/green] {display_path}")
        except Exception as exc:
            self.notify(f"Save failed: {exc}", severity="error", timeout=8)

    def _load_script(self) -> None:
        """Load a script JSON file into the editor."""
        from hftool.tui.widgets.file_browser import FilePickerScreen

        def on_selected(path: str) -> None:
            if not path:
                return
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                editor = self.query_one("#script-editor", TextArea)
                try:
                    editor.load_text(content)
                except Exception:
                    editor.text = content
                self.notify(f"Loaded {os.path.basename(path)}", severity="information")
                log = self.query_one("#log", RichLog)
                log.write(f"[green]Script loaded:[/green] {path}")
            except Exception as exc:
                self.notify(f"Load failed: {exc}", severity="error", timeout=8)

        self.app.push_screen(
            FilePickerScreen(title="Load Script", extensions=[".json", ".srt"]),
            on_selected,
        )
