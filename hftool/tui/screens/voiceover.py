"""VoiceoverScreen — multi-step voiceover wizard with inline script editor."""

import json
import os
import threading
from typing import Optional

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import (
    Button, Footer, Header, Input, Label, ProgressBar,
    RadioButton, RadioSet, RichLog, Select, Static, TextArea,
)


class VoiceoverScreen(Screen):
    """Multi-step voiceover wizard.

    Modes:
    - Auto: video → scene detection → VLM → script editor → TTS → merge
    - Re-voice: video → ASR → script editor → TTS → merge
    - From Script: script file → TTS → merge
    """

    BINDINGS = [
        Binding("escape", "cancel_or_back", "Cancel/Back"),
    ]

    DEFAULT_CSS = """
    VoiceoverScreen {
        layout: vertical;
    }
    #vo-content {
        padding: 1 2;
        height: 1fr;
    }
    .field-label {
        margin: 1 0 0 0;
        text-style: bold;
    }
    #mode-select {
        height: auto;
        margin: 0 0 1 0;
    }
    #video-input, #output-input, #script-input {
        margin: 0 0 1 0;
    }
    #options-row {
        height: auto;
        margin: 0 0 1 0;
    }
    #options-row > * {
        width: 1fr;
        margin: 0 1 0 0;
    }
    #start-btn {
        margin: 1 0;
    }
    #stage-label {
        margin: 1 0;
        text-style: bold;
    }
    #progress {
        margin: 0 0 1 0;
    }
    #script-editor {
        height: 16;
        margin: 1 0;
    }
    #editor-buttons {
        height: auto;
        margin: 0 0 1 0;
    }
    #log {
        height: 1fr;
        min-height: 6;
        border: solid $surface-lighten-2;
    }
    #result-panel {
        padding: 1 2;
        background: $surface;
        border: solid $success;
        margin: 1 0;
        height: auto;
    }
    """

    def __init__(self):
        super().__init__()
        self._running = False
        self._script_ready = threading.Event()
        self._edited_script: Optional[str] = None
        self._cancel_event = threading.Event()

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="vo-content"):
            yield Label("[bold]Voiceover Wizard[/bold]")

            # Mode selection
            yield Label("Mode:", classes="field-label")
            with RadioSet(id="mode-select"):
                yield RadioButton("Auto (video → VLM → script → TTS)", id="mode-auto", value=True)
                yield RadioButton("Re-voice (video → ASR → TTS)", id="mode-revoice")
                yield RadioButton("From Script (script → TTS)", id="mode-script")

            # Input fields
            yield Label("Video path:", classes="field-label")
            yield Input(placeholder="/workspace/input.mp4", id="video-input")

            yield Label("Script path (From Script mode only):", classes="field-label")
            yield Input(placeholder="/workspace/script.srt", id="script-input")

            yield Label("Output path:", classes="field-label")
            yield Input(placeholder="/workspace/output.mp4", id="output-input")

            # Options
            yield Label("Options:", classes="field-label")
            with Horizontal(id="options-row"):
                yield Select(
                    [("kokoro", "kokoro"), ("chatterbox", "chatterbox")],
                    value="kokoro",
                    id="tts-model-select",
                )
                yield Select(
                    [("tutorial", "tutorial"), ("presentation", "presentation"),
                     ("demo", "demo"), ("casual", "casual"), ("formal", "formal")],
                    value="tutorial",
                    id="style-select",
                )
                yield Input(placeholder="Device (auto)", value="auto", id="device-input")

            yield Button("Start Voiceover", variant="success", id="start-btn")

            # Progress section (shown during execution)
            yield Label("", id="stage-label")
            yield ProgressBar(total=100, show_eta=True, id="progress")

            # Inline script editor (shown when script is ready for review)
            yield Label("[bold]Edit Script:[/bold] Modify the generated script below, then click Continue.", id="editor-label")
            yield TextArea(id="script-editor")
            with Horizontal(id="editor-buttons"):
                yield Button("Continue", variant="success", id="continue-btn")
                yield Button("Cancel", variant="error", id="cancel-edit-btn")

            yield RichLog(highlight=True, markup=True, id="log")
            yield Static(id="result-panel")
        yield Footer()

    def on_mount(self) -> None:
        # Hide progress/editor sections initially
        self.query_one("#stage-label").display = False
        self.query_one("#progress").display = False
        self.query_one("#editor-label").display = False
        self.query_one("#script-editor").display = False
        self.query_one("#editor-buttons").display = False
        self.query_one("#result-panel").display = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start-btn":
            self._start_voiceover()
        elif event.button.id == "continue-btn":
            self._continue_after_edit()
        elif event.button.id == "cancel-edit-btn":
            self._cancel_event.set()
            self._script_ready.set()  # Unblock worker thread

    def _start_voiceover(self) -> None:
        """Validate inputs and launch voiceover in worker thread."""
        # Determine mode
        mode_set = self.query_one("#mode-select", RadioSet)
        mode = "auto"
        if mode_set.pressed_index == 1:
            mode = "revoice"
        elif mode_set.pressed_index == 2:
            mode = "script"

        video_path = self.query_one("#video-input", Input).value.strip()
        script_path = self.query_one("#script-input", Input).value.strip()
        output_path = self.query_one("#output-input", Input).value.strip()
        tts_model = self.query_one("#tts-model-select", Select).value
        style = self.query_one("#style-select", Select).value
        device = self.query_one("#device-input", Input).value.strip() or "auto"

        # Validate
        if mode in ("auto", "revoice") and not video_path:
            self.notify("Video path is required", severity="error")
            return
        if mode == "script" and not script_path:
            self.notify("Script path is required", severity="error")
            return
        if not output_path:
            self.notify("Output path is required", severity="error")
            return

        # Show progress, hide config
        self.query_one("#start-btn").display = False
        self.query_one("#stage-label").display = True
        self.query_one("#progress").display = True

        self._running = True
        self._log(f"Starting voiceover ({mode} mode)...")

        self._run_voiceover_worker(
            mode, video_path, script_path, output_path,
            tts_model, style, device,
        )

    @work(thread=True)
    def _run_voiceover_worker(
        self,
        mode: str,
        video_path: str,
        script_path: str,
        output_path: str,
        tts_model: str,
        style: str,
        device: str,
    ) -> None:
        """Run the voiceover pipeline in a background thread."""
        from hftool.tasks.voiceover import VoiceoverTask

        try:
            task = VoiceoverTask(
                device=device,
                tts_model=tts_model,
                narration_style=style,
                no_edit=True,  # We handle editing in the TUI
            )

            if mode == "auto":
                self._run_auto_with_edit(task, video_path, output_path)
            elif mode == "revoice":
                self._run_revoice_with_edit(task, video_path, output_path)
            else:
                # Script mode — run directly
                self.call_from_thread(self._update_stage, "Running from script...")
                task.run(
                    script_path=script_path,
                    output_path=output_path,
                    video_path=video_path if video_path else None,
                )

            if not self._cancel_event.is_set():
                self.call_from_thread(self._on_voiceover_done, output_path, None)
            else:
                self.call_from_thread(self._on_voiceover_done, None, "Cancelled by user")

        except Exception as e:
            self.call_from_thread(self._on_voiceover_done, None, str(e))
        finally:
            try:
                task.cleanup()
            except Exception:
                pass

    def _run_auto_with_edit(self, task, video_path: str, output_path: str) -> None:
        """Run auto voiceover with TUI script editing pause."""
        from hftool.utils.deps import check_ffmpeg
        from hftool.io.scene_detector import detect_scenes, extract_keyframes
        from hftool.io.script_generator import analyze_frames, generate_script

        check_ffmpeg()

        work_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(work_dir, exist_ok=True)

        # Step 1: Scene detection
        self.call_from_thread(self._update_stage, "Step 1: Detecting scenes...")
        scenes = detect_scenes(video_path, threshold=task.scene_threshold)
        self.call_from_thread(self._log, f"  Found {len(scenes.scenes)} scenes")

        if self._cancel_event.is_set():
            return

        # Step 2: Keyframe extraction
        self.call_from_thread(self._update_stage, "Step 2: Extracting keyframes...")
        keyframe_dir = os.path.join(work_dir, "voiceover_keyframes")
        scenes = extract_keyframes(video_path, scenes, keyframe_dir)
        total_frames = sum(len(s.keyframe_paths) for s in scenes.scenes)
        self.call_from_thread(self._log, f"  Extracted {total_frames} keyframes")

        if self._cancel_event.is_set():
            return

        # Step 3: VLM analysis
        self.call_from_thread(self._update_stage, "Step 3: Analyzing frames with VLM...")
        task._load_vlm()
        analyses = analyze_frames(task._vlm_task, scenes)
        self.call_from_thread(self._log, f"  Analyzed {len(analyses)} frames")

        if self._cancel_event.is_set():
            task._unload_vlm()
            return

        # Step 4: Generate script
        self.call_from_thread(self._update_stage, "Step 4: Generating script...")
        script = generate_script(
            task._vlm_task, analyses, scenes,
            style=task.narration_style,
            video_duration_ms=scenes.video_duration_ms,
        )
        self.call_from_thread(self._log, f"  Generated {len(script.segments)} segments")

        # Unload VLM before TTS
        task._unload_vlm()

        if self._cancel_event.is_set():
            return

        # Step 5: Pause for script editing
        self.call_from_thread(self._show_script_editor, script)
        self._script_ready.wait()  # Block until user clicks Continue

        if self._cancel_event.is_set():
            return

        # Parse edited script
        if self._edited_script:
            try:
                from hftool.io.script_parser import parse_script_json
                script = parse_script_json(self._edited_script)
            except Exception as e:
                self.call_from_thread(self._log, f"[yellow]Script parse error, using original: {e}[/yellow]")

        # Step 6: TTS + merge
        self.call_from_thread(self._update_stage, "Step 6: Generating voiceover audio...")
        self.call_from_thread(self._hide_editor)
        task._generate_and_merge(script, output_path, video_path, keep_audio=False)

    def _run_revoice_with_edit(self, task, video_path: str, output_path: str) -> None:
        """Run re-voice with TUI script editing pause."""
        from hftool.utils.deps import check_ffmpeg
        from hftool.io.script_review import review_script

        check_ffmpeg()

        work_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(work_dir, exist_ok=True)

        # Step 1: Extract audio
        self.call_from_thread(self._update_stage, "Step 1: Extracting audio...")
        audio_path = os.path.join(work_dir, "extracted_audio.wav")
        task._extract_audio(video_path, audio_path)

        if self._cancel_event.is_set():
            return

        # Step 2: ASR transcription
        self.call_from_thread(self._update_stage, "Step 2: Transcribing audio...")
        script = task._transcribe(audio_path)
        self.call_from_thread(self._log, f"  Transcribed {len(script.segments)} segments")

        if self._cancel_event.is_set():
            return

        # Step 3: Pause for editing
        self.call_from_thread(self._show_script_editor, script)
        self._script_ready.wait()

        if self._cancel_event.is_set():
            return

        if self._edited_script:
            try:
                from hftool.io.script_parser import parse_script_json
                script = parse_script_json(self._edited_script)
            except Exception as e:
                self.call_from_thread(self._log, f"[yellow]Script parse error, using original: {e}[/yellow]")

        # Step 4: TTS + merge
        self.call_from_thread(self._update_stage, "Step 4: Generating voiceover audio...")
        self.call_from_thread(self._hide_editor)
        task._generate_and_merge(script, output_path, video_path, keep_audio=False)

    # --- UI update methods (called on UI thread via call_from_thread) ---

    def _update_stage(self, text: str) -> None:
        self.query_one("#stage-label", Label).update(f"[bold]{text}[/bold]")

    def _show_script_editor(self, script) -> None:
        """Show the inline script editor with the generated script."""
        self._update_stage("Step 5: Review and edit script")
        self._log("[green]Script ready for editing. Modify below and click Continue.[/green]")

        # Serialize script to JSON for editing
        try:
            script_json = json.dumps(
                [{"start_ms": s.start_ms, "end_ms": s.end_ms, "text": s.text}
                 for s in script.segments],
                indent=2,
            )
        except Exception:
            script_json = str(script)

        editor = self.query_one("#script-editor", TextArea)
        editor.load_text(script_json)
        self.query_one("#editor-label").display = True
        editor.display = True
        self.query_one("#editor-buttons").display = True
        editor.focus()

    def _hide_editor(self) -> None:
        self.query_one("#editor-label").display = False
        self.query_one("#script-editor").display = False
        self.query_one("#editor-buttons").display = False

    def _continue_after_edit(self) -> None:
        """User clicked Continue — unblock the worker thread."""
        editor = self.query_one("#script-editor", TextArea)
        self._edited_script = editor.text
        self._script_ready.set()

    def _on_voiceover_done(self, output_path: Optional[str], error: Optional[str]) -> None:
        self._running = False
        result_panel = self.query_one("#result-panel", Static)
        self._hide_editor()

        if error:
            self.query_one("#stage-label", Label).update("[red]Failed[/red]")
            self._log(f"[red]Error: {error}[/red]")
            result_panel.update(f"[red]{error}[/red]")
        else:
            self.query_one("#stage-label", Label).update("[green]Complete![/green]")
            self.query_one("#progress", ProgressBar).update(total=100, progress=100)
            size_str = ""
            if output_path and os.path.exists(output_path):
                size = os.path.getsize(output_path)
                size_str = f" ({size / (1024*1024):.1f} MB)" if size > 1024*1024 else f" ({size / 1024:.1f} KB)"
            result_panel.update(f"[bold green]Output:[/bold green] {output_path}{size_str}")
            self._log(f"[green]Voiceover complete: {output_path}[/green]")

        result_panel.display = True

    def _log(self, text: str) -> None:
        try:
            self.query_one("#log", RichLog).write(text)
        except Exception:
            pass

    def action_cancel_or_back(self) -> None:
        if self._running:
            self._cancel_event.set()
            self._script_ready.set()  # Unblock if waiting for edit
            self._log("[yellow]Cancelling...[/yellow]")
        else:
            self.app.pop_screen()
