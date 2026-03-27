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
    #mode-section, #files-section, #options-section {
        border: round $accent 40%;
        border-title-color: $text-muted;
        border-title-style: bold;
        padding: 1 2;
        margin: 0 0 1 0;
        height: auto;
    }
    #mode-section:focus-within, #files-section:focus-within, #options-section:focus-within {
        border: round $accent 100%;
        border-title-color: $text;
    }
    #mode-select {
        height: auto;
    }
    .file-row {
        height: auto;
        margin: 0 0 1 0;
    }
    .file-row > Input {
        width: 1fr;
    }
    .file-row > Button {
        width: 12;
        margin: 0 0 0 1;
    }
    #options-row {
        height: auto;
        margin: 0 0 1 0;
    }
    #options-row > * {
        width: 1fr;
        margin: 0 1 0 0;
    }
    #voice-row {
        height: auto;
        margin: 0 0 1 0;
    }
    #voice-row > Select {
        width: 1fr;
    }
    #voice-ref-row {
        height: auto;
        margin: 0 0 1 0;
    }
    #voice-ref-row > Input {
        width: 1fr;
    }
    #voice-ref-row > Button {
        width: 12;
        margin: 0 0 0 1;
    }
    #capture-row {
        height: auto;
    }
    #capture-row > Select {
        width: 1fr;
    }
    #start-btn {
        margin: 1 0;
    }
    #progress-section {
        border: round $accent 40%;
        border-title-color: $text-muted;
        border-title-style: bold;
        padding: 1 2;
        margin: 0 0 1 0;
        height: auto;
    }
    #stage-label {
        margin: 0 0 1 0;
        text-style: bold;
    }
    #editor-section {
        border: round $warning 60%;
        border-title-color: $text;
        border-title-style: bold;
        padding: 1 2;
        margin: 0 0 1 0;
        height: auto;
    }
    #script-editor {
        height: 16;
        margin: 0 0 1 0;
    }
    #editor-buttons {
        height: auto;
    }
    #log {
        height: 1fr;
        min-height: 6;
    }
    #result-panel {
        border: round $success 60%;
        border-title-color: $text-muted;
        border-title-style: bold;
        padding: 1 2;
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
            yield Label("[bold]Voiceover Wizard[/bold]", classes="section-title")

            # Mode selection
            with Vertical(id="mode-section"):
                with RadioSet(id="mode-select"):
                    yield RadioButton("Auto (video → VLM → script → TTS)", id="mode-auto", value=True)
                    yield RadioButton("Re-voice (video → ASR → TTS)", id="mode-revoice")
                    yield RadioButton("From Script (script → TTS)", id="mode-script")

            # File inputs with browse buttons
            with Vertical(id="files-section"):
                yield Label("Video:", classes="field-label")
                with Horizontal(classes="file-row"):
                    yield Input(placeholder="Path to input video", id="video-input")
                    yield Button("Browse", id="browse-video")

                yield Label("Script (From Script mode):", classes="field-label")
                with Horizontal(classes="file-row"):
                    yield Input(placeholder="Path to SRT or JSON script", id="script-input")
                    yield Button("Browse", id="browse-script")

                yield Label("Output:", classes="field-label")
                with Horizontal(classes="file-row"):
                    yield Input(placeholder="Output file path", id="output-input")
                    yield Button("Browse", id="browse-output")

            # Options
            with Vertical(id="options-section"):
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

                with Horizontal(id="voice-row"):
                    yield Select(
                        [
                            ("af_heart (Female, warm)", "af_heart"),
                            ("af_bella (Female, clear)", "af_bella"),
                            ("af_nicole (Female, soft)", "af_nicole"),
                            ("af_nova (Female, bright)", "af_nova"),
                            ("af_sarah (Female, natural)", "af_sarah"),
                            ("af_sky (Female, airy)", "af_sky"),
                            ("am_adam (Male, deep)", "am_adam"),
                            ("am_michael (Male, warm)", "am_michael"),
                            ("am_eric (Male, clear)", "am_eric"),
                            ("am_liam (Male, young)", "am_liam"),
                            ("bf_emma (Female, British)", "bf_emma"),
                            ("bf_lily (Female, British)", "bf_lily"),
                            ("bm_george (Male, British)", "bm_george"),
                            ("bm_daniel (Male, British)", "bm_daniel"),
                        ],
                        value="af_heart",
                        id="voice-select",
                    )

                # Voice cloning (Chatterbox only — hidden by default)
                with Horizontal(id="voice-ref-row"):
                    yield Input(placeholder="Voice clone ref audio (.wav)", id="voice-ref-input")
                    yield Button("Browse", id="browse-voice-ref")

                with Horizontal(id="capture-row"):
                    yield Select(
                        [
                            ("Auto (scene detection)", "auto"),
                            ("Every 3 seconds", "3"),
                            ("Every 5 seconds", "5"),
                            ("Every 10 seconds", "10"),
                            ("Every 15 seconds", "15"),
                            ("Every 30 seconds", "30"),
                        ],
                        value="auto",
                        id="capture-interval-select",
                    )

            yield Button("Start Voiceover", variant="success", id="start-btn")

            # Progress section (shown during execution)
            with Vertical(id="progress-section"):
                yield Label("", id="stage-label")
                yield ProgressBar(total=100, show_eta=True, id="progress")

            # Inline script editor (shown when script is ready for review)
            with Vertical(id="editor-section"):
                yield TextArea(id="script-editor")
                with Horizontal(id="editor-buttons"):
                    yield Button("Continue", variant="success", id="continue-btn")
                    yield Button("Cancel", variant="error", id="cancel-edit-btn")

            yield RichLog(highlight=True, markup=True, id="log")
            yield Static(id="result-panel")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#mode-section").border_title = "Mode"
        self.query_one("#files-section").border_title = "Files"
        self.query_one("#options-section").border_title = "Options"
        self.query_one("#voice-ref-row").display = False  # Only for Chatterbox
        self.query_one("#progress-section").border_title = "Progress"
        self.query_one("#progress-section").display = False
        self.query_one("#editor-section").border_title = "Edit Script"
        self.query_one("#editor-section").display = False
        self.query_one("#result-panel").display = False
        self.query_one("#result-panel").border_title = "Result"

    def on_select_changed(self, event: Select.Changed) -> None:
        """Show/hide voice cloning row based on TTS model selection."""
        try:
            control_id = event.control.id
        except AttributeError:
            return
        if control_id == "tts-model-select":
            is_chatterbox = event.value == "chatterbox"
            self.query_one("#voice-ref-row").display = is_chatterbox
            # Hide voice presets for chatterbox (uses cloned voice)
            self.query_one("#voice-row").display = not is_chatterbox

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start-btn":
            self._start_voiceover()
        elif event.button.id == "continue-btn":
            self._continue_after_edit()
        elif event.button.id == "cancel-edit-btn":
            self._cancel_event.set()
            self._script_ready.set()
        elif event.button.id == "browse-video":
            self._browse_file("video-input", "Select Video", [".mp4", ".mkv", ".avi", ".mov", ".webm"])
        elif event.button.id == "browse-script":
            self._browse_file("script-input", "Select Script", [".srt", ".json"])
        elif event.button.id == "browse-output":
            self._browse_file("output-input", "Select Output Location", [".mp4", ".wav"])
        elif event.button.id == "browse-voice-ref":
            self._browse_file("voice-ref-input", "Select Voice Reference Audio", [".wav", ".mp3", ".flac", ".ogg"])

    def _browse_file(self, input_id: str, title: str, extensions: list[str]) -> None:
        """Open the file picker modal and populate the input on selection."""
        from hftool.tui.widgets.file_browser import FilePickerScreen

        def on_selected(path: str) -> None:
            if path:
                self.query_one(f"#{input_id}", Input).value = path

        self.app.push_screen(FilePickerScreen(title=title, extensions=extensions), on_selected)

    def _start_voiceover(self) -> None:
        """Validate inputs and launch voiceover in worker thread."""
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
        voice = self.query_one("#voice-select", Select).value
        voice_ref = self.query_one("#voice-ref-input", Input).value.strip()
        capture_interval = self.query_one("#capture-interval-select", Select).value

        if mode in ("auto", "revoice") and not video_path:
            self.notify("Video path is required", severity="error")
            return
        if mode == "script" and not script_path:
            self.notify("Script path is required", severity="error")
            return
        if not output_path:
            self.notify("Output path is required", severity="error")
            return

        # Reset state from any previous run
        self._script_ready = threading.Event()
        self._edited_script = None
        self._cancel_event = threading.Event()
        self.query_one("#log", RichLog).clear()
        self.query_one("#result-panel").display = False
        self.query_one("#editor-section").display = False
        self.query_one("#stage-label", Label).update("")
        self.query_one("#progress", ProgressBar).update(total=100, progress=0)

        self.query_one("#start-btn").display = False
        self.query_one("#progress-section").display = True

        self._running = True
        self._log(f"Starting voiceover ({mode} mode)...")

        self._run_voiceover_worker(
            mode, video_path, script_path, output_path,
            tts_model, style, device, voice, voice_ref, capture_interval,
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
        voice: str = "af_heart",
        voice_ref: str = "",
        capture_interval: str = "auto",
    ) -> None:
        """Run the voiceover pipeline in a background thread."""
        import sys, io
        from hftool.tasks.voiceover import VoiceoverTask

        # Capture stderr to prevent HIP/MIOpen warnings from spilling
        # onto the terminal behind the TUI
        _original_stderr = sys.stderr
        sys.stderr = io.StringIO()

        try:
            task = VoiceoverTask(
                device=device,
                tts_model=tts_model,
                voice=voice,
                voice_ref=voice_ref or None,
                narration_style=style,
                no_edit=True,  # We handle editing in the TUI
            )

            # Store voice and capture interval for the auto flow
            self._voice = voice
            self._capture_interval = capture_interval

            if mode == "auto":
                self._run_auto_with_edit(task, video_path, output_path)
            elif mode == "revoice":
                self._run_revoice_with_edit(task, video_path, output_path)
            else:
                self.app.call_from_thread(self._update_stage, "Running from script...")
                task.run(
                    script_path=script_path,
                    output_path=output_path,
                    video_path=video_path if video_path else None,
                )

            if not self._cancel_event.is_set():
                self.app.call_from_thread(self._on_voiceover_done, output_path, None)
            else:
                self.app.call_from_thread(self._on_voiceover_done, None, "Cancelled by user")

        except SystemExit as e:
            # spacy.cli.download calls sys.exit on failure — catch it
            self.app.call_from_thread(self._on_voiceover_done, None, f"A dependency failed to install (exit code {e.code}). Try rebuilding the Docker image.")
        except Exception as e:
            self.app.call_from_thread(self._on_voiceover_done, None, str(e))
        finally:
            # Restore stderr and log any captured warnings
            captured = sys.stderr.getvalue() if hasattr(sys.stderr, 'getvalue') else ""
            sys.stderr = _original_stderr
            if captured.strip():
                # Show non-empty warnings in the log
                for line in captured.strip().split("\n")[:5]:  # max 5 lines
                    self.app.call_from_thread(self._log, f"[dim]{line}[/dim]")

            try:
                task.cleanup()
            except Exception:
                pass

    def _run_auto_with_edit(self, task, video_path: str, output_path: str) -> None:
        """Run auto voiceover with TUI script editing pause."""
        from hftool.utils.deps import check_ffmpeg
        from hftool.io.scene_detector import detect_scenes, extract_keyframes

        check_ffmpeg()

        work_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(work_dir, exist_ok=True)

        # Step 1: Scene detection (or fixed interval)
        capture_interval = getattr(self, '_capture_interval', 'auto')
        if capture_interval != "auto":
            interval_s = float(capture_interval)
            self.app.call_from_thread(self._update_stage, f"Step 1/6 — Capturing every {interval_s:.0f}s...")
            self.app.call_from_thread(self._set_progress_indeterminate)
            # Use fixed intervals instead of scene detection
            from hftool.io.scene_detector import _fixed_interval_scenes, get_video_duration_ms, SceneDetectionResult, SceneInfo
            duration_ms = get_video_duration_ms(video_path)
            intervals = _fixed_interval_scenes(duration_ms, interval_s=interval_s)
            scenes = SceneDetectionResult(
                scenes=[SceneInfo(index=i, start_ms=s, end_ms=e) for i, (s, e) in enumerate(intervals)],
                video_duration_ms=duration_ms,
                video_path=video_path,
            )
        else:
            self.app.call_from_thread(self._update_stage, "Step 1/6 — Detecting scenes...")
            self.app.call_from_thread(self._set_progress_indeterminate)
            scenes = detect_scenes(video_path, threshold=task.scene_threshold)
        self.app.call_from_thread(self._log, f"  Found {len(scenes.scenes)} scenes")

        if self._cancel_event.is_set():
            return

        # Step 2: Keyframe extraction
        self.app.call_from_thread(self._update_stage, "Step 2/6 — Extracting keyframes...")
        keyframe_dir = os.path.join(work_dir, "voiceover_keyframes")
        scenes = extract_keyframes(video_path, scenes, keyframe_dir)
        total_frames = sum(len(s.keyframe_paths) for s in scenes.scenes)
        self.app.call_from_thread(self._log, f"  Extracted {total_frames} keyframes")

        if self._cancel_event.is_set():
            return

        # Step 3: VLM analysis (per-frame progress)
        total_frames = sum(len(s.keyframe_paths) for s in scenes.scenes)
        self.app.call_from_thread(self._update_stage, f"Step 3/6 — Analyzing frames with VLM (0/{total_frames})...")
        self.app.call_from_thread(self._update_progress, 0, total_frames)
        task._load_vlm()

        # Inline the frame analysis loop for per-frame progress
        from hftool.io.script_generator import FrameAnalysis, FRAME_ANALYSIS_PROMPT
        analyses = []
        prev_description = ""
        frame_idx = 0
        for scene in scenes.scenes:
            for image_path in scene.keyframe_paths:
                if self._cancel_event.is_set():
                    task._unload_vlm()
                    return
                prompt = FRAME_ANALYSIS_PROMPT.format(
                    previous_description=prev_description or "None yet."
                )
                description = task._vlm_task.analyze_frame(
                    image_path, prompt, previous_context=prev_description,
                )
                analyses.append(FrameAnalysis(
                    scene_index=scene.index,
                    timestamp_ms=scene.start_ms,
                    image_path=image_path,
                    description=description,
                ))
                prev_description = description
                frame_idx += 1
                self.app.call_from_thread(
                    self._update_stage,
                    f"Step 3/6 — Analyzing frames with VLM ({frame_idx}/{total_frames})...",
                )
                self.app.call_from_thread(self._update_progress, frame_idx, total_frames)

                # Free VRAM between frames — KV cache accumulates
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

        self.app.call_from_thread(self._log, f"  Analyzed {len(analyses)} frames")

        # Aggressively free VRAM before script generation — the frame analysis
        # loop accumulates KV cache fragments that prevent the large allocation
        # needed for the script assembly prompt (all 48 descriptions at once)
        try:
            import torch, gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        if self._cancel_event.is_set():
            task._unload_vlm()
            return

        # Step 4: Generate script
        self.app.call_from_thread(self._update_stage, "Step 4/6 — Generating script...")
        script = generate_script(
            task._vlm_task, analyses, scenes,
            style=task.narration_style,
            video_duration_ms=scenes.video_duration_ms,
        )
        self.app.call_from_thread(self._log, f"  Generated {len(script.segments)} segments")

        # Unload VLM before TTS
        task._unload_vlm()

        if self._cancel_event.is_set():
            return

        # Step 5: Pause for script editing
        self.app.call_from_thread(self._show_script_editor, script)
        self._script_ready.wait()  # Block until user clicks Continue

        if self._cancel_event.is_set():
            return

        # Parse edited script
        if self._edited_script:
            try:
                import json as _json
                from hftool.io.script_parser import ScriptData, ScriptSegment
                raw = _json.loads(self._edited_script)
                script = ScriptData(segments=[
                    ScriptSegment(
                        id=i + 1,
                        start_ms=int(s.get("start_ms", 0)),
                        end_ms=int(s.get("end_ms", 0)),
                        text=s.get("text", ""),
                    )
                    for i, s in enumerate(raw)
                ])
            except Exception as e:
                self.app.call_from_thread(self._log, f"[yellow]Script parse error, using original: {e}[/yellow]")

        # Step 6: TTS + merge
        self.app.call_from_thread(self._update_stage, "Step 6/6 — Generating voiceover audio...")
        self.app.call_from_thread(self._hide_editor)
        self.app.call_from_thread(self._set_progress_indeterminate)
        task._generate_and_merge(script, output_path, video_path, keep_audio=False)

    def _run_revoice_with_edit(self, task, video_path: str, output_path: str) -> None:
        """Run re-voice with TUI script editing pause."""
        from hftool.utils.deps import check_ffmpeg

        check_ffmpeg()

        work_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(work_dir, exist_ok=True)

        # Step 1: Extract audio
        self.app.call_from_thread(self._update_stage, "Step 1/4 — Extracting audio...")
        audio_path = os.path.join(work_dir, "extracted_audio.wav")
        task._extract_audio(video_path, audio_path)

        if self._cancel_event.is_set():
            return

        # Step 2: ASR transcription
        self.app.call_from_thread(self._update_stage, "Step 2/4 — Transcribing audio...")
        script = task._transcribe_to_script(audio_path)
        self.app.call_from_thread(self._log, f"  Transcribed {len(script.segments)} segments")

        if self._cancel_event.is_set():
            return

        # Step 3: Pause for editing
        self.app.call_from_thread(self._show_script_editor, script)
        self._script_ready.wait()

        if self._cancel_event.is_set():
            return

        if self._edited_script:
            try:
                import json as _json
                from hftool.io.script_parser import ScriptData, ScriptSegment
                raw = _json.loads(self._edited_script)
                script = ScriptData(segments=[
                    ScriptSegment(
                        id=i + 1,
                        start_ms=int(s.get("start_ms", 0)),
                        end_ms=int(s.get("end_ms", 0)),
                        text=s.get("text", ""),
                    )
                    for i, s in enumerate(raw)
                ])
            except Exception as e:
                self.app.call_from_thread(self._log, f"[yellow]Script parse error, using original: {e}[/yellow]")

        # Step 4: TTS + merge
        self.app.call_from_thread(self._update_stage, "Step 4/4 — Generating voiceover audio...")
        self.app.call_from_thread(self._hide_editor)
        task._generate_and_merge(script, output_path, video_path, keep_audio=False)

    # --- UI update methods (called on UI thread via call_from_thread) ---

    def _update_stage(self, text: str) -> None:
        self.query_one("#stage-label", Label).update(f"[bold]{text}[/bold]")

    def _update_progress(self, current: int, total: int) -> None:
        self.query_one("#progress", ProgressBar).update(total=total, progress=current)

    def _set_progress_indeterminate(self) -> None:
        self.query_one("#progress", ProgressBar).update(total=None)

    def _show_script_editor(self, script) -> None:
        """Show the inline script editor with the generated script."""
        self._update_stage("Step 5/6 — Review and edit script")
        self._log("[green]Script ready for editing. Modify below and click Continue.[/green]")

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
        self.query_one("#editor-section").display = True
        editor.focus()

    def _hide_editor(self) -> None:
        self.query_one("#editor-section").display = False

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
            result_panel.add_class("-error")
            result_panel.border_title = "Error"
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

        # Show start button again so user can run another voiceover
        self.query_one("#start-btn").display = True
        self.query_one("#start-btn").label = "Start New Voiceover"

    def _log(self, text: str) -> None:
        try:
            self.query_one("#log", RichLog).write(text)
        except Exception:
            pass

    def action_cancel_or_back(self) -> None:
        if self._running:
            self._cancel_event.set()
            self._script_ready.set()
            self._log("[yellow]Cancelling...[/yellow]")
        else:
            self.app.pop_screen()
