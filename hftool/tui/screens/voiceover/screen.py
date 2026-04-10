"""VoiceoverScreen — multi-step voiceover wizard with inline script editor.

This is the thin Screen shell.  Pipeline orchestration lives in
``pipeline.py``, VLM model selection in ``vlm_selector.py``, the inline
script editor in ``script_editor.py``, and path utilities in ``path_utils.py``.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RadioButton,
    RadioSet,
    RichLog,
    Select,
    Static,
    TextArea,
)

from hftool.tui.screens.voiceover.path_utils import open_path
from hftool.tui.screens.voiceover.pipeline import VoiceoverPipeline
from hftool.tui.screens.voiceover.script_editor import ScriptEditorMixin
from hftool.tui.screens.voiceover.vlm_selector import VlmSelectorMixin


class VoiceoverScreen(VlmSelectorMixin, ScriptEditorMixin, Screen):
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
    #vlm-source-row, #vlm-local-row, #vlm-online-row {
        height: auto;
        margin: 0 0 1 0;
    }
    #vlm-source-row > Select {
        width: 1fr;
    }
    #vlm-local-row > Select {
        width: 1fr;
    }
    #vlm-online-row > Select {
        width: 1fr;
    }
    #vlm-online-row > Button {
        width: 10;
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
    #result-text {
        width: 1fr;
    }
    #open-output-btn {
        width: 10;
        margin: 0 0 0 1;
    }
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        super().__init__()
        self._init_vlm_selector()
        self._init_script_editor()
        self._running = False
        self._run_seq = 0
        self._active_run_id: Optional[int] = None
        self._cancel_event = threading.Event()
        self._output_path: Optional[str] = None

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

                yield Label("VLM (frame analysis):", classes="field-label")
                with Horizontal(id="vlm-source-row"):
                    yield Select(
                        [
                            ("Local model (in container)", "local"),
                            ("Online API model", "online"),
                        ],
                        value="local",
                        id="vlm-source-select",
                    )

                with Horizontal(id="vlm-local-row"):
                    from hftool.io.vlm_providers import get_available_vlm_options

                    vlm_options = get_available_vlm_options()
                    yield Select(
                        vlm_options,
                        value=vlm_options[0][1] if vlm_options else "qwen3-vl-8b",
                        id="vlm-local-select",
                    )

                with Horizontal(id="vlm-online-row"):
                    from hftool.io.vlm_providers import (
                        DEFAULT_ONLINE_VLM_MODEL,
                        get_default_cloud_vlm_models,
                    )

                    default_google_models = get_default_cloud_vlm_models("google")
                    yield Select(
                        [("Google Gemini", "google"), ("OpenAI", "openai")],
                        value="google",
                        id="vlm-provider-select",
                    )
                    yield Select(
                        [(m, m) for m in default_google_models] if default_google_models
                        else [(DEFAULT_ONLINE_VLM_MODEL, DEFAULT_ONLINE_VLM_MODEL)],
                        value=DEFAULT_ONLINE_VLM_MODEL,
                        id="vlm-online-model-select",
                    )
                    yield Button("Refresh", id="refresh-vlm-models")
                yield Label(
                    "Online models: loading defaults. Press Refresh to query provider endpoint.",
                    id="vlm-online-status",
                    classes="field-help",
                )
                yield Label(
                    "Cloud providers need API keys: OPENAI_API_KEY or GOOGLE_API_KEY env vars",
                    classes="field-help",
                )

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
                    yield Button("Save Script", variant="primary", id="save-script-btn")
                    yield Button("Load Script", variant="default", id="load-script-btn")
                    yield Button("View Keyframes", variant="default", id="open-keyframes-btn")
                    yield Button("Cancel", variant="error", id="cancel-edit-btn")

            yield RichLog(highlight=True, markup=True, id="log")
            with Horizontal(id="result-panel"):
                yield Static(id="result-text")
                yield Button("Open", variant="primary", id="open-output-btn")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#mode-section").border_title = "Mode"
        self.query_one("#files-section").border_title = "Files"
        self.query_one("#options-section").border_title = "Options"
        self.query_one("#voice-ref-row").display = False
        self.query_one("#progress-section").border_title = "Progress"
        self.query_one("#progress-section").display = False
        self.query_one("#editor-section").border_title = "Edit Script"
        self.query_one("#editor-section").display = False
        self.query_one("#result-panel").display = False
        self.query_one("#result-panel").border_title = "Result"
        self.query_one("#open-output-btn").display = False
        self._apply_vlm_source_visibility("local")
        self._refresh_online_models(force_refresh=False)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_select_changed(self, event: Select.Changed) -> None:
        try:
            control_id = event.control.id
        except AttributeError:
            return
        if control_id == "tts-model-select":
            is_chatterbox = event.value == "chatterbox"
            self.query_one("#voice-ref-row").display = is_chatterbox
            self.query_one("#voice-row").display = not is_chatterbox
        elif control_id == "vlm-source-select":
            self._apply_vlm_source_visibility(str(event.value))
        elif control_id == "vlm-provider-select":
            self._refresh_online_models(force_refresh=False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "start-btn":
            self._start_voiceover()
        elif bid == "continue-btn":
            self._continue_after_edit()
        elif bid == "cancel-edit-btn":
            self._cancel_event.set()
            self._script_ready.set()
        elif bid == "browse-video":
            self._browse_file("video-input", "Select Video", [".mp4", ".mkv", ".avi", ".mov", ".webm"])
        elif bid == "browse-script":
            self._browse_file("script-input", "Select Script", [".srt", ".json"])
        elif bid == "browse-output":
            self._browse_file("output-input", "Select Output Location", [".mp4", ".wav"])
        elif bid == "browse-voice-ref":
            self._browse_file("voice-ref-input", "Select Voice Reference Audio", [".wav", ".mp3", ".flac", ".ogg"])
        elif bid == "refresh-vlm-models":
            self._refresh_online_models(force_refresh=True)
        elif bid == "save-script-btn":
            self._save_script()
        elif bid == "load-script-btn":
            self._load_script()
        elif bid == "open-keyframes-btn":
            self._open_keyframe_dir()
        elif bid == "open-output-btn":
            self._open_output_file()

    def action_cancel_or_back(self) -> None:
        if self._running:
            if self.query_one("#editor-section").display:
                self.notify(
                    "Script is ready. Use Continue or Cancel buttons in the editor.",
                    severity="warning",
                    timeout=5,
                )
                try:
                    self.query_one("#script-editor", TextArea).focus()
                except Exception:
                    pass
                return
            self._cancel_event.set()
            self._script_ready.set()
            self._log("[yellow]Cancelling...[/yellow]")
        else:
            self.app.pop_screen()

    # ------------------------------------------------------------------
    # File browsing / opening
    # ------------------------------------------------------------------

    def _browse_file(self, input_id: str, title: str, extensions: list[str]) -> None:
        from hftool.tui.widgets.file_browser import FilePickerScreen

        def on_selected(path: str) -> None:
            if path:
                self.query_one(f"#{input_id}", Input).value = path

        self.app.push_screen(FilePickerScreen(title=title, extensions=extensions), on_selected)

    def _open_keyframe_dir(self) -> None:
        if not self._keyframe_dir or not os.path.isdir(self._keyframe_dir):
            self.notify("Keyframes not available", severity="warning")
            return
        self._do_open_path(self._keyframe_dir)

    def _open_output_file(self) -> None:
        if not self._output_path or not os.path.isfile(self._output_path):
            self.notify("Output file not found", severity="warning")
            return
        self._do_open_path(self._output_path)

    def _do_open_path(self, path: str) -> None:
        """Open a path using the platform-appropriate opener."""
        open_path(
            path,
            notify_fn=lambda msg, sev: self.notify(msg, severity=sev, timeout=10),
            log_fn=self._log,
        )

    # ------------------------------------------------------------------
    # Cleanup between runs
    # ------------------------------------------------------------------

    def _cleanup_work_dirs(self, output_path: str) -> None:
        """Remove cached keyframes and segments from a previous run."""
        import shutil

        work_dir = os.path.dirname(os.path.abspath(output_path))

        for subdir in ("voiceover_keyframes", "voiceover_segments"):
            path = os.path.join(work_dir, subdir)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)

        self._keyframe_dir = None

    # ------------------------------------------------------------------
    # Form validation → worker launch
    # ------------------------------------------------------------------

    def _start_voiceover(self) -> None:
        """Validate inputs and launch voiceover in a worker thread."""
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
        vlm_source = str(self.query_one("#vlm-source-select", Select).value)

        if vlm_source == "online":
            provider = str(self.query_one("#vlm-provider-select", Select).value).strip().lower()
            online_model = str(self.query_one("#vlm-online-model-select", Select).value).strip()
            if not online_model:
                self.notify("Select an online model", severity="error")
                return
            vlm_model = f"{provider}/{online_model}"
        else:
            vlm_model = str(self.query_one("#vlm-local-select", Select).value).strip() or "qwen3-vl-8b"

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
        self._output_path = None
        self.query_one("#log", RichLog).clear()
        self.query_one("#result-panel").display = False
        self.query_one("#editor-section").display = False
        self.query_one("#stage-label", Label).update("")
        self.query_one("#progress", ProgressBar).update(total=100, progress=0)

        # Clean up cached data from previous run
        self._cleanup_work_dirs(output_path)

        self.query_one("#start-btn").display = False
        self.query_one("#progress-section").display = True
        self._run_seq += 1
        self._active_run_id = self._run_seq
        run_id = self._active_run_id
        self.query_one("#stage-label", Label).update(f"[bold]Starting run #{run_id}...[/bold]")

        self._running = True
        self._log(f"Starting voiceover run #{run_id} ({mode} mode)...")

        # Scroll progress section into view so user sees feedback immediately
        self.call_after_refresh(self._scroll_progress_visible)

        self._run_voiceover_worker(
            mode, video_path, script_path, output_path,
            tts_model, style, device, voice, voice_ref, capture_interval,
            vlm_model,
        )

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

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
        vlm_model: str = "qwen3-vl-8b",
    ) -> None:
        """Run the voiceover pipeline in a background thread."""
        import logging

        from hftool.tasks.voiceover import VoiceoverTask

        # Suppress noisy loggers and progress bars via env vars.
        # IMPORTANT: Do NOT redirect fd 2 (OS-level stderr) — it is
        # process-wide and kills Textual's terminal driver, freezing
        # the entire TUI.  Use env vars to suppress at the source.
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
        os.environ["MIOPEN_LOG_LEVEL"] = "4"  # errors only
        os.environ["ROCBLAS_LAYER"] = "0"
        os.environ["HSA_TOOLS_LIB"] = ""  # suppress HSA trace
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        for _name in ("httpx", "httpcore", "google", "urllib3",
                       "transformers", "diffusers", "torch"):
            logging.getLogger(_name).setLevel(logging.WARNING)

        try:
            task = VoiceoverTask(
                device=device,
                tts_model=tts_model,
                voice=voice,
                voice_ref=voice_ref or None,
                vlm_model=vlm_model,
                narration_style=style,
                no_edit=True,
            )

            # Route task log/progress through the TUI's RichLog
            task.log_callback = lambda msg: self.app.call_from_thread(self._log, msg)
            task.progress_callback = lambda cur, tot: (
                self.app.call_from_thread(
                    self._update_stage,
                    f"Generating audio — segment {cur}/{tot}",
                ),
                self.app.call_from_thread(self._update_progress, cur, tot),
            )

            pipeline = VoiceoverPipeline(self, run_id=self._active_run_id)

            if mode == "auto":
                pipeline.run_auto(task, video_path, output_path, capture_interval)
            elif mode == "revoice":
                pipeline.run_revoice(task, video_path, output_path)
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
            self.app.call_from_thread(
                self._on_voiceover_done,
                None,
                f"A dependency failed to install (exit code {e.code}). "
                f"Try rebuilding the Docker image.",
            )
        except Exception as e:
            import traceback

            from hftool.utils.errors import HFToolError

            tb = traceback.format_exc()
            self.app.call_from_thread(self._log, f"[red]{tb}[/red]")
            if isinstance(e, HFToolError):
                detail = str(e)
                if e.suggestion:
                    detail = f"{detail}\nSuggestion: {e.suggestion}"
                if e.original_error:
                    detail = f"{detail}\nDetails: {e.original_error}"
                self.app.call_from_thread(self._on_voiceover_done, None, detail)
            else:
                self.app.call_from_thread(self._on_voiceover_done, None, str(e))
        finally:
            try:
                task.cleanup()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # UI update methods (called on the UI thread via call_from_thread)
    # ------------------------------------------------------------------

    def _scroll_progress_visible(self) -> None:
        """Scroll the progress section into view after layout refresh."""
        try:
            self.query_one("#progress-section").scroll_visible(top=True)
        except Exception:
            pass

    def _update_stage(self, text: str) -> None:
        self.query_one("#stage-label", Label).update(f"[bold]{text}[/bold]")

    def _update_progress(self, current: int, total: int) -> None:
        self.query_one("#progress", ProgressBar).update(total=total, progress=current)

    def _set_progress_indeterminate(self) -> None:
        self.query_one("#progress", ProgressBar).update(total=None)

    def _on_voiceover_done(self, output_path: Optional[str], error: Optional[str]) -> None:
        self._running = False
        finished_run_id = self._active_run_id
        self._active_run_id = None
        result_text = self.query_one("#result-text", Static)
        result_panel = self.query_one("#result-panel")
        self._hide_editor()

        if error:
            self.query_one("#stage-label", Label).update("[red]Failed[/red]")
            if finished_run_id is not None:
                self._log(f"[red]Run #{finished_run_id} error: {error}[/red]")
            else:
                self._log(f"[red]Error: {error}[/red]")
            result_text.update(f"[red]{error}[/red]")
            result_panel.add_class("-error")
            result_panel.border_title = "Error"
            self.query_one("#open-output-btn").display = False
        else:
            self.query_one("#stage-label", Label).update("[green]Complete![/green]")
            self.query_one("#progress", ProgressBar).update(total=100, progress=100)
            size_str = ""
            if output_path and os.path.exists(output_path):
                size = os.path.getsize(output_path)
                size_str = f" ({size / (1024*1024):.1f} MB)" if size > 1024 * 1024 else f" ({size / 1024:.1f} KB)"
            self._output_path = output_path
            display_path = output_path
            if os.environ.get("HFTOOL_IN_DOCKER") and output_path:
                from hftool.tui.screens.voiceover.path_utils import container_to_host_path

                display_path = container_to_host_path(output_path)
            result_text.update(f"[bold green]Output:[/bold green] {display_path}{size_str}")
            self.query_one("#open-output-btn").display = True
            if finished_run_id is not None:
                self._log(f"[green]Run #{finished_run_id} complete: {display_path}[/green]")
            else:
                self._log(f"[green]Voiceover complete: {display_path}[/green]")

        result_panel.display = True
        self.query_one("#start-btn").display = True
        self.query_one("#start-btn").label = "Start New Voiceover"

    def _log(self, text: str) -> None:
        try:
            self.query_one("#log", RichLog).write(text)
        except Exception:
            pass
