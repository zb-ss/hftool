"""Voiceover pipeline orchestration for the TUI.

Runs the multi-step auto/revoice/script pipelines in a worker thread,
communicating progress to the UI thread via ``app.call_from_thread``.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import threading
import time
from typing import Any, Callable, Optional


class VoiceoverPipeline:
    """Orchestrates the voiceover pipeline steps in a background thread.

    All UI updates go through thread-safe wrappers that call
    ``app.call_from_thread`` on the owning Screen.

    Args:
        screen: The ``VoiceoverScreen`` instance (provides UI callbacks
                and shared state like ``_cancel_event``, ``_script_ready``).
        run_id: Numeric run identifier for debug logs.
    """

    def __init__(self, screen: Any, run_id: Optional[int] = None) -> None:
        self._ui = screen
        self._run_id = run_id

    # ------------------------------------------------------------------
    # Thread-safe UI wrappers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        self._ui.app.call_from_thread(self._ui._log, msg)

    def _update_stage(self, text: str) -> None:
        self._ui.app.call_from_thread(self._ui._update_stage, text)

    def _update_progress(self, current: int, total: int) -> None:
        self._ui.app.call_from_thread(self._ui._update_progress, current, total)

    def _set_progress_indeterminate(self) -> None:
        self._ui.app.call_from_thread(self._ui._set_progress_indeterminate)

    def _show_editor(self, script: Any) -> None:
        self._ui.app.call_from_thread(self._ui._show_script_editor, script)

    def _hide_editor(self) -> None:
        self._ui.app.call_from_thread(self._ui._hide_editor)

    @property
    def _is_cancelled(self) -> bool:
        return self._ui._cancel_event.is_set()

    # ------------------------------------------------------------------
    # Debug logging
    # ------------------------------------------------------------------

    def _debug_log(self, msg: str) -> None:
        """Write a timestamped message to the voiceover debug log file."""
        try:
            log_path = os.path.join(
                os.environ.get("HFTOOL_CONFIG", "/tmp"),
                "voiceover_debug.log",
            )
            run_prefix = f"run#{self._run_id} " if self._run_id is not None else ""
            with open(log_path, "a") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] {run_prefix}{msg}\n")
                f.flush()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Heartbeat wrapper
    # ------------------------------------------------------------------

    def _call_with_heartbeat(
        self,
        fn: Callable,
        wait_label: str,
        heartbeat_s: int = 15,
        timeout_s: Optional[int] = None,
    ) -> Any:
        """Run a blocking function and emit periodic progress logs.

        Spawns *fn* in a daemon thread and polls for completion, emitting
        a heartbeat log every *heartbeat_s* seconds so the user knows
        the pipeline hasn't stalled.

        Args:
            fn: Callable to execute.
            wait_label: Human-readable label for log messages.
            heartbeat_s: Seconds between heartbeat logs.
            timeout_s: Optional hard timeout (raises ``TimeoutError``).

        Returns:
            The return value of *fn*.
        """
        state: dict = {"done": False, "result": None, "error": None}

        def _runner() -> None:
            try:
                state["result"] = fn()
            except Exception as exc:
                state["error"] = exc
            finally:
                state["done"] = True

        worker = threading.Thread(target=_runner, daemon=True)
        worker.start()

        start = time.monotonic()
        next_heartbeat = min(3, heartbeat_s)

        while not state["done"]:
            worker.join(timeout=0.5)
            if state["done"]:
                break
            elapsed = int(time.monotonic() - start)
            if timeout_s is not None and elapsed >= timeout_s:
                state["error"] = TimeoutError(
                    f"{wait_label} exceeded timeout ({timeout_s}s)",
                )
                break
            if elapsed >= next_heartbeat:
                self._debug_log(f"heartbeat {wait_label} {elapsed}s")
                self._log(f"  {wait_label}... {elapsed}s elapsed")
                self._update_stage(f"{wait_label} ({elapsed}s)")
                next_heartbeat += heartbeat_s

        if state["error"] is not None:
            raise state["error"]

        return state["result"]

    # ------------------------------------------------------------------
    # Shared pipeline helpers
    # ------------------------------------------------------------------

    def _wait_for_script_review(self, step_label: str) -> bool:
        """Block until the user finishes editing the script.

        Returns False if cancelled, True otherwise.
        """
        reminder_after_s = 20
        reminder_every_s = 20
        started_waiting = time.monotonic()
        next_reminder = reminder_after_s

        while not self._ui._script_ready.wait(timeout=0.5):
            if self._is_cancelled:
                return False
            elapsed = int(time.monotonic() - started_waiting)
            if elapsed >= next_reminder:
                self._update_stage(f"{step_label} ({elapsed}s)")
                self._log(
                    "  Waiting for your review. Edit script in the editor "
                    "and click Continue.",
                )
                next_reminder += reminder_every_s

        return not self._is_cancelled

    def _apply_edited_script(self, script: Any) -> Any:
        """Parse the user's edited script JSON, falling back to *script*."""
        edited = self._ui._edited_script
        if not edited:
            self._log("  [dim]No edits detected, using generated script[/dim]")
            return script

        self._log(f"  Applying edited script ({len(edited)} chars)...")
        try:
            from hftool.io.script_parser import ScriptData, parse_editor_segments

            raw = json.loads(edited)
            segments = parse_editor_segments(raw)
            result = ScriptData(segments=segments)
            self._log(f"  Parsed {len(result.segments)} edited segments")
            return result
        except Exception as e:
            self._log(f"[yellow]Script parse error, using original: {e}[/yellow]")
            return script

    def _finalize_and_merge(
        self,
        task: Any,
        script: Any,
        output_path: str,
        video_path: str,
        step_label: str,
    ) -> None:
        """Run TTS generation + audio/video merge (shared by all modes)."""
        self._update_stage(step_label)
        self._hide_editor()
        self._set_progress_indeterminate()
        self._log(
            f"  Voice: {task.voice}, TTS: {task.tts_model}, "
            f"Segments: {len(script.segments)}",
        )

        # Remove old segment files so voice/script changes take effect
        seg_dir = os.path.join(
            os.path.dirname(os.path.abspath(output_path)),
            "voiceover_segments",
        )
        if os.path.isdir(seg_dir):
            shutil.rmtree(seg_dir, ignore_errors=True)

        task._generate_and_merge(script, output_path, video_path, keep_audio=False)

    def _free_vram(self) -> None:
        """Aggressively free GPU VRAM (KV cache fragments, etc.)."""
        try:
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Auto pipeline (video → VLM → script → TTS → merge)
    # ------------------------------------------------------------------

    def run_auto(
        self,
        task: Any,
        video_path: str,
        output_path: str,
        capture_interval: str = "auto",
    ) -> None:
        """Run the full auto-voiceover pipeline with TUI script editing."""
        from hftool.io.scene_detector import extract_keyframes
        from hftool.utils.deps import check_ffmpeg

        self._debug_log(f"auto_start vlm={task.vlm_model}")
        self._update_stage("Preparing auto voiceover pipeline...")
        self._set_progress_indeterminate()
        check_ffmpeg()

        work_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(work_dir, exist_ok=True)

        # Step 1: Scene detection (or fixed interval)
        scenes = self._detect_scenes(task, video_path, capture_interval)
        self._log(f"  Found {len(scenes.scenes)} scenes")
        if self._is_cancelled:
            return

        # Step 2: Keyframe extraction
        self._update_stage("Step 2/7 — Extracting keyframes...")
        keyframe_dir = os.path.join(work_dir, "voiceover_keyframes")
        self._ui._keyframe_dir = keyframe_dir
        scenes = extract_keyframes(video_path, scenes, keyframe_dir)
        total_frames = sum(len(s.keyframe_paths) for s in scenes.scenes)
        self._log(f"  Extracted {total_frames} keyframes")
        if self._is_cancelled:
            return

        # Step 3: VLM frame analysis
        analyses = self._analyze_frames(task, scenes)
        if self._is_cancelled:
            task._unload_vlm()
            return

        # Step 4: Scene grouping
        scenes, analyses = self._group_scenes(task, scenes, analyses)
        if self._is_cancelled:
            task._unload_vlm()
            return

        # Step 5: Script generation
        script = self._generate_script(task, scenes, analyses)
        task._unload_vlm()
        if self._is_cancelled:
            return

        # Step 6: Script review
        self._show_editor(script)
        if not self._wait_for_script_review("Step 6/7 — Waiting for script review"):
            return
        script = self._apply_edited_script(script)
        if self._is_cancelled:
            return

        # Step 7: TTS + merge
        self._finalize_and_merge(
            task, script, output_path, video_path,
            "Step 7/7 — Generating voiceover audio...",
        )

    # ------------------------------------------------------------------
    # Auto pipeline sub-steps
    # ------------------------------------------------------------------

    def _detect_scenes(self, task: Any, video_path: str, capture_interval: str):
        """Step 1: Scene detection or fixed-interval capture."""
        self._debug_log(f"scene_detect capture_interval={capture_interval}")

        if capture_interval != "auto":
            interval_s = float(capture_interval)
            self._update_stage(f"Step 1/7 — Capturing every {interval_s:.0f}s...")
            self._set_progress_indeterminate()

            from hftool.io.scene_detector import (
                SceneDetectionResult,
                SceneInfo,
                _fixed_interval_scenes,
                get_video_duration_ms,
            )

            duration_ms = get_video_duration_ms(video_path)
            intervals = _fixed_interval_scenes(duration_ms, interval_s=interval_s)
            return SceneDetectionResult(
                scenes=[
                    SceneInfo(index=i, start_ms=s, end_ms=e)
                    for i, (s, e) in enumerate(intervals)
                ],
                video_duration_ms=duration_ms,
                video_path=video_path,
                keyframe_dir="",
            )

        self._update_stage("Step 1/7 — Detecting scenes...")
        self._set_progress_indeterminate()

        from hftool.io.scene_detector import detect_scenes

        return detect_scenes(video_path, threshold=task.scene_threshold)

    def _analyze_frames(self, task: Any, scenes: Any) -> list:
        """Step 3: Load VLM and analyze each keyframe."""
        from hftool.io.script_generator import FRAME_ANALYSIS_PROMPT, FrameAnalysis
        from hftool.io.vlm_providers import parse_vlm_model

        total_frames = sum(len(s.keyframe_paths) for s in scenes.scenes)
        self._update_stage(f"Step 3/7 — Loading VLM ({task.vlm_model})...")
        self._set_progress_indeterminate()
        self._log(f"  VLM: {task.vlm_model}")

        # Detect cloud provider for informational logging
        try:
            prefix, _ = parse_vlm_model(task.vlm_model)
            if prefix:
                self._log(
                    f"  [dim]Cloud VLM provider detected ({prefix}); "
                    f"checking SDK/API access...[/dim]",
                )
        except Exception:
            pass

        self._debug_log("loading_vlm")
        self._call_with_heartbeat(
            task._load_vlm,
            f"Loading VLM ({task.vlm_model})",
            heartbeat_s=10,
        )
        self._debug_log(f"vlm_loaded type={type(task._vlm_task).__name__}")

        self._update_stage(
            f"Step 3/7 — Analyzing frames with VLM (0/{total_frames})...",
        )
        self._update_progress(0, total_frames)

        # Per-frame analysis with progress updates
        analyses: list[FrameAnalysis] = []
        prev_description = ""
        frame_idx = 0

        for scene in scenes.scenes:
            for image_path in scene.keyframe_paths:
                if self._is_cancelled:
                    task._unload_vlm()
                    return analyses

                prompt = FRAME_ANALYSIS_PROMPT.format(
                    previous_description=prev_description or "None yet.",
                )
                self._log(
                    f"  [dim]Analyzing scene {scene.index}, "
                    f"{os.path.basename(image_path)}...[/dim]",
                )

                description = self._call_with_heartbeat(
                    lambda: task._vlm_task.analyze_frame(
                        image_path, prompt, previous_context=prev_description,
                    ),
                    f"Analyzing frame (scene {scene.index})",
                    heartbeat_s=10,
                )

                analyses.append(FrameAnalysis(
                    scene_index=scene.index,
                    timestamp_ms=scene.start_ms,
                    image_path=image_path,
                    description=description,
                ))
                prev_description = description
                frame_idx += 1

                self._update_stage(
                    f"Step 3/7 — Analyzing frames with VLM "
                    f"({frame_idx}/{total_frames})...",
                )
                self._update_progress(frame_idx, total_frames)
                self._free_vram()

        self._log(f"  Analyzed {len(analyses)} frames")
        self._free_vram()
        return analyses

    def _group_scenes(self, task: Any, scenes: Any, analyses: list) -> tuple:
        """Step 4: Use VLM to group adjacent scenes by activity."""
        from hftool.io.script_generator import group_scenes

        if len(scenes.scenes) <= 2:
            return scenes, analyses

        self._debug_log("grouping_start")
        self._update_stage("Step 4/7 — Grouping scenes by activity...")
        self._set_progress_indeterminate()

        scenes, analyses = self._call_with_heartbeat(
            lambda: group_scenes(task._vlm_task, analyses, scenes),
            "Grouping scenes by activity",
            heartbeat_s=10,
        )
        self._log(f"  Grouped into {len(scenes.scenes)} logical segments")
        return scenes, analyses

    def _generate_script(self, task: Any, scenes: Any, analyses: list) -> Any:
        """Step 5: Generate narration script from frame analyses."""
        from hftool.io.script_generator import generate_script

        self._debug_log("script_gen_start")
        self._update_stage("Step 5/7 — Generating script...")
        self._log("  [dim]Script generation may take a minute...[/dim]")

        timeout_s = int(os.environ.get("HFTOOL_SCRIPT_GEN_TIMEOUT_S", "240"))

        try:
            script = self._call_with_heartbeat(
                lambda: generate_script(
                    task._vlm_task,
                    analyses,
                    scenes,
                    style=task.narration_style,
                    video_duration_ms=scenes.video_duration_ms,
                ),
                "Generating narration script",
                heartbeat_s=10,
                timeout_s=timeout_s,
            )
        except TimeoutError:
            from hftool.io.script_generator import _fallback_script

            self._debug_log("script_gen_timeout_fallback")
            self._log(
                f"[yellow]Script synthesis timed out after {timeout_s}s; "
                f"using fallback script.[/yellow]",
            )
            script = _fallback_script(analyses, scenes.video_duration_ms)

        self._log(f"  Generated {len(script.segments)} segments")
        return script

    # ------------------------------------------------------------------
    # Re-voice pipeline (video → ASR → script → TTS → merge)
    # ------------------------------------------------------------------

    def run_revoice(
        self,
        task: Any,
        video_path: str,
        output_path: str,
    ) -> None:
        """Run the re-voice pipeline with TUI script editing."""
        from hftool.utils.deps import check_ffmpeg

        check_ffmpeg()

        work_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(work_dir, exist_ok=True)

        # Step 1: Extract audio
        self._update_stage("Step 1/4 — Extracting audio...")
        audio_path = os.path.join(work_dir, "extracted_audio.wav")
        task._extract_audio(video_path, audio_path)
        if self._is_cancelled:
            return

        # Step 2: ASR transcription
        self._update_stage("Step 2/4 — Transcribing audio...")
        self._set_progress_indeterminate()
        script = self._call_with_heartbeat(
            lambda: task._transcribe_to_script(audio_path),
            "Transcribing audio",
            heartbeat_s=10,
        )
        self._log(f"  Transcribed {len(script.segments)} segments")
        if self._is_cancelled:
            return

        # Step 3: Script review
        self._show_editor(script)
        if not self._wait_for_script_review("Step 3/4 — Waiting for script review"):
            return
        script = self._apply_edited_script(script)
        if self._is_cancelled:
            return

        # Step 4: TTS + merge
        self._finalize_and_merge(
            task, script, output_path, video_path,
            "Step 4/4 — Generating voiceover audio...",
        )
