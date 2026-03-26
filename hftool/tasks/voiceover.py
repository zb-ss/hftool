"""Voiceover pipeline task.

Orchestrates script parsing, TTS generation per segment, audio merging,
and optional video muxing. Supports three entry points:

  A. Auto-voiceover: video → scene detection → VLM analysis → script → TTS → merge
  B. Re-voice: video → extract audio → ASR → script → TTS → merge
  C. Manual script: script file → TTS → merge (original, unchanged)

Supports Kokoro and Chatterbox TTS models.
"""

import os
import subprocess
import time
from typing import Any, Dict, List, Optional

from hftool.io.script_parser import ScriptData, ScriptSegment, parse_script
from hftool.io.audio_mixer import SegmentAudio, merge_segments, merge_with_video


class VoiceoverTask:
    """Voiceover pipeline orchestrator.

    Entry points:
    - run()        → Entry Point C (manual script → TTS → merge)
    - run_auto()   → Entry Point A (video → VLM → script → TTS → merge)
    - run_revoice() → Entry Point B (video → ASR → script → TTS → merge)
    """

    def __init__(
        self,
        device: str = "auto",
        dtype: Optional[str] = None,
        tts_model: str = "kokoro",
        voice_ref: Optional[str] = None,
        exaggeration: float = 0.4,
        segments_dir: Optional[str] = None,
        vlm_model: str = "qwen3-vl-8b",
        narration_style: str = "tutorial",
        scene_threshold: float = 3.0,
        no_edit: bool = False,
        save_script: Optional[str] = None,
    ):
        self.device = device
        self.dtype = dtype
        self.tts_model = tts_model
        self.voice_ref = voice_ref
        self.exaggeration = exaggeration
        self.segments_dir = segments_dir
        self.vlm_model = vlm_model
        self.narration_style = narration_style
        self.scene_threshold = scene_threshold
        self.no_edit = no_edit
        self.save_script = save_script
        self._tts_task = None
        self._vlm_task = None
        self._console = None

    def _get_console(self):
        """Get or create a Rich console for progress reporting."""
        if self._console is None:
            try:
                from rich.console import Console
                self._console = Console()
            except ImportError:
                self._console = None
        return self._console

    def _log(self, message: str) -> None:
        """Print a status message."""
        console = self._get_console()
        if console:
            console.print(message)
        else:
            print(message)

    def _resolve_tts_model(self, tts_model: str) -> str:
        """Resolve TTS model short name to repo_id."""
        from hftool.core.models import MODEL_REGISTRY

        tts_models = MODEL_REGISTRY.get("text-to-speech", {})

        # Direct match on short name
        if tts_model in tts_models:
            return tts_models[tts_model].repo_id

        # Check if it's already a repo_id
        for info in tts_models.values():
            if info.repo_id == tts_model:
                return tts_model

        # Default to what was given (let TTS task handle errors)
        return tts_model

    def _load_tts(self) -> None:
        """Load the TTS task and model, downloading if needed."""
        from hftool.tasks.text_to_speech import TextToSpeechTask
        from hftool.core.download import ensure_model_available, install_pip_dependencies
        from hftool.core.models import MODEL_REGISTRY

        self._tts_task = TextToSpeechTask(device=self.device, dtype=self.dtype)

        model_id = self._resolve_tts_model(self.tts_model)

        # Auto-install pip dependencies and download model if needed
        tts_models = MODEL_REGISTRY.get("text-to-speech", {})
        model_info = tts_models.get(self.tts_model)
        if model_info:
            # Always ensure pip deps are installed (e.g. kokoro pip package)
            if model_info.pip_dependencies:
                install_pip_dependencies(model_info.pip_dependencies)
            # Download HF model weights if not cached
            if not os.path.exists(model_id):
                self._log(f"  Downloading TTS model: {model_info.name} ({model_info.size_str})...")
                ensure_model_available(
                    repo_id=model_id,
                    size_gb=model_info.size_gb,
                    task_name="text-to-speech",
                    model_name=model_info.name,
                )

        load_kwargs = {}
        if self.voice_ref:
            load_kwargs["voice_ref"] = self.voice_ref

        self._log(f"  Loading TTS model: {self.tts_model}")
        self._tts_task.load_pipeline(model_id, **load_kwargs)
        self._tts_task._pipeline = self._tts_task._pipeline  # already set by load_pipeline

    def _get_segments_dir(self, output_path: str) -> str:
        """Determine segments directory."""
        if self.segments_dir:
            return self.segments_dir

        # Default: alongside output file
        output_dir = os.path.dirname(os.path.abspath(output_path))
        return os.path.join(output_dir, "voiceover_segments")

    def _generate_segment(
        self,
        segment: ScriptSegment,
        seg_path: str,
        index: int,
        total: int,
    ) -> float:
        """Generate TTS audio for a single segment.

        Returns:
            Duration of generated audio in seconds
        """
        self._log(
            f"  [{index}/{total}] Seg {segment.id} @ {segment.start_ms / 1000:.0f}s"
        )

        t0 = time.time()

        infer_kwargs = {}
        if self.voice_ref:
            infer_kwargs["voice_ref"] = self.voice_ref
        if "chatterbox" in self.tts_model.lower():
            infer_kwargs["exaggeration"] = self.exaggeration

        result = self._tts_task.run_inference(
            self._tts_task._pipeline, segment.text, **infer_kwargs
        )

        self._tts_task.save_output(result, seg_path)

        elapsed = time.time() - t0
        sample_rate = result.get("sampling_rate", 24000)
        audio = result.get("audio")

        if hasattr(audio, "shape"):
            # numpy array or torch tensor
            n_samples = audio.shape[-1] if len(audio.shape) > 1 else audio.shape[0]
            duration = n_samples / sample_rate
        else:
            duration = 0.0

        self._log(
            f"           {duration:.1f}s audio in {elapsed:.1f}s"
        )

        return duration

    def _resolve_vlm_model(self, vlm_model: str) -> str:
        """Resolve VLM model short name to repo_id."""
        from hftool.core.models import MODEL_REGISTRY

        vlm_models = MODEL_REGISTRY.get("vision-language", {})

        if vlm_model in vlm_models:
            return vlm_models[vlm_model].repo_id

        for info in vlm_models.values():
            if info.repo_id == vlm_model:
                return vlm_model

        return vlm_model

    def _load_vlm(self) -> None:
        """Load the VLM task and model, downloading if needed."""
        from hftool.tasks.vision_language import VisionLanguageTask
        from hftool.core.download import ensure_model_available, install_pip_dependencies
        from hftool.core.models import MODEL_REGISTRY

        self._vlm_task = VisionLanguageTask(device=self.device, dtype=self.dtype)
        model_id = self._resolve_vlm_model(self.vlm_model)

        # Auto-install pip dependencies and download model if needed
        vlm_models = MODEL_REGISTRY.get("vision-language", {})
        model_info = vlm_models.get(self.vlm_model)
        if model_info:
            if model_info.pip_dependencies:
                install_pip_dependencies(model_info.pip_dependencies)
            if not os.path.exists(model_id):
                self._log(f"  Downloading VLM model: {model_info.name} ({model_info.size_str})...")
                ensure_model_available(
                    repo_id=model_id,
                    size_gb=model_info.size_gb,
                    task_name="vision-language",
                    model_name=model_info.name,
                )

        self._log(f"  Loading VLM model: {self.vlm_model}")
        self._vlm_task.load_pipeline(model_id)

    def _unload_vlm(self) -> None:
        """Unload VLM to free VRAM before loading TTS."""
        if self._vlm_task:
            self._log("  Unloading VLM to free VRAM...")
            self._vlm_task.cleanup()
            self._vlm_task = None

    def _extract_audio(self, video_path: str, output_path: str) -> str:
        """Extract audio track from a video file as WAV.

        Args:
            video_path: Path to input video
            output_path: Path for extracted WAV file

        Returns:
            Path to the extracted audio file

        Raises:
            HFToolError: If FFmpeg fails
        """
        from hftool.utils.errors import HFToolError
        from hftool.utils.deps import check_ffmpeg

        check_ffmpeg()

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            stderr_tail = result.stderr[-500:] if result.stderr else "No error output"
            raise HFToolError(
                f"FFmpeg audio extraction failed: {stderr_tail}",
                suggestion="Check that the video file contains an audio track.",
            )

        return output_path

    def _transcribe_to_script(self, audio_path: str) -> ScriptData:
        """Transcribe audio to a ScriptData with timed segments.

        Uses the ASR (Whisper) task to get timestamped chunks, then converts
        them into ScriptSegment objects.

        Args:
            audio_path: Path to WAV audio file

        Returns:
            ScriptData with transcribed segments
        """
        from hftool.tasks.speech_to_text import SpeechToTextTask

        stt_task = SpeechToTextTask(device=self.device, dtype=self.dtype)
        stt_task.load_pipeline("openai/whisper-large-v3")

        result = stt_task.run_inference(
            stt_task._pipeline,
            audio_path,
            return_timestamps=True,
        )

        stt_task.cleanup()

        chunks = result.get("chunks", [])
        segments: List[ScriptSegment] = []

        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "").strip()
            if not text:
                continue

            timestamp = chunk.get("timestamp", (0.0, 0.0))
            start_s, end_s = timestamp
            if start_s is None:
                start_s = 0.0
            if end_s is None:
                end_s = start_s + 5.0

            segments.append(ScriptSegment(
                id=i + 1,
                start_ms=int(start_s * 1000),
                end_ms=int(end_s * 1000),
                text=text,
            ))

        return ScriptData(
            segments=segments,
            metadata={"source_format": "asr", "source_file": audio_path},
        )

    def _generate_and_merge(
        self,
        script: ScriptData,
        output_path: str,
        video_path: Optional[str] = None,
        keep_audio: bool = False,
    ) -> str:
        """Run TTS generation, audio merge, and video merge from a ScriptData.

        This is the shared backend for all three entry points (A, B, C).

        Args:
            script: Parsed script with segments
            output_path: Path for final output
            video_path: Optional video to merge with
            keep_audio: If True, duck original audio instead of stripping

        Returns:
            Path to the output file
        """
        from hftool.utils.deps import check_ffmpeg

        check_ffmpeg()

        # Load TTS model
        self._log("[bold]Loading TTS model...[/bold]")
        self._load_tts()

        # Generate TTS segments
        self._log("[bold]Generating TTS audio...[/bold]")
        seg_dir = self._get_segments_dir(output_path)
        os.makedirs(seg_dir, exist_ok=True)

        segment_audios: List[SegmentAudio] = []
        total = len(script.segments)
        t_start = time.time()

        for i, seg in enumerate(script.segments):
            seg_path = os.path.join(seg_dir, f"seg_{seg.id:03d}.wav")

            if os.path.exists(seg_path):
                self._log(f"  [{i + 1}/{total}] Seg {seg.id} exists, skipping")
            else:
                self._generate_segment(seg, seg_path, i + 1, total)

            segment_audios.append(SegmentAudio(
                path=seg_path,
                start_ms=seg.start_ms,
                segment_id=seg.id,
            ))

        gen_time = time.time() - t_start
        self._log(f"  TTS generation completed in {gen_time:.1f}s")

        # Merge audio segments
        self._log("[bold]Merging audio segments...[/bold]")
        merged_audio_path = os.path.join(seg_dir, "merged_narration.wav")
        merge_segments(segment_audios, merged_audio_path)
        self._log(f"  Merged audio: {merged_audio_path}")

        # Merge with video or copy audio output
        if video_path:
            self._log("[bold]Merging with video...[/bold]")
            merge_with_video(video_path, merged_audio_path, output_path, keep_original=keep_audio)
        else:
            import shutil
            shutil.copy2(merged_audio_path, output_path)

        size_mb = os.path.getsize(output_path) / 1024 / 1024
        total_time = time.time() - t_start
        self._log(f"\n[bold green]Done![/bold green] {output_path} ({size_mb:.1f} MB) in {total_time / 60:.1f} min")

        return output_path

    def run(
        self,
        script_path: str,
        output_path: str,
        video_path: Optional[str] = None,
        keep_audio: bool = False,
    ) -> str:
        """Entry Point C: Run voiceover from a manual script file.

        Args:
            script_path: Path to SRT or JSON script file
            output_path: Path for final output (WAV if audio-only, MP4 if video)
            video_path: Optional path to input video (for video+audio output)
            keep_audio: If True, duck original video audio instead of stripping

        Returns:
            Path to the output file
        """
        self._log("[bold]Step 1:[/bold] Parsing script...")
        script = parse_script(script_path)
        self._log(f"  Loaded {len(script.segments)} segments ({script.total_duration_s:.0f}s total)")

        return self._generate_and_merge(script, output_path, video_path, keep_audio)

    def run_auto(
        self,
        video_path: str,
        output_path: str,
        keep_audio: bool = False,
    ) -> str:
        """Entry Point A: Auto-generate voiceover from video.

        Pipeline: scene detection → keyframe extraction → VLM frame analysis
        → script generation → optional review → TTS → audio merge → video merge.

        VRAM management: VLM is loaded, used, and unloaded before TTS is loaded.

        Args:
            video_path: Path to input video file
            output_path: Path for final output
            keep_audio: If True, duck original video audio

        Returns:
            Path to the output file
        """
        from hftool.utils.errors import HFToolError
        from hftool.utils.deps import check_ffmpeg
        from hftool.io.scene_detector import detect_scenes, extract_keyframes
        from hftool.io.script_generator import analyze_frames, generate_script
        from hftool.io.script_review import review_script

        check_ffmpeg()

        if not os.path.exists(video_path):
            raise HFToolError(
                f"Video file not found: {video_path}",
                suggestion="Check the file path and try again.",
            )

        work_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(work_dir, exist_ok=True)

        # Step 1: Scene detection
        self._log("[bold]Step 1:[/bold] Detecting scenes...")
        scenes = detect_scenes(video_path, threshold=self.scene_threshold)
        self._log(f"  Found {len(scenes.scenes)} scenes in {scenes.video_duration_ms / 1000:.0f}s video")

        # Step 2: Keyframe extraction
        self._log("[bold]Step 2:[/bold] Extracting keyframes...")
        keyframe_dir = os.path.join(work_dir, "voiceover_keyframes")
        scenes = extract_keyframes(video_path, scenes, keyframe_dir)
        total_frames = sum(len(s.keyframe_paths) for s in scenes.scenes)
        self._log(f"  Extracted {total_frames} keyframes")

        # Step 3: VLM frame analysis + script generation
        self._log("[bold]Step 3:[/bold] Analyzing frames with VLM...")
        self._load_vlm()

        analyses = analyze_frames(self._vlm_task, scenes)
        self._log(f"  Analyzed {len(analyses)} frames")

        self._log("[bold]Step 4:[/bold] Generating narration script...")
        script = generate_script(
            self._vlm_task,
            analyses,
            scenes,
            style=self.narration_style,
            video_duration_ms=scenes.video_duration_ms,
        )
        self._log(f"  Generated {len(script.segments)} narration segments")

        # Unload VLM before TTS (critical for VRAM on single-GPU systems)
        self._unload_vlm()

        # Step 4: Save script if requested
        if self.save_script:
            from hftool.io.script_review import _write_json
            _write_json(script, self.save_script)
            self._log(f"  Script saved to: {self.save_script}")

        # Step 5: Review script
        self._log("[bold]Step 5:[/bold] Reviewing script...")
        script = review_script(
            script,
            work_dir=work_dir,
            no_edit=self.no_edit,
            save_path=self.save_script,
        )

        # Step 6-8: TTS + merge (shared with other entry points)
        self._log("[bold]Step 6:[/bold] Generating voiceover...")
        return self._generate_and_merge(script, output_path, video_path, keep_audio)

    def run_revoice(
        self,
        video_path: str,
        output_path: str,
        keep_audio: bool = False,
    ) -> str:
        """Entry Point B: Re-voice existing narration.

        Pipeline: extract audio → ASR transcription → optional review → TTS
        → audio merge → video merge.

        Args:
            video_path: Path to input video with existing narration
            output_path: Path for final output
            keep_audio: If True, duck original audio instead of replacing

        Returns:
            Path to the output file
        """
        from hftool.utils.errors import HFToolError
        from hftool.utils.deps import check_ffmpeg
        from hftool.io.script_review import review_script

        check_ffmpeg()

        if not os.path.exists(video_path):
            raise HFToolError(
                f"Video file not found: {video_path}",
                suggestion="Check the file path and try again.",
            )

        work_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(work_dir, exist_ok=True)

        # Step 1: Extract audio
        self._log("[bold]Step 1:[/bold] Extracting audio from video...")
        audio_path = os.path.join(work_dir, "voiceover_extracted_audio.wav")
        self._extract_audio(video_path, audio_path)
        self._log(f"  Extracted audio: {audio_path}")

        # Step 2: Transcribe with Whisper
        self._log("[bold]Step 2:[/bold] Transcribing audio with ASR...")
        script = self._transcribe_to_script(audio_path)
        self._log(f"  Transcribed {len(script.segments)} segments")

        # Step 3: Review script
        self._log("[bold]Step 3:[/bold] Reviewing transcript...")
        if self.save_script:
            from hftool.io.script_review import _write_json
            _write_json(script, self.save_script)
            self._log(f"  Script saved to: {self.save_script}")

        script = review_script(
            script,
            work_dir=work_dir,
            no_edit=self.no_edit,
            save_path=self.save_script,
        )

        # Step 4-6: TTS + merge
        self._log("[bold]Step 4:[/bold] Generating voiceover...")
        return self._generate_and_merge(script, output_path, video_path, keep_audio)

    def cleanup(self) -> None:
        """Clean up all loaded model resources."""
        if self._tts_task:
            self._tts_task.cleanup()
            self._tts_task = None
        if self._vlm_task:
            self._vlm_task.cleanup()
            self._vlm_task = None


def create_task(device: str = "auto", dtype: Optional[str] = None) -> VoiceoverTask:
    """Factory function to create a VoiceoverTask.

    Args:
        device: Device to run on ("auto", "cuda", "mps", "cpu")
        dtype: Data type ("bfloat16", "float16", "float32")

    Returns:
        Configured VoiceoverTask instance
    """
    return VoiceoverTask(device=device, dtype=dtype)
