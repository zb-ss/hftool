"""Voiceover command — generate voiceover from video or script."""

import sys
from typing import Optional

import click


@click.command("voiceover")
@click.option("--script", "-s", default=None, help="Path to SRT or JSON script file (Entry Point C)")
@click.option("--auto", "mode_auto", is_flag=True, help="Auto-generate voiceover from video (Entry Point A)")
@click.option("--revoice", "mode_revoice", is_flag=True, help="Re-voice existing narration (Entry Point B)")
@click.option("--video", "-v", default=None, help="Path to input video")
@click.option("--output", "-o", required=True, help="Output file path (.mp4 with --video, .wav without)")
@click.option("--tts-model", default="kokoro", help="TTS model: kokoro (default), chatterbox")
@click.option("--keep-audio", is_flag=True, help="Duck original video audio instead of stripping")
@click.option("--segments-dir", default=None, help="Directory to store/find segment WAV files")
@click.option("--voice-ref", default=None, help="Reference audio for voice cloning (Chatterbox only)")
@click.option("--exaggeration", type=float, default=0.4, help="Emotion control for Chatterbox (default: 0.4)")
@click.option("--vlm-model", default="qwen3-vl-8b", help="VLM for frame analysis (default: qwen3-vl-8b)")
@click.option("--style", type=click.Choice(["tutorial", "presentation", "demo", "casual", "formal"]), default="tutorial", help="Narration style for auto mode")
@click.option("--scene-threshold", type=float, default=3.0, help="Scene detection sensitivity (default: 3.0)")
@click.option("--no-edit", is_flag=True, help="Skip editor review of generated script")
@click.option("--save-script", default=None, help="Save generated script to file path")
@click.option("--device", "-d", default="auto", help="Device to use (auto, cuda, mps, cpu)")
@click.option("--dtype", default=None, help="Data type (bfloat16, float16, float32)")
@click.pass_context
def voiceover_command(
    ctx: click.Context,
    script: Optional[str],
    mode_auto: bool,
    mode_revoice: bool,
    video: Optional[str],
    output: str,
    tts_model: str,
    keep_audio: bool,
    segments_dir: Optional[str],
    voice_ref: Optional[str],
    exaggeration: float,
    vlm_model: str,
    style: str,
    scene_threshold: float,
    no_edit: bool,
    save_script: Optional[str],
    device: str,
    dtype: Optional[str],
):
    """Generate voiceover from video or script.

    Three modes of operation:

    \b
    --auto     Auto-generate voiceover: video → VLM analysis → TTS → merge
    --revoice  Re-voice existing narration: video → ASR → TTS → merge
    --script   Manual script (SRT/JSON) → TTS → merge

    \b
    Examples:
      # Entry Point A: Auto-voiceover from video
      hftool voiceover --auto --video demo.mp4 --output final.mp4

      # Auto with options
      hftool voiceover --auto --video demo.mp4 --output final.mp4 \\
          --style presentation --no-edit

      # Docker two-phase workflow
      hftool voiceover --auto --video demo.mp4 --output final.mp4 \\
          --save-script script.json --no-edit

      # Entry Point B: Re-voice existing narration
      hftool voiceover --revoice --video tutorial.mp4 --output revoiced.mp4

      # Entry Point C: Manual script
      hftool voiceover --script timing.srt --video input.mp4 --output final.mp4

      # Keep original audio (ducked) + voiceover
      hftool voiceover --script timing.srt --video input.mp4 --output final.mp4 --keep-audio
    """
    from hftool.tasks.voiceover import VoiceoverTask
    from hftool.cli.commands.setup import ensure_pytorch_ready

    if not ensure_pytorch_ready():
        sys.exit(1)

    # Validate mode selection
    modes_selected = sum([mode_auto, mode_revoice, script is not None])
    if modes_selected == 0:
        click.echo(click.style(
            "Error: Specify one of --auto, --revoice, or --script",
            fg="red",
        ), err=True)
        sys.exit(1)
    if modes_selected > 1:
        click.echo(click.style(
            "Error: Only one of --auto, --revoice, or --script can be used at a time",
            fg="red",
        ), err=True)
        sys.exit(1)

    if (mode_auto or mode_revoice) and not video:
        click.echo(click.style(
            "Error: --auto and --revoice require --video",
            fg="red",
        ), err=True)
        sys.exit(1)

    task = VoiceoverTask(
        device=device,
        dtype=dtype,
        tts_model=tts_model,
        voice_ref=voice_ref,
        exaggeration=exaggeration,
        segments_dir=segments_dir,
        vlm_model=vlm_model,
        narration_style=style,
        scene_threshold=scene_threshold,
        no_edit=no_edit,
        save_script=save_script,
    )

    try:
        if mode_auto:
            task.run_auto(
                video_path=video,
                output_path=output,
                keep_audio=keep_audio,
            )
        elif mode_revoice:
            task.run_revoice(
                video_path=video,
                output_path=output,
                keep_audio=keep_audio,
            )
        else:
            task.run(
                script_path=script,
                output_path=output,
                video_path=video,
                keep_audio=keep_audio,
            )
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(1)
    finally:
        task.cleanup()
