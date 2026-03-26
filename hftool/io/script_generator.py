"""Script generator for the auto-voiceover pipeline.

Uses a Vision-Language Model (VLM) to analyze video keyframes and generate
a timed narration script from scene detection results.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from hftool.io.scene_detector import SceneDetectionResult

from hftool.io.script_parser import ScriptData, ScriptSegment

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

FRAME_ANALYSIS_PROMPT = """Analyze this video frame in detail. Describe:
1. What is visible on screen (UI elements, text, actions)
2. What has changed compared to the previous frame
3. Any text visible on screen (read it verbatim)

Previous frame description: {previous_description}

If this is the first frame, ignore the previous description.
Be concise but thorough. Focus on what a narrator would need to describe."""

SCRIPT_ASSEMBLY_PROMPT = """You are creating a narration script for a video. Based on these frame descriptions, \
generate a timed narration script in JSON format.

Video duration: {duration_s:.1f} seconds
Narration style: {style}
{style_instructions}

Frame descriptions:
{frame_descriptions}

Generate a JSON array of narration segments. Each segment should have:
- "id": sequential number starting from 1
- "start": timestamp in "HH:MM:SS.mmm" format
- "end": timestamp in "HH:MM:SS.mmm" format  
- "text": the narration text for this segment

Rules:
- Cover the entire video duration
- Leave brief pauses between segments (0.5-1s)
- Each segment should be 5-15 seconds long
- Text should be natural spoken language
- Do not describe UI elements literally; explain what the user is doing
- Match the specified narration style

Return ONLY the JSON array, no other text."""

# ---------------------------------------------------------------------------
# Narration styles
# ---------------------------------------------------------------------------

NARRATION_STYLES: dict[str, str] = {
    "tutorial": "Use second-person ('you'). Guide step by step. Explain what to do and why. Pace: moderate, clear.",
    "presentation": "Use formal tone. Highlight key points. Suitable for conference talks. Pace: measured.",
    "demo": "Casual but professional. Focus on features being shown. Pace: energetic.",
    "casual": "Friendly first-person. Like explaining to a friend. Pace: relaxed, conversational.",
    "formal": "Third-person professional. Documentation style. Pace: steady, precise.",
}

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class FrameAnalysis:
    """VLM description of a single keyframe."""

    scene_index: int
    timestamp_ms: int
    image_path: str
    description: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_style_prompt(style: str) -> str:
    """Return narration style instructions.

    Falls back to 'tutorial' for unknown style names.

    Args:
        style: Style key (tutorial, presentation, demo, casual, formal)

    Returns:
        Style instruction string
    """
    return NARRATION_STYLES.get(style, NARRATION_STYLES["tutorial"])


def analyze_frames(
    vlm_task: Any,
    scenes: "SceneDetectionResult",
    prompt_template: Optional[str] = None,
) -> List[FrameAnalysis]:
    """Analyze all keyframes in scene detection results using a VLM.

    Args:
        vlm_task: Loaded VisionLanguageTask instance with analyze_frame()
        scenes: SceneDetectionResult from scene_detector
        prompt_template: Override the default FRAME_ANALYSIS_PROMPT

    Returns:
        List of FrameAnalysis in scene/keyframe order
    """
    template = prompt_template if prompt_template is not None else FRAME_ANALYSIS_PROMPT

    analyses: List[FrameAnalysis] = []
    prev_description = ""

    for scene in scenes.scenes:
        for image_path in scene.keyframe_paths:
            prompt = template.format(previous_description=prev_description or "None yet.")
            print(
                f"  Analyzing frame: scene {scene.index}, "
                f"t={scene.start_ms}ms ({image_path})"
            )
            description = vlm_task.analyze_frame(
                image_path,
                prompt,
                previous_context=prev_description,
            )
            analysis = FrameAnalysis(
                scene_index=scene.index,
                timestamp_ms=scene.start_ms,
                image_path=image_path,
                description=description,
            )
            analyses.append(analysis)
            prev_description = description

    return analyses


def generate_script(
    vlm_task: Any,
    analyses: List[FrameAnalysis],
    scenes: "SceneDetectionResult",
    style: str = "tutorial",
    video_duration_ms: int = 0,
) -> ScriptData:
    """Generate a timed narration ScriptData from frame analyses.

    Calls the VLM to assemble a JSON narration script, then parses it into
    ScriptData.  Falls back to evenly-spaced segments if JSON parsing fails.

    Args:
        vlm_task: Loaded VisionLanguageTask instance with run_inference()
        analyses: Frame analyses produced by analyze_frames()
        scenes: SceneDetectionResult (used for duration fallback)
        style: Narration style key (see NARRATION_STYLES)
        video_duration_ms: Override video duration (uses scenes.video_duration_ms if 0)

    Returns:
        ScriptData with timed narration segments
    """
    from hftool.utils.errors import HFToolError

    if not analyses:
        raise HFToolError(
            "No frame analyses provided — cannot generate script.",
            suggestion="Ensure scene detection produced at least one keyframe.",
        )

    duration_ms = video_duration_ms or scenes.video_duration_ms
    duration_s = duration_ms / 1000.0

    # Build frame descriptions block
    frame_descriptions = "\n".join(
        f"[{_ms_to_timestamp(a.timestamp_ms)}] Scene {a.scene_index}: {a.description}"
        for a in analyses
    )

    style_instructions = get_style_prompt(style)

    prompt = SCRIPT_ASSEMBLY_PROMPT.format(
        duration_s=duration_s,
        style=style,
        style_instructions=style_instructions,
        frame_descriptions=frame_descriptions,
    )

    print(f"  Generating narration script (style: {style}, duration: {duration_s:.1f}s)...")

    # Text-only call — script assembly uses frame descriptions, no image needed.
    # This saves ~3GB VRAM vs sending an image with the prompt.
    response = vlm_task.run_inference(
        vlm_task._pipeline,
        {"prompt": prompt},
    )

    # run_inference returns a dict; extract the text response
    response_text: str = ""
    if isinstance(response, dict):
        response_text = response.get("text", response.get("output", str(response)))
    else:
        response_text = str(response)

    # Parse the JSON script
    try:
        raw_segments = _extract_json(response_text)
        segments = _parse_raw_segments(raw_segments)
        metadata = {
            "style": style,
            "generated_by": "script_generator",
            "video_duration_ms": duration_ms,
        }
        return ScriptData(segments=segments, metadata=metadata)
    except (ValueError, KeyError, TypeError) as exc:
        print(f"  Warning: JSON parsing failed ({exc}), using fallback script.")
        return _fallback_script(analyses, duration_ms)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> list:
    """Extract a JSON array from a VLM response string.

    Handles markdown fenced blocks and tries direct JSON parsing before
    falling back to a regex-based extraction.

    Args:
        text: Raw text from VLM response

    Returns:
        Parsed list (JSON array)

    Raises:
        ValueError: If no valid JSON array can be extracted
    """
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    cleaned = cleaned.rstrip("`").strip()

    # Attempt full-text parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
        # Handle {"segments": [...]} wrapper
        if isinstance(result, dict):
            for key in ("segments", "script", "narration"):
                if isinstance(result.get(key), list):
                    return result[key]
    except json.JSONDecodeError:
        pass

    # Regex fallback: find outermost [...] block
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract a JSON array from response (length={len(text)})")


def _parse_raw_segments(raw_segments: list) -> List[ScriptSegment]:
    """Convert raw JSON dicts into ScriptSegment objects.

    Args:
        raw_segments: List of dicts with id, start, end, text keys

    Returns:
        List of ScriptSegment
    """
    from hftool.io.script_parser import _timestamp_to_ms  # reuse existing helper

    segments: List[ScriptSegment] = []
    for entry in raw_segments:
        text = str(entry.get("text", "")).strip()
        if not text:
            continue
        seg_id = int(entry.get("id", len(segments) + 1))
        start_ms = _timestamp_to_ms(str(entry.get("start", "00:00:00.000")))
        end_ms = _timestamp_to_ms(str(entry.get("end", "00:00:00.000")))
        segments.append(ScriptSegment(id=seg_id, start_ms=start_ms, end_ms=end_ms, text=text))
    return segments


def _fallback_script(analyses: List[FrameAnalysis], video_duration_ms: int) -> ScriptData:
    """Create a basic evenly-spaced script from frame descriptions.

    Used when VLM-generated JSON cannot be parsed.

    Args:
        analyses: Frame analyses with descriptions
        video_duration_ms: Total video duration in milliseconds

    Returns:
        ScriptData with evenly-spaced segments
    """
    if not analyses:
        return ScriptData(segments=[], metadata={"generated_by": "fallback_script"})

    GAP_MS = 500
    slot_ms = video_duration_ms // len(analyses)
    usable_ms = max(slot_ms - GAP_MS, 1)

    segments: List[ScriptSegment] = []
    for i, analysis in enumerate(analyses):
        start_ms = i * slot_ms
        end_ms = start_ms + usable_ms
        # Truncate description to a reasonable spoken length (~200 chars)
        text = analysis.description[:200].strip()
        if not text:
            text = f"Scene {analysis.scene_index}."
        segments.append(
            ScriptSegment(id=i + 1, start_ms=start_ms, end_ms=end_ms, text=text)
        )

    return ScriptData(
        segments=segments,
        metadata={"generated_by": "fallback_script", "video_duration_ms": video_duration_ms},
    )


def _ms_to_timestamp(ms: int) -> str:
    """Convert milliseconds to HH:MM:SS.mmm string."""
    hours = ms // 3_600_000
    ms %= 3_600_000
    minutes = ms // 60_000
    ms %= 60_000
    seconds = ms // 1_000
    millis = ms % 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
