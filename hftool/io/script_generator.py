"""Script generator for the auto-voiceover pipeline.

Uses a Vision-Language Model (VLM) to analyze video keyframes and generate
a timed narration script from scene detection results.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

if TYPE_CHECKING:
    from hftool.io.scene_detector import SceneDetectionResult

from hftool.io.script_parser import ScriptData, ScriptSegment

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

FRAME_ANALYSIS_PROMPT = """You are narrating a screen recording. Describe what the user is DOING in this frame — the action, not just what is visible. Write 1-2 short sentences as if narrating live.

What happened before: {previous_description}

Continue the narrative naturally. Do NOT start with "This image shows" or "The screen displays". Instead describe the action: "The user opens...", "Next, they click...", "A list of results appears..."."""

SCRIPT_ASSEMBLY_PROMPT = """Create a voiceover narration for a screen recording. Narrate ONLY key moments — silence is natural and preferred over filler or repetition.

Video length: {duration_s:.1f} seconds
Style: {style} — {style_instructions}

Scene-by-scene actions (from frame analysis):
{frame_descriptions}

Write a JSON array. Each segment is a short spoken line (1-2 sentences) timed to a specific meaningful action:
[
  {{"id": 1, "start": "00:00:01.000", "end": "00:00:06.000", "text": "You open the dashboard and navigate to settings."}},
  {{"id": 2, "start": "00:00:12.000", "end": "00:00:17.000", "text": "Here you configure the notification preferences."}}
]

Rules:
- ONLY narrate when something meaningful happens — skip trivial, repetitive, or loading actions
- If consecutive scenes show the same activity continuing (scrolling, waiting, typing), narrate it ONCE then stay silent
- Never repeat information already covered in a previous segment
- Segments must NOT overlap — leave gaps of silence between narrated moments
- Each segment: 3-10 seconds of speech, max 2 sentences
- Aim for 40-70% coverage of the video — the rest should be silence
- Describe what the user DOES and WHY, not what the UI looks like
- Output ONLY the JSON array, nothing else"""

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

SCENE_GROUPING_PROMPT = """Analyze these scene descriptions from a video and group adjacent scenes that represent the SAME logical activity into unified segments.

Scenes:
{scene_descriptions}

Output a JSON array of groups. Each group contains adjacent scene indices that belong to the same activity:
[
  {{"label": "Brief activity description", "scenes": [0, 1, 2]}},
  {{"label": "Brief activity description", "scenes": [3]}},
  {{"label": "Brief activity description", "scenes": [4, 5]}}
]

Rules:
- Only merge ADJACENT scenes (consecutive indices, no gaps)
- Merge scenes that show the same action continuing (scrolling, typing, navigating the same page/area)
- Keep scenes separate if the user switches to a clearly different task, screen, or application
- A single scene can be its own group if it represents a distinct activity
- Every scene index must appear in exactly one group
- Output ONLY the JSON array, nothing else"""

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
# Internal — provider compatibility
# ---------------------------------------------------------------------------


def _vlm_text_call(vlm: Any, prompt: str) -> str:
    """Run a text-only VLM call, supporting both VLMProvider and legacy vlm_task."""
    # VLMProvider (new)
    if hasattr(vlm, "text_inference"):
        return vlm.text_inference(prompt)

    # Legacy VisionLanguageTask
    response = vlm.run_inference(vlm._pipeline, {"prompt": prompt})
    if isinstance(response, dict):
        return response.get("text", response.get("output", str(response)))
    return str(response)


def _vlm_frame_call(vlm: Any, image_path: str, prompt: str, previous_context: str = "") -> str:
    """Run a frame analysis call, supporting both VLMProvider and legacy vlm_task."""
    return vlm.analyze_frame(image_path, prompt, previous_context=previous_context)


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
            description = _vlm_frame_call(vlm_task, image_path, prompt, prev_description)
            analysis = FrameAnalysis(
                scene_index=scene.index,
                timestamp_ms=scene.start_ms,
                image_path=image_path,
                description=description,
            )
            analyses.append(analysis)
            prev_description = description

    return analyses


def group_scenes(
    vlm_task: Any,
    analyses: List[FrameAnalysis],
    scenes: "SceneDetectionResult",
) -> Tuple["SceneDetectionResult", List[FrameAnalysis]]:
    """Use VLM to group adjacent scenes that represent the same logical activity.

    Sends scene descriptions (text-only, no images) to the VLM and asks it to
    identify which adjacent scenes should be merged.  This produces fewer,
    more meaningful segments for script assembly.

    Args:
        vlm_task: Loaded VisionLanguageTask instance with run_inference()
        analyses: Frame analyses from analyze_frames()
        scenes: SceneDetectionResult from scene_detector

    Returns:
        Tuple of (updated SceneDetectionResult, updated FrameAnalysis list)
        with merged scenes and remapped indices.  Falls back to the originals
        if grouping fails.
    """
    from hftool.io.scene_detector import SceneInfo, SceneDetectionResult as SDResult

    if len(scenes.scenes) <= 2:
        return scenes, analyses

    # Index analyses by scene
    analysis_by_scene: dict[int, List[FrameAnalysis]] = {}
    for a in analyses:
        analysis_by_scene.setdefault(a.scene_index, []).append(a)

    # Build description block for the prompt
    desc_lines = []
    for scene in scenes.scenes:
        parts = [a.description for a in analysis_by_scene.get(scene.index, [])]
        description = " ".join(parts) if parts else "No description available."
        start_ts = _ms_to_timestamp(scene.start_ms)
        end_ts = _ms_to_timestamp(scene.end_ms)
        desc_lines.append(f"[{start_ts} - {end_ts}] Scene {scene.index}: {description}")

    prompt = SCENE_GROUPING_PROMPT.format(scene_descriptions="\n".join(desc_lines))

    print("  Grouping scenes by logical activity...")

    # Text-only VLM call — no image, minimal VRAM
    response_text = _vlm_text_call(vlm_task, prompt)

    try:
        raw_groups = _extract_json(response_text)
        groups = _validate_scene_groups(raw_groups, len(scenes.scenes))
    except (ValueError, KeyError, TypeError) as exc:
        print(f"  Warning: Scene grouping failed ({exc}), keeping original scenes.")
        return scenes, analyses

    # Merge scenes according to validated groups
    scene_by_index = {s.index: s for s in scenes.scenes}
    merged_scenes: List[SceneInfo] = []
    merged_analyses: List[FrameAnalysis] = []

    for group_idx, group in enumerate(groups):
        indices = group["scenes"]
        first = scene_by_index[indices[0]]
        last = scene_by_index[indices[-1]]

        combined_keyframes: List[str] = []
        for si in indices:
            combined_keyframes.extend(scene_by_index[si].keyframe_paths)

        merged_scenes.append(SceneInfo(
            index=group_idx,
            start_ms=first.start_ms,
            end_ms=last.end_ms,
            keyframe_paths=combined_keyframes,
        ))

        for si in indices:
            for a in analysis_by_scene.get(si, []):
                merged_analyses.append(FrameAnalysis(
                    scene_index=group_idx,
                    timestamp_ms=a.timestamp_ms,
                    image_path=a.image_path,
                    description=a.description,
                ))

    original_count = len(scenes.scenes)
    merged_count = len(merged_scenes)
    print(f"  Grouped {original_count} scenes → {merged_count} logical segments")

    merged_result = SDResult(
        scenes=merged_scenes,
        video_duration_ms=scenes.video_duration_ms,
        video_path=scenes.video_path,
        keyframe_dir=scenes.keyframe_dir,
    )

    return merged_result, merged_analyses


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
    response_text = _vlm_text_call(vlm_task, prompt)

    # Parse the JSON script
    try:
        raw_segments = _extract_json(response_text)
        segments = _parse_raw_segments(raw_segments)
        metadata = {
            "style": style,
            "generated_by": "script_generator",
            "video_duration_ms": duration_ms,
            "scene_contexts": [
                {"timestamp_ms": a.timestamp_ms, "scene_index": a.scene_index,
                 "description": a.description, "image_path": a.image_path}
                for a in analyses
            ],
        }
        return ScriptData(segments=segments, metadata=metadata)
    except (ValueError, KeyError, TypeError) as exc:
        print(f"  Warning: JSON parsing failed ({exc}), using fallback script.")
        return _fallback_script(analyses, duration_ms)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_scene_groups(raw_groups: list, num_scenes: int) -> List[dict]:
    """Validate and normalize VLM scene groupings.

    Ensures every scene index appears exactly once, groups contain only
    adjacent indices, and gaps are filled with singleton groups.

    Args:
        raw_groups: Parsed JSON list from VLM response
        num_scenes: Total number of scenes to validate against

    Returns:
        Sorted list of group dicts with "label" and "scenes" keys

    Raises:
        ValueError: If raw_groups is empty or has no valid entries
    """
    groups: List[dict] = []
    seen: set[int] = set()

    for entry in raw_groups:
        if not isinstance(entry, dict):
            continue
        indices = entry.get("scenes", [])
        label = str(entry.get("label", ""))

        # Filter to valid, unseen indices
        valid = [i for i in indices if isinstance(i, int) and 0 <= i < num_scenes and i not in seen]
        if not valid:
            continue

        # Split into contiguous runs (VLM might skip an index)
        valid.sort()
        runs: List[List[int]] = [[valid[0]]]
        for i in range(1, len(valid)):
            if valid[i] == runs[-1][-1] + 1:
                runs[-1].append(valid[i])
            else:
                runs.append([valid[i]])

        for run in runs:
            groups.append({"label": label, "scenes": run})
            seen.update(run)

    # Fill any missing indices as singletons
    for i in range(num_scenes):
        if i not in seen:
            groups.append({"label": f"Scene {i}", "scenes": [i]})

    groups.sort(key=lambda g: g["scenes"][0])

    if not groups:
        raise ValueError("No valid scene groups produced")

    return groups


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
