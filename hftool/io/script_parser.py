"""Script parser for voiceover pipeline.

Parses SRT and JSON script formats into a unified internal representation
for the voiceover pipeline to generate TTS audio from timed script segments.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ScriptSegment:
    """A single timed segment in a voiceover script."""

    id: int
    start_ms: int
    end_ms: int
    text: str
    voice: Optional[str] = None
    emotion: Optional[str] = None

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def duration_s(self) -> float:
        return self.duration_ms / 1000.0


@dataclass
class ScriptData:
    """Parsed voiceover script with segments and metadata."""

    segments: List[ScriptSegment]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_duration_ms(self) -> int:
        if not self.segments:
            return 0
        return self.segments[-1].end_ms

    @property
    def total_duration_s(self) -> float:
        return self.total_duration_ms / 1000.0

    def to_srt(self) -> str:
        """Convert to SRT format string."""
        lines = []
        for seg in self.segments:
            start = _ms_to_srt_time(seg.start_ms)
            end = _ms_to_srt_time(seg.end_ms)
            lines.append(f"{seg.id}")
            lines.append(f"{start} --> {end}")
            lines.append(seg.text)
            lines.append("")
        return "\n".join(lines)

    def to_json(self, include_context: bool = False) -> str:
        """Convert to JSON format string.

        Args:
            include_context: If True and scene_contexts are available in
                metadata, add a read-only ``context`` field to each segment
                showing what is on screen at that timestamp.
        """
        contexts = self.metadata.get("scene_contexts", []) if include_context else []

        def _find_context(start_ms: int, end_ms: int) -> str:
            best, best_dist = "", float("inf")
            mid = (start_ms + end_ms) // 2
            for ctx in contexts:
                dist = abs(ctx.get("timestamp_ms", 0) - mid)
                if dist < best_dist:
                    best_dist = dist
                    best = ctx.get("description", "")
            return best

        seg_list = []
        for seg in self.segments:
            entry: dict = {
                "id": seg.id,
                "start": _ms_to_timestamp(seg.start_ms),
                "end": _ms_to_timestamp(seg.end_ms),
                "text": seg.text,
            }
            if contexts:
                ctx = _find_context(seg.start_ms, seg.end_ms)
                if ctx:
                    entry["context"] = ctx
            if seg.voice:
                entry["voice"] = seg.voice
            if seg.emotion:
                entry["emotion"] = seg.emotion
            seg_list.append(entry)

        data = {"metadata": self.metadata, "segments": seg_list}
        return json.dumps(data, indent=2, ensure_ascii=False)

    def to_editor_json(self) -> str:
        """Convert to a human-friendly JSON format for script editing.

        Uses short timestamps (M:SS) instead of milliseconds and includes
        scene context annotations and keyframe paths so the editor can see
        what is on screen without watching the video.  Delete a segment to
        create silence.
        """
        contexts = self.metadata.get("scene_contexts", [])

        def _find_context(start_ms: int, end_ms: int) -> tuple:
            """Return (description, image_path) for the closest context."""
            best_desc, best_img, best_dist = "", "", float("inf")
            mid = (start_ms + end_ms) // 2
            for ctx in contexts:
                dist = abs(ctx.get("timestamp_ms", 0) - mid)
                if dist < best_dist:
                    best_dist = dist
                    best_desc = ctx.get("description", "")
                    best_img = ctx.get("image_path", "")
            return best_desc, best_img

        segments = []
        for seg in self.segments:
            entry: dict = {
                "start": _ms_to_short_timestamp(seg.start_ms),
                "end": _ms_to_short_timestamp(seg.end_ms),
            }
            if contexts:
                desc, img = _find_context(seg.start_ms, seg.end_ms)
                if desc:
                    entry["context"] = desc
                if img:
                    entry["keyframe"] = img
            entry["text"] = seg.text
            segments.append(entry)

        return json.dumps(segments, indent=2, ensure_ascii=False)


def parse_script(path: str) -> ScriptData:
    """Parse a voiceover script file (auto-detects format by extension).

    Args:
        path: Path to SRT or JSON script file

    Returns:
        Parsed ScriptData

    Raises:
        HFToolError: If file cannot be parsed
    """
    from hftool.utils.errors import HFToolError

    if not os.path.exists(path):
        raise HFToolError(
            f"Script file not found: {path}",
            suggestion="Check the file path and try again.",
        )

    ext = os.path.splitext(path)[1].lower()

    if ext == ".srt":
        return parse_srt(path)
    elif ext == ".json":
        return parse_json(path)
    else:
        raise HFToolError(
            f"Unsupported script format: {ext}",
            suggestion="Use .srt or .json format.",
        )


def parse_srt(path: str) -> ScriptData:
    """Parse an SRT subtitle file into ScriptData.

    Args:
        path: Path to SRT file

    Returns:
        Parsed ScriptData
    """
    from hftool.utils.errors import HFToolError

    try:
        import pysrt
    except ImportError:
        raise HFToolError(
            "pysrt is required for SRT parsing.",
            suggestion="Install with: pip install hftool[with_voiceover]",
        )

    try:
        subs = pysrt.open(path)
    except Exception as e:
        raise HFToolError(
            f"Failed to parse SRT file: {e}",
            suggestion="Ensure the file is valid SRT format.",
        )

    segments = []
    for sub in subs:
        start_ms = _srt_time_to_ms(sub.start)
        end_ms = _srt_time_to_ms(sub.end)
        text = sub.text.strip()

        if not text:
            continue

        segments.append(ScriptSegment(
            id=sub.index,
            start_ms=start_ms,
            end_ms=end_ms,
            text=text,
        ))

    if not segments:
        raise HFToolError(
            "SRT file contains no valid segments.",
            suggestion="Check that the file has subtitle entries with text.",
        )

    _validate_segments(segments)

    return ScriptData(segments=segments, metadata={"source_format": "srt", "source_file": path})


def parse_json(path: str) -> ScriptData:
    """Parse a JSON voiceover script into ScriptData.

    Expected JSON format:
        {
            "metadata": { ... },
            "segments": [
                {
                    "id": 1,
                    "start": "00:00:01.000",
                    "end": "00:00:08.000",
                    "text": "...",
                    "voice": "...",      # optional
                    "emotion": "..."     # optional
                },
                ...
            ]
        }

    Args:
        path: Path to JSON file

    Returns:
        Parsed ScriptData
    """
    from hftool.utils.errors import HFToolError

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise HFToolError(
            f"Invalid JSON file: {e}",
            suggestion="Ensure the file is valid JSON.",
        )
    except OSError as e:
        raise HFToolError(f"Cannot read file: {e}")

    if "segments" not in data:
        raise HFToolError(
            "JSON file missing 'segments' key.",
            suggestion="The JSON must have a 'segments' array with id, start, end, text fields.",
        )

    metadata = data.get("metadata", {})
    metadata["source_format"] = "json"
    metadata["source_file"] = path

    segments = []
    for i, entry in enumerate(data["segments"]):
        seg_id = entry.get("id", i + 1)
        text = entry.get("text", "").strip()

        if not text:
            continue

        start_ms = _timestamp_to_ms(entry.get("start", "00:00:00.000"))
        end_ms = _timestamp_to_ms(entry.get("end", "00:00:00.000"))

        segments.append(ScriptSegment(
            id=seg_id,
            start_ms=start_ms,
            end_ms=end_ms,
            text=text,
            voice=entry.get("voice"),
            emotion=entry.get("emotion"),
        ))

    if not segments:
        raise HFToolError(
            "JSON file contains no valid segments.",
            suggestion="Check that segments have non-empty 'text' fields.",
        )

    _validate_segments(segments)

    return ScriptData(segments=segments, metadata=metadata)


def _validate_segments(segments: List[ScriptSegment]) -> None:
    """Validate segment list for ordering and overlap issues.

    Raises:
        HFToolError: If segments have invalid timestamps
    """
    from hftool.utils.errors import HFToolError

    for i, seg in enumerate(segments):
        if seg.start_ms < 0 or seg.end_ms < 0:
            raise HFToolError(
                f"Segment {seg.id} has negative timestamps.",
                suggestion="Ensure all timestamps are positive.",
            )
        if seg.end_ms <= seg.start_ms:
            raise HFToolError(
                f"Segment {seg.id} has end time <= start time ({seg.start_ms}ms -> {seg.end_ms}ms).",
                suggestion="Ensure end timestamps are after start timestamps.",
            )

    for i in range(1, len(segments)):
        if segments[i].start_ms < segments[i - 1].start_ms:
            raise HFToolError(
                f"Segments are not in chronological order: segment {segments[i].id} "
                f"starts at {segments[i].start_ms}ms but segment {segments[i-1].id} "
                f"starts at {segments[i-1].start_ms}ms.",
                suggestion="Sort segments by start time.",
            )


def _srt_time_to_ms(t) -> int:
    """Convert pysrt SubRipTime to milliseconds."""
    return (t.hours * 3600 + t.minutes * 60 + t.seconds) * 1000 + t.milliseconds


def _timestamp_to_ms(ts: str) -> int:
    """Convert HH:MM:SS.mmm timestamp string to milliseconds."""
    parts = ts.replace(",", ".").split(":")
    if len(parts) == 3:
        hours, minutes = int(parts[0]), int(parts[1])
        sec_parts = parts[2].split(".")
        seconds = int(sec_parts[0])
        millis = int(sec_parts[1].ljust(3, "0")[:3]) if len(sec_parts) > 1 else 0
        return (hours * 3600 + minutes * 60 + seconds) * 1000 + millis
    elif len(parts) == 2:
        minutes = int(parts[0])
        sec_parts = parts[1].split(".")
        seconds = int(sec_parts[0])
        millis = int(sec_parts[1].ljust(3, "0")[:3]) if len(sec_parts) > 1 else 0
        return (minutes * 60 + seconds) * 1000 + millis
    else:
        sec_parts = parts[0].split(".")
        seconds = int(sec_parts[0])
        millis = int(sec_parts[1].ljust(3, "0")[:3]) if len(sec_parts) > 1 else 0
        return seconds * 1000 + millis


def _ms_to_srt_time(ms: int) -> str:
    """Convert milliseconds to SRT timestamp format HH:MM:SS,mmm."""
    hours = ms // 3600000
    ms %= 3600000
    minutes = ms // 60000
    ms %= 60000
    seconds = ms // 1000
    millis = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _ms_to_timestamp(ms: int) -> str:
    """Convert milliseconds to HH:MM:SS.mmm timestamp string."""
    hours = ms // 3600000
    ms %= 3600000
    minutes = ms // 60000
    ms %= 60000
    seconds = ms // 1000
    millis = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _ms_to_short_timestamp(ms: int) -> str:
    """Convert milliseconds to short M:SS or H:MM:SS format for editing."""
    total_seconds = ms // 1000
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def parse_editor_segments(raw: list) -> List[ScriptSegment]:
    """Parse segments from the editor JSON array format.

    Handles both human-readable timestamps (``"start": "1:05"``) and
    millisecond timestamps (``"start_ms": 65000``).  The ``context`` field
    is ignored (it is a read-only annotation for the editor).

    Args:
        raw: List of dicts from parsed editor JSON.

    Returns:
        List of ScriptSegment objects.
    """
    segments: List[ScriptSegment] = []
    for i, entry in enumerate(raw):
        text = str(entry.get("text", "")).strip()
        if not text:
            continue

        if "start" in entry and isinstance(entry["start"], str):
            start_ms = _timestamp_to_ms(str(entry["start"]))
            end_ms = _timestamp_to_ms(str(entry.get("end", "0:00")))
        else:
            start_ms = int(entry.get("start_ms", 0))
            end_ms = int(entry.get("end_ms", 0))

        segments.append(ScriptSegment(
            id=i + 1,
            start_ms=start_ms,
            end_ms=end_ms,
            text=text,
        ))

    return segments
