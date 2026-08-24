"""Tests for hftool/io/script_generator.py."""

from __future__ import annotations

import json
import pytest

from hftool.io.scene_detector import SceneDetectionResult, SceneInfo
from hftool.io.script_generator import (
    NARRATION_STYLES,
    FrameAnalysis,
    _extract_json,
    _fallback_script,
    analyze_frames,
    generate_script,
    get_style_prompt,
)
from hftool.io.script_parser import ScriptData


# ---------------------------------------------------------------------------
# Mock VLM task
# ---------------------------------------------------------------------------


class MockVLMTask:
    def __init__(self, responses=None):
        self._pipeline = {"type": "mock"}
        self._responses = responses or ["frame description"]
        self._call_count = 0

    def analyze_frame(self, image_path, prompt, previous_context=""):
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]

    def run_inference(self, pipeline, input_data, **kwargs):
        return {"text": self._responses[-1]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scenes(n: int, duration_ms: int = 10_000) -> SceneDetectionResult:
    """Build a SceneDetectionResult with n scenes, each with one keyframe path."""
    slot = duration_ms // n if n else duration_ms
    scenes = [
        SceneInfo(
            index=i,
            start_ms=i * slot,
            end_ms=(i + 1) * slot,
            keyframe_paths=[f"/tmp/fake_frame_{i}.png"],
        )
        for i in range(n)
    ]
    return SceneDetectionResult(
        scenes=scenes,
        video_duration_ms=duration_ms,
        video_path="/tmp/fake.mp4",
        keyframe_dir="/tmp/",
    )


def _make_analyses(n: int) -> list[FrameAnalysis]:
    return [
        FrameAnalysis(
            scene_index=i,
            timestamp_ms=i * 1000,
            image_path=f"/tmp/frame_{i}.png",
            description=f"Description of scene {i}.",
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# TestGetStylePrompt
# ---------------------------------------------------------------------------


class TestGetStylePrompt:
    def test_known_styles(self):
        for style in NARRATION_STYLES:
            result = get_style_prompt(style)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_unknown_style_falls_back_to_tutorial(self):
        result = get_style_prompt("nonexistent_style")
        assert result == NARRATION_STYLES["tutorial"]


# ---------------------------------------------------------------------------
# TestExtractJson
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_plain_json_array(self):
        payload = '[{"id": 1, "start": "00:00:00.000", "end": "00:00:05.000", "text": "Hello"}]'
        result = _extract_json(payload)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["text"] == "Hello"

    def test_markdown_fenced_json(self):
        payload = '```json\n[{"id": 1, "text": "Fenced"}]\n```'
        result = _extract_json(payload)
        assert isinstance(result, list)
        assert result[0]["text"] == "Fenced"

    def test_json_with_segments_key(self):
        inner = [{"id": 1, "text": "Wrapped"}]
        payload = json.dumps({"segments": inner})
        result = _extract_json(payload)
        assert result == inner

    def test_regex_fallback(self):
        payload = 'Some preamble text [{"id": 1, "text": "Extracted"}] trailing noise'
        result = _extract_json(payload)
        assert isinstance(result, list)
        assert result[0]["text"] == "Extracted"

    def test_raises_on_no_json(self):
        with pytest.raises(ValueError):
            _extract_json("completely invalid text without any JSON structure")


# ---------------------------------------------------------------------------
# TestAnalyzeFrames
# ---------------------------------------------------------------------------


class TestAnalyzeFrames:
    def test_iterates_all_keyframes(self):
        scenes = _make_scenes(3)
        vlm = MockVLMTask(responses=["desc A", "desc B", "desc C"])
        analyses = analyze_frames(vlm, scenes)
        assert len(analyses) == 3
        assert vlm._call_count == 3

    def test_passes_previous_context(self):
        """Second frame's analyze_frame call should receive the first description."""
        scenes = _make_scenes(2)

        received_contexts: list[str] = []

        class ContextCapturingVLM:
            _pipeline = {"type": "mock"}
            _call_count = 0

            def analyze_frame(self, image_path, prompt, previous_context=""):
                received_contexts.append(previous_context)
                self._call_count += 1
                return f"response {self._call_count}"

        vlm = ContextCapturingVLM()
        analyze_frames(vlm, scenes)

        # First call: no prior context (empty string)
        assert received_contexts[0] == ""
        # Second call: context is the first frame's response
        assert received_contexts[1] == "response 1"

    def test_empty_scenes_returns_empty(self):
        scenes = SceneDetectionResult(
            scenes=[],
            video_duration_ms=5000,
            video_path="/tmp/fake.mp4",
            keyframe_dir="/tmp/",
        )
        vlm = MockVLMTask()
        analyses = analyze_frames(vlm, scenes)
        assert analyses == []
        assert vlm._call_count == 0


# ---------------------------------------------------------------------------
# TestGenerateScript
# ---------------------------------------------------------------------------


class TestGenerateScript:
    def _valid_json_response(self) -> str:
        segments = [
            {"id": 1, "start": "00:00:00.000", "end": "00:00:05.000", "text": "Welcome to the demo."},
            {"id": 2, "start": "00:00:05.500", "end": "00:00:10.000", "text": "Let me show you the features."},
        ]
        return json.dumps(segments)

    def test_produces_script_data(self):
        scenes = _make_scenes(2, duration_ms=10_000)
        analyses = _make_analyses(2)
        vlm = MockVLMTask(responses=[self._valid_json_response()])
        result = generate_script(vlm, analyses, scenes, style="tutorial")
        assert isinstance(result, ScriptData)
        assert len(result.segments) == 2
        assert result.segments[0].text == "Welcome to the demo."

    def test_fallback_on_malformed_json(self):
        scenes = _make_scenes(2, duration_ms=10_000)
        analyses = _make_analyses(2)
        vlm = MockVLMTask(responses=["this is not json at all"])
        result = generate_script(vlm, analyses, scenes)
        assert isinstance(result, ScriptData)
        # Fallback creates one segment per analysis
        assert len(result.segments) == 2
        assert result.metadata.get("generated_by") == "fallback_script"

    def test_all_narration_styles(self):
        scenes = _make_scenes(1, duration_ms=5_000)
        analyses = _make_analyses(1)
        for style in NARRATION_STYLES:
            vlm = MockVLMTask(responses=[self._valid_json_response()])
            result = generate_script(vlm, analyses, scenes, style=style)
            assert isinstance(result, ScriptData)

    def test_raises_on_empty_analyses(self):
        from hftool.utils.errors import HFToolError

        scenes = _make_scenes(1, duration_ms=5_000)
        vlm = MockVLMTask()
        with pytest.raises(HFToolError):
            generate_script(vlm, [], scenes)


# ---------------------------------------------------------------------------
# TestFallbackScript
# ---------------------------------------------------------------------------


class TestFallbackScript:
    def test_creates_evenly_spaced_segments(self):
        analyses = _make_analyses(4)
        duration_ms = 8_000
        result = _fallback_script(analyses, duration_ms)

        assert len(result.segments) == 4

        slot_ms = duration_ms // 4  # 2000 ms
        gap_ms = 500
        usable_ms = slot_ms - gap_ms  # 1500 ms

        for i, seg in enumerate(result.segments):
            assert seg.start_ms == i * slot_ms
            assert seg.end_ms == i * slot_ms + usable_ms

    def test_truncates_long_descriptions(self):
        long_description = "A" * 300
        analyses = [
            FrameAnalysis(
                scene_index=0,
                timestamp_ms=0,
                image_path="/tmp/frame_0.png",
                description=long_description,
            )
        ]
        result = _fallback_script(analyses, video_duration_ms=5_000)
        assert len(result.segments) == 1
        assert len(result.segments[0].text) <= 200

    def test_empty_analyses_returns_empty_script(self):
        result = _fallback_script([], video_duration_ms=10_000)
        assert isinstance(result, ScriptData)
        assert result.segments == []
        assert result.metadata.get("generated_by") == "fallback_script"
