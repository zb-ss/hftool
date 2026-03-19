"""Tests for the script parser module."""

import json
import os
import tempfile

import pytest

try:
    import pysrt
    _PYSRT_AVAILABLE = True
except ImportError:
    _PYSRT_AVAILABLE = False

_requires_pysrt = pytest.mark.skipif(not _PYSRT_AVAILABLE, reason="pysrt not installed")


class TestScriptSegment:
    """Tests for the ScriptSegment dataclass."""

    def test_duration_ms(self):
        from hftool.io.script_parser import ScriptSegment
        seg = ScriptSegment(id=1, start_ms=1000, end_ms=5000, text="Hello")
        assert seg.duration_ms == 4000

    def test_duration_s(self):
        from hftool.io.script_parser import ScriptSegment
        seg = ScriptSegment(id=1, start_ms=0, end_ms=2500, text="Hello")
        assert seg.duration_s == 2.5

    def test_optional_fields(self):
        from hftool.io.script_parser import ScriptSegment
        seg = ScriptSegment(id=1, start_ms=0, end_ms=1000, text="Hello", voice="narrator", emotion="happy")
        assert seg.voice == "narrator"
        assert seg.emotion == "happy"


class TestScriptData:
    """Tests for the ScriptData dataclass."""

    def test_total_duration(self):
        from hftool.io.script_parser import ScriptData, ScriptSegment
        segments = [
            ScriptSegment(id=1, start_ms=0, end_ms=5000, text="First"),
            ScriptSegment(id=2, start_ms=5000, end_ms=10000, text="Second"),
        ]
        data = ScriptData(segments=segments)
        assert data.total_duration_ms == 10000
        assert data.total_duration_s == 10.0

    def test_empty_segments(self):
        from hftool.io.script_parser import ScriptData
        data = ScriptData(segments=[])
        assert data.total_duration_ms == 0

    def test_to_srt(self):
        from hftool.io.script_parser import ScriptData, ScriptSegment
        segments = [
            ScriptSegment(id=1, start_ms=1000, end_ms=5000, text="Hello world"),
        ]
        data = ScriptData(segments=segments)
        srt = data.to_srt()
        assert "1" in srt
        assert "00:00:01,000 --> 00:00:05,000" in srt
        assert "Hello world" in srt

    def test_to_json(self):
        from hftool.io.script_parser import ScriptData, ScriptSegment
        segments = [
            ScriptSegment(id=1, start_ms=1000, end_ms=5000, text="Hello world"),
        ]
        data = ScriptData(segments=segments, metadata={"title": "test"})
        result = json.loads(data.to_json())
        assert result["metadata"]["title"] == "test"
        assert len(result["segments"]) == 1
        assert result["segments"][0]["text"] == "Hello world"


@_requires_pysrt
class TestParseSRT:
    """Tests for SRT parsing."""

    def _write_srt(self, content):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False)
        f.write(content)
        f.close()
        return f.name

    def test_parse_simple_srt(self):
        from hftool.io.script_parser import parse_srt

        path = self._write_srt(
            "1\n00:00:01,000 --> 00:00:05,000\nHello world\n\n"
            "2\n00:00:06,000 --> 00:00:10,000\nGoodbye world\n\n"
        )
        try:
            result = parse_srt(path)
            assert len(result.segments) == 2
            assert result.segments[0].text == "Hello world"
            assert result.segments[0].start_ms == 1000
            assert result.segments[0].end_ms == 5000
            assert result.segments[1].text == "Goodbye world"
        finally:
            os.unlink(path)

    def test_parse_srt_skips_empty_text(self):
        from hftool.io.script_parser import parse_srt

        path = self._write_srt(
            "1\n00:00:01,000 --> 00:00:05,000\nHello\n\n"
            "2\n00:00:06,000 --> 00:00:10,000\n  \n\n"
            "3\n00:00:11,000 --> 00:00:15,000\nWorld\n\n"
        )
        try:
            result = parse_srt(path)
            assert len(result.segments) == 2
        finally:
            os.unlink(path)

    def test_parse_srt_metadata(self):
        from hftool.io.script_parser import parse_srt

        path = self._write_srt("1\n00:00:01,000 --> 00:00:05,000\nHello\n\n")
        try:
            result = parse_srt(path)
            assert result.metadata["source_format"] == "srt"
        finally:
            os.unlink(path)

    def test_parse_srt_file_not_found(self):
        from hftool.io.script_parser import parse_srt
        from hftool.utils.errors import HFToolError

        with pytest.raises(HFToolError):
            parse_srt("/nonexistent/file.srt")

    def test_parse_srt_empty_file_raises(self):
        from hftool.io.script_parser import parse_srt
        from hftool.utils.errors import HFToolError

        path = self._write_srt("")
        try:
            with pytest.raises(HFToolError, match="no valid segments"):
                parse_srt(path)
        finally:
            os.unlink(path)


class TestParseJSON:
    """Tests for JSON parsing."""

    def _write_json(self, data):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(data, f)
        f.close()
        return f.name

    def test_parse_simple_json(self):
        from hftool.io.script_parser import parse_json

        data = {
            "metadata": {"title": "Test"},
            "segments": [
                {"id": 1, "start": "00:00:01.000", "end": "00:00:05.000", "text": "Hello world"},
                {"id": 2, "start": "00:00:06.000", "end": "00:00:10.000", "text": "Goodbye"},
            ],
        }
        path = self._write_json(data)
        try:
            result = parse_json(path)
            assert len(result.segments) == 2
            assert result.segments[0].text == "Hello world"
            assert result.segments[0].start_ms == 1000
            assert result.segments[0].end_ms == 5000
            assert result.metadata["title"] == "Test"
        finally:
            os.unlink(path)

    def test_parse_json_with_emotion(self):
        from hftool.io.script_parser import parse_json

        data = {
            "segments": [
                {"id": 1, "start": "00:00:01.000", "end": "00:00:05.000", "text": "Hello", "emotion": "happy"},
            ],
        }
        path = self._write_json(data)
        try:
            result = parse_json(path)
            assert result.segments[0].emotion == "happy"
        finally:
            os.unlink(path)

    def test_parse_json_missing_segments_key(self):
        from hftool.io.script_parser import parse_json
        from hftool.utils.errors import HFToolError

        path = self._write_json({"data": []})
        try:
            with pytest.raises(HFToolError, match="missing 'segments'"):
                parse_json(path)
        finally:
            os.unlink(path)

    def test_parse_json_empty_segments(self):
        from hftool.io.script_parser import parse_json
        from hftool.utils.errors import HFToolError

        path = self._write_json({"segments": []})
        try:
            with pytest.raises(HFToolError, match="no valid segments"):
                parse_json(path)
        finally:
            os.unlink(path)


class TestParseScript:
    """Tests for auto-detection parse_script."""

    @_requires_pysrt
    def test_detect_srt(self):
        from hftool.io.script_parser import parse_script

        f = tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False)
        f.write("1\n00:00:01,000 --> 00:00:05,000\nHello\n\n")
        f.close()
        try:
            result = parse_script(f.name)
            assert len(result.segments) == 1
        finally:
            os.unlink(f.name)

    def test_detect_json(self):
        from hftool.io.script_parser import parse_script

        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump({
            "segments": [
                {"id": 1, "start": "00:00:01.000", "end": "00:00:05.000", "text": "Hello"},
            ]
        }, f)
        f.close()
        try:
            result = parse_script(f.name)
            assert len(result.segments) == 1
        finally:
            os.unlink(f.name)

    def test_unsupported_format(self):
        from hftool.io.script_parser import parse_script
        from hftool.utils.errors import HFToolError

        f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        f.write("hello")
        f.close()
        try:
            with pytest.raises(HFToolError, match="Unsupported script format"):
                parse_script(f.name)
        finally:
            os.unlink(f.name)

    def test_file_not_found(self):
        from hftool.io.script_parser import parse_script
        from hftool.utils.errors import HFToolError

        with pytest.raises(HFToolError, match="not found"):
            parse_script("/nonexistent/script.srt")


class TestTimestampConversions:
    """Tests for timestamp conversion helpers."""

    def test_ms_to_srt_time(self):
        from hftool.io.script_parser import _ms_to_srt_time
        assert _ms_to_srt_time(0) == "00:00:00,000"
        assert _ms_to_srt_time(1500) == "00:00:01,500"
        assert _ms_to_srt_time(3661500) == "01:01:01,500"

    def test_ms_to_timestamp(self):
        from hftool.io.script_parser import _ms_to_timestamp
        assert _ms_to_timestamp(0) == "00:00:00.000"
        assert _ms_to_timestamp(1500) == "00:00:01.500"

    def test_timestamp_to_ms(self):
        from hftool.io.script_parser import _timestamp_to_ms
        assert _timestamp_to_ms("00:00:01.000") == 1000
        assert _timestamp_to_ms("01:30:00.000") == 5400000
        assert _timestamp_to_ms("00:00:01.500") == 1500

    def test_timestamp_roundtrip(self):
        from hftool.io.script_parser import _ms_to_timestamp, _timestamp_to_ms
        for ms in [0, 500, 1000, 61000, 3661500]:
            assert _timestamp_to_ms(_ms_to_timestamp(ms)) == ms


class TestValidation:
    """Tests for segment validation."""

    def test_negative_timestamp_raises(self):
        from hftool.io.script_parser import _validate_segments, ScriptSegment
        from hftool.utils.errors import HFToolError

        segments = [ScriptSegment(id=1, start_ms=-100, end_ms=1000, text="Bad")]
        with pytest.raises(HFToolError, match="negative"):
            _validate_segments(segments)

    def test_end_before_start_raises(self):
        from hftool.io.script_parser import _validate_segments, ScriptSegment
        from hftool.utils.errors import HFToolError

        segments = [ScriptSegment(id=1, start_ms=5000, end_ms=1000, text="Bad")]
        with pytest.raises(HFToolError, match="end time <= start time"):
            _validate_segments(segments)

    def test_out_of_order_raises(self):
        from hftool.io.script_parser import _validate_segments, ScriptSegment
        from hftool.utils.errors import HFToolError

        segments = [
            ScriptSegment(id=1, start_ms=5000, end_ms=8000, text="Second"),
            ScriptSegment(id=2, start_ms=1000, end_ms=4000, text="First"),
        ]
        with pytest.raises(HFToolError, match="not in chronological order"):
            _validate_segments(segments)

    def test_valid_segments_pass(self):
        from hftool.io.script_parser import _validate_segments, ScriptSegment

        segments = [
            ScriptSegment(id=1, start_ms=0, end_ms=5000, text="First"),
            ScriptSegment(id=2, start_ms=5000, end_ms=10000, text="Second"),
        ]
        _validate_segments(segments)  # Should not raise
