"""Tests for hftool.io.script_review module."""

import os
from unittest.mock import MagicMock, patch

import pytest


def _make_script():
    from hftool.io.script_parser import ScriptData, ScriptSegment

    return ScriptData(segments=[
        ScriptSegment(id=1, start_ms=0, end_ms=5000, text="Hello world"),
        ScriptSegment(id=2, start_ms=6000, end_ms=12000, text="Second segment"),
    ])


class TestIsDocker:
    """Tests for hftool.io.script_review._is_docker."""

    def test_returns_true_when_set(self):
        from hftool.io.script_review import _is_docker

        with patch.dict(os.environ, {"HFTOOL_IN_DOCKER": "1"}, clear=False):
            assert _is_docker() is True

    def test_returns_false_when_unset(self):
        from hftool.io.script_review import _is_docker

        env = {k: v for k, v in os.environ.items() if k != "HFTOOL_IN_DOCKER"}
        with patch.dict(os.environ, env, clear=True):
            assert _is_docker() is False

    def test_returns_true_for_true_string(self):
        from hftool.io.script_review import _is_docker

        with patch.dict(os.environ, {"HFTOOL_IN_DOCKER": "true"}, clear=False):
            assert _is_docker() is True


class TestFindEditor:
    """Tests for hftool.io.script_review._find_editor."""

    def test_uses_editor_env(self):
        from hftool.io.script_review import _find_editor

        with patch.dict(os.environ, {"EDITOR": "/usr/bin/vim", "VISUAL": ""}, clear=False):
            result = _find_editor()
        assert result == "/usr/bin/vim"

    def test_uses_visual_env(self):
        from hftool.io.script_review import _find_editor

        env_patch = {k: v for k, v in os.environ.items()}
        env_patch.pop("EDITOR", None)
        env_patch["VISUAL"] = "/usr/bin/code"
        with patch.dict(os.environ, env_patch, clear=True):
            result = _find_editor()
        assert result == "/usr/bin/code"

    def test_falls_back_to_nano(self):
        from hftool.io.script_review import _find_editor

        env_patch = {k: v for k, v in os.environ.items()}
        env_patch.pop("EDITOR", None)
        env_patch.pop("VISUAL", None)

        with patch.dict(os.environ, env_patch, clear=True):
            with patch("shutil.which", return_value="/usr/bin/nano"):
                result = _find_editor()

        assert result == "/usr/bin/nano"

    def test_returns_none_when_nothing_found(self):
        from hftool.io.script_review import _find_editor

        env_patch = {k: v for k, v in os.environ.items()}
        env_patch.pop("EDITOR", None)
        env_patch.pop("VISUAL", None)

        with patch.dict(os.environ, env_patch, clear=True):
            with patch("shutil.which", return_value=None):
                result = _find_editor()

        assert result is None


class TestOpenInEditor:
    """Tests for hftool.io.script_review.open_in_editor."""

    def test_calls_subprocess(self, tmp_path):
        from hftool.io.script_review import open_in_editor

        test_file = str(tmp_path / "script.json")
        with patch("hftool.io.script_review._find_editor", return_value="/usr/bin/nano"):
            with patch("subprocess.call", return_value=0) as mock_call:
                result = open_in_editor(test_file)

        mock_call.assert_called_once_with(["/usr/bin/nano", test_file])
        assert result is True

    def test_returns_false_on_no_editor(self, tmp_path):
        from hftool.io.script_review import open_in_editor

        with patch("hftool.io.script_review._find_editor", return_value=None):
            result = open_in_editor(str(tmp_path / "script.json"))

        assert result is False


class TestReviewScript:
    """Tests for hftool.io.script_review.review_script."""

    def test_no_edit_returns_unchanged(self, tmp_path):
        from hftool.io.script_review import review_script

        script = _make_script()
        result = review_script(script, work_dir=str(tmp_path), no_edit=True)

        assert result is script
        assert len(result.segments) == 2
        assert result.segments[0].text == "Hello world"

    def test_no_edit_saves_when_path_given(self, tmp_path):
        from hftool.io.script_review import review_script

        script = _make_script()
        save_path = str(tmp_path / "saved_script.json")

        review_script(script, work_dir=str(tmp_path), no_edit=True, save_path=save_path)

        assert os.path.exists(save_path)
        content = open(save_path).read()
        assert "Hello world" in content

    def test_docker_mode_saves_script(self, tmp_path):
        from hftool.io.script_review import review_script

        script = _make_script()
        save_path = str(tmp_path / "docker_script.json")

        # Write the file out first so parse_json can read it back after the mock input()
        with patch.dict(os.environ, {"HFTOOL_IN_DOCKER": "1"}, clear=False):
            with patch("hftool.io.script_review.input", return_value=""):
                result = review_script(
                    script,
                    work_dir=str(tmp_path),
                    save_path=save_path,
                )

        assert os.path.exists(save_path)
        assert len(result.segments) == 2

    def test_script_roundtrip(self, tmp_path):
        from hftool.io.script_parser import parse_json
        from hftool.io.script_review import _write_json

        script = _make_script()
        json_path = str(tmp_path / "roundtrip.json")

        _write_json(script, json_path)
        parsed = parse_json(json_path)

        assert len(parsed.segments) == len(script.segments)
        for orig, recovered in zip(script.segments, parsed.segments):
            assert orig.id == recovered.id
            assert orig.start_ms == recovered.start_ms
            assert orig.end_ms == recovered.end_ms
            assert orig.text == recovered.text
