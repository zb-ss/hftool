"""CLI smoke tests for the voiceover command."""

import subprocess
import sys

import pytest


class TestVoiceoverCLI:
    """Tests for the voiceover CLI subcommand."""

    def test_voiceover_help(self):
        """hftool voiceover --help should work and show all options."""
        result = subprocess.run(
            [sys.executable, "-m", "hftool", "voiceover", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "voiceover" in result.stdout.lower()
        # Original options
        assert "--script" in result.stdout
        assert "--video" in result.stdout
        assert "--output" in result.stdout
        assert "--tts-model" in result.stdout
        assert "--keep-audio" in result.stdout
        assert "--segments-dir" in result.stdout
        assert "--voice-ref" in result.stdout
        assert "--exaggeration" in result.stdout
        # New auto-voiceover options
        assert "--auto" in result.stdout
        assert "--revoice" in result.stdout
        assert "--vlm-model" in result.stdout
        assert "--style" in result.stdout
        assert "--scene-threshold" in result.stdout
        assert "--no-edit" in result.stdout
        assert "--save-script" in result.stdout

    def test_voiceover_missing_script_option(self):
        """hftool voiceover without --script should fail."""
        result = subprocess.run(
            [sys.executable, "-m", "hftool", "voiceover", "--output", "out.wav"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0

    def test_voiceover_missing_output_option(self):
        """hftool voiceover without --output should fail."""
        result = subprocess.run(
            [sys.executable, "-m", "hftool", "voiceover", "--script", "test.srt"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0

    def test_list_tasks_includes_voiceover(self):
        """hftool --list-tasks should include voiceover."""
        result = subprocess.run(
            [sys.executable, "-m", "hftool", "--list-tasks"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "voiceover" in result.stdout.lower()

    def test_version_is_0_9_0(self):
        """Version should be 0.9.0."""
        from hftool import __version__
        assert __version__ == "0.9.0"
