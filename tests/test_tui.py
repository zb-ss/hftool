"""Tests for the TUI (Terminal User Interface)."""

import asyncio
import os

import pytest


class TestTUIImports:
    """Test that all TUI modules import correctly."""

    def test_import_app(self):
        from hftool.tui.app import HFToolApp
        assert HFToolApp is not None

    def test_import_bridge(self):
        from hftool.tui.bridge import TUIProgressBridge, ProgressUpdate, StageStarted, TaskComplete
        assert TUIProgressBridge is not None
        assert ProgressUpdate is not None

    def test_import_runner(self):
        from hftool.tui.runner import TaskRunner
        assert TaskRunner is not None

    def test_import_screens(self):
        from hftool.tui.screens.home import HomeScreen
        from hftool.tui.screens.task import TaskScreen
        from hftool.tui.screens.generation import GenerationScreen
        from hftool.tui.screens.models import ModelBrowserScreen
        from hftool.tui.screens.settings import SettingsScreen
        from hftool.tui.screens.voiceover import VoiceoverScreen
        assert HomeScreen is not None
        assert VoiceoverScreen is not None

    def test_import_widgets(self):
        from hftool.tui.widgets.system_info import SystemInfo
        from hftool.tui.widgets.model_table import ModelTable
        from hftool.tui.widgets.file_browser import FilePickerScreen
        assert SystemInfo is not None
        assert ModelTable is not None
        assert FilePickerScreen is not None


class TestFileSearch:
    """Test the file search functionality."""

    def test_search_finds_python_files(self, tmp_path):
        from hftool.tui.widgets.file_browser import _search_files

        # Create test files
        (tmp_path / "foo.py").touch()
        (tmp_path / "bar.py").touch()
        (tmp_path / "baz.txt").touch()
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.py").touch()

        results = _search_files(str(tmp_path), "py", extensions=[".py"])
        filenames = [os.path.basename(r) for r in results]
        assert "foo.py" in filenames
        assert "bar.py" in filenames
        assert "deep.py" in filenames
        assert "baz.txt" not in filenames

    def test_search_substring_match(self, tmp_path):
        from hftool.tui.widgets.file_browser import _search_files

        (tmp_path / "servonaut-1.mp4").touch()
        (tmp_path / "other-video.mp4").touch()

        results = _search_files(str(tmp_path), "servo")
        assert len(results) == 1
        assert "servonaut-1.mp4" in results[0]

    def test_search_case_insensitive(self, tmp_path):
        from hftool.tui.widgets.file_browser import _search_files

        (tmp_path / "MyVideo.MP4").touch()
        results = _search_files(str(tmp_path), "myvideo")
        assert len(results) == 1

    def test_search_respects_max_results(self, tmp_path):
        from hftool.tui.widgets.file_browser import _search_files

        for i in range(20):
            (tmp_path / f"file_{i}.txt").touch()

        results = _search_files(str(tmp_path), "file", max_results=5)
        assert len(results) == 5

    def test_search_skips_hidden_dirs(self, tmp_path):
        from hftool.tui.widgets.file_browser import _search_files

        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "secret.txt").touch()
        (tmp_path / "visible.txt").touch()

        results = _search_files(str(tmp_path), "txt")
        filenames = [os.path.basename(r) for r in results]
        assert "visible.txt" in filenames
        assert "secret.txt" not in filenames

    def test_get_browse_root_native(self):
        from hftool.tui.widgets.file_browser import get_browse_root

        # Outside Docker, should return home dir
        old = os.environ.pop("HFTOOL_IN_DOCKER", None)
        try:
            root = get_browse_root()
            assert root == os.path.expanduser("~")
        finally:
            if old is not None:
                os.environ["HFTOOL_IN_DOCKER"] = old


class TestTUIBridge:
    """Test the TUI progress bridge."""

    def test_bridge_cancel(self):
        from hftool.tui.bridge import TUIProgressBridge
        messages = []
        bridge = TUIProgressBridge(post_message=messages.append)
        assert not bridge.is_cancelled
        bridge.cancel()
        assert bridge.is_cancelled

    def test_bridge_update_raises_on_cancel(self):
        from hftool.tui.bridge import TUIProgressBridge
        bridge = TUIProgressBridge(post_message=lambda m: None)
        bridge.cancel()
        with pytest.raises(KeyboardInterrupt, match="Cancelled"):
            bridge.update(1, 10)

    def test_bridge_start_stage(self):
        from hftool.tui.bridge import TUIProgressBridge, StageStarted
        messages = []
        bridge = TUIProgressBridge(post_message=messages.append)
        bridge.start_stage("Loading model", total=5)
        assert len(messages) == 1
        assert isinstance(messages[0], StageStarted)
        assert messages[0].name == "Loading model"
        assert messages[0].total == 5

    def test_bridge_update(self):
        from hftool.tui.bridge import TUIProgressBridge, ProgressUpdate
        messages = []
        bridge = TUIProgressBridge(post_message=messages.append)
        bridge.update(3, 10, "Processing...")
        assert len(messages) == 1
        assert isinstance(messages[0], ProgressUpdate)
        assert messages[0].current == 3
        assert messages[0].total == 10


class TestTUIRunner:
    """Test the task runner."""

    def test_runner_creation(self):
        from hftool.core.executor import TaskRequest
        from hftool.tui.runner import TaskRunner

        request = TaskRequest(
            task_name="text-to-image",
            input_data="A cat",
        )
        runner = TaskRunner(request)
        assert runner.request is request
        assert runner.bridge is None

    def test_runner_with_progress(self):
        from hftool.core.executor import TaskRequest
        from hftool.tui.runner import TaskRunner

        request = TaskRequest(
            task_name="text-to-image",
            input_data="A cat",
        )
        runner = TaskRunner(request, post_message=lambda m: None)
        assert runner.bridge is not None

    def test_runner_cancel(self):
        from hftool.core.executor import TaskRequest
        from hftool.tui.runner import TaskRunner

        request = TaskRequest(task_name="text-to-image", input_data="A cat")
        runner = TaskRunner(request, post_message=lambda m: None)
        runner.cancel()
        assert runner.bridge.is_cancelled


class TestCoreExecutor:
    """Test the core executor dataclasses."""

    def test_task_request_defaults(self):
        from hftool.core.executor import TaskRequest

        req = TaskRequest(task_name="text-to-image")
        assert req.task_name == "text-to-image"
        assert req.model is None
        assert req.input_data == ""
        assert req.device == "auto"
        assert req.extra_kwargs == {}

    def test_task_result_success(self):
        from hftool.core.executor import TaskResult

        result = TaskResult(success=True, output_path="/tmp/out.png", elapsed_s=1.5)
        assert result.success
        assert result.output_path == "/tmp/out.png"
        assert result.error is None

    def test_task_result_failure(self):
        from hftool.core.executor import TaskResult

        result = TaskResult(success=False, error="Model not found")
        assert not result.success
        assert result.error == "Model not found"


class TestTUIApp:
    """Test the TUI app initialization using Textual's test harness."""

    @pytest.fixture
    def event_loop(self):
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    def test_app_launches_home_screen(self, event_loop):
        from hftool.tui.app import HFToolApp

        async def _test():
            app = HFToolApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                assert app.screen.__class__.__name__ == "HomeScreen"

        event_loop.run_until_complete(_test())

    def test_app_title(self, event_loop):
        from hftool.tui.app import HFToolApp

        async def _test():
            app = HFToolApp()
            async with app.run_test(size=(120, 40)) as pilot:
                assert app.title == "hftool"

        event_loop.run_until_complete(_test())

    def test_models_screen_navigation(self, event_loop):
        from hftool.tui.app import HFToolApp

        async def _test():
            app = HFToolApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                await pilot.press("m")
                await pilot.pause()
                assert app.screen.__class__.__name__ == "ModelBrowserScreen"
                await pilot.press("escape")
                await pilot.pause()
                assert app.screen.__class__.__name__ == "HomeScreen"

        event_loop.run_until_complete(_test())

    def test_settings_screen_navigation(self, event_loop):
        from hftool.tui.app import HFToolApp

        async def _test():
            app = HFToolApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                await pilot.press("s")
                await pilot.pause()
                assert app.screen.__class__.__name__ == "SettingsScreen"

        event_loop.run_until_complete(_test())


class TestTUICommand:
    """Test the CLI tui subcommand."""

    def test_tui_help(self):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "hftool", "tui", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "Launch interactive TUI" in result.stdout
        assert "--native" in result.stdout

    def test_tui_native_flag_in_help(self):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "hftool", "tui", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert "--native" in result.stdout
        assert "Docker" in result.stdout
