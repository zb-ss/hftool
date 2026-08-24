"""Textual and CLI regression coverage for catalog status presentation."""

import json
import subprocess
import sys

import pytest


@pytest.mark.asyncio
async def test_model_table_renders_recommended_and_legacy_statuses():
    from textual.app import App, ComposeResult

    from hftool.tui.widgets.model_table import ModelTable

    class CatalogApp(App):
        def compose(self) -> ComposeResult:
            yield ModelTable("text-to-image", id="models")

    app = CatalogApp()
    async with app.run_test(size=(150, 35)) as pilot:
        await pilot.pause()
        table = app.query_one("#models", ModelTable)
        recommended_row = [str(value) for value in table.get_row("flux2-klein-4b")]
        legacy_row = [str(value) for value in table.get_row("sdxl")]

    assert "recommended" in recommended_row
    assert "Apache-2.0" in recommended_row
    assert "legacy" in legacy_row


def test_cli_dry_run_is_json_and_does_not_require_a_download():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hftool",
            "-t",
            "t2i",
            "-m",
            "flux2-klein-4b",
            "-i",
            "test prompt",
            "--dry-run",
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    preview = json.loads(result.stdout)
    assert preview["catalog_status"] == "recommended"
    assert preview["download_status"] in {"downloaded", "partial", "not downloaded"}
    assert preview["inference_defaults"]["num_inference_steps"] == 4
    assert "gpu_message" in preview
