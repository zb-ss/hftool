"""ModelBrowserScreen — browse, download, and delete models across all tasks."""

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Label, Static


class ModelBrowserScreen(Screen):
    """Browse all models across all tasks."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("d", "download", "Download"),
        Binding("x", "delete", "Delete"),
        Binding("r", "refresh", "Refresh"),
    ]

    DEFAULT_CSS = """
    ModelBrowserScreen {
        layout: vertical;
    }
    #models-content {
        padding: 1 2;
        height: 1fr;
    }
    #all-models-table {
        height: 1fr;
        border: round $accent 40%;
        border-title-color: $text-muted;
        border-title-style: bold;
    }
    #all-models-table:focus {
        border: round $accent 100%;
        border-title-color: $text;
    }
    #models-footer-info {
        height: auto;
        padding: 1 0;
        color: $text-muted;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="models-content"):
            yield DataTable(id="all-models-table")
            yield Static("", id="models-footer-info")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#all-models-table").border_title = "Models"
        self._load_all_models()

    def _load_all_models(self) -> None:
        table = self.query_one("#all-models-table", DataTable)
        table.clear(columns=True)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns("Task", "Model", "Size", "Status", "Repo ID")

        try:
            from hftool.core.models import MODEL_REGISTRY
            from hftool.core.download import get_download_status, get_models_disk_usage

            total_downloaded = 0

            for task_name, models in MODEL_REGISTRY.items():
                for short_name, info in models.items():
                    status = get_download_status(info.repo_id)
                    status_display = {
                        "downloaded": "[green]✓ Downloaded[/green]",
                        "partial": "[yellow]~ Partial[/yellow]",
                    }.get(status, "[dim]Not downloaded[/dim]")

                    if status == "downloaded":
                        total_downloaded += 1

                    default_tag = " [cyan]★[/cyan]" if info.is_default else ""
                    table.add_row(
                        task_name,
                        f"{short_name}{default_tag}",
                        info.size_str,
                        status_display,
                        info.repo_id,
                        key=f"{task_name}/{short_name}",
                    )

            try:
                usage = get_models_disk_usage()
                total_str = usage.get("total_str", "0 B")
            except Exception:
                total_str = "?"

            footer = self.query_one("#models-footer-info", Static)
            footer.update(
                f"  {total_downloaded} downloaded  ·  Disk: {total_str}  ·  "
                f"[bold]d[/bold] download  ·  [bold]x[/bold] delete  ·  [bold]r[/bold] refresh"
            )

        except Exception as e:
            self.notify(f"Error loading models: {e}", severity="error")

    def _get_selected_model(self) -> tuple[str, str] | None:
        table = self.query_one("#all-models-table", DataTable)
        if table.row_count == 0:
            return None
        try:
            row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
            key = str(row_key.value)
            if "/" in key:
                task_name, short_name = key.split("/", 1)
                return task_name, short_name
        except Exception:
            pass
        return None

    def action_download(self) -> None:
        selected = self._get_selected_model()
        if not selected:
            self.notify("No model selected", severity="warning")
            return
        task_name, short_name = selected
        self.notify(f"Downloading {short_name}...")
        self._download_model(task_name, short_name)

    @work(thread=True)
    def _download_model(self, task_name: str, short_name: str) -> None:
        try:
            from hftool.core.models import get_model_info
            from hftool.core.download import download_model_with_progress

            info = get_model_info(task_name, short_name)
            download_model_with_progress(
                repo_id=info.repo_id,
                size_gb=info.size_gb,
                pip_dependencies=info.pip_dependencies if info.pip_dependencies else None,
            )
            self.app.call_from_thread(self.notify, f"Downloaded {short_name}")
            self.app.call_from_thread(self._load_all_models)
        except Exception as e:
            self.app.call_from_thread(self.notify, f"Download failed: {e}", severity="error")

    def action_delete(self) -> None:
        selected = self._get_selected_model()
        if not selected:
            self.notify("No model selected", severity="warning")
            return
        task_name, short_name = selected
        self._delete_model(task_name, short_name)

    @work(thread=True)
    def _delete_model(self, task_name: str, short_name: str) -> None:
        try:
            from hftool.core.models import get_model_info
            from hftool.core.download import delete_model

            info = get_model_info(task_name, short_name)
            if delete_model(info.repo_id):
                self.app.call_from_thread(self.notify, f"Deleted {short_name}")
            else:
                self.app.call_from_thread(self.notify, f"Model not found on disk", severity="warning")
            self.app.call_from_thread(self._load_all_models)
        except Exception as e:
            self.app.call_from_thread(self.notify, f"Delete failed: {e}", severity="error")

    def action_refresh(self) -> None:
        self._load_all_models()
        self.notify("Refreshed")

    def action_back(self) -> None:
        self.app.pop_screen()
