"""ModelTable widget — DataTable displaying models for a task."""

from textual.widgets import DataTable


class ModelTable(DataTable):
    """Reusable DataTable that shows models for a given task."""

    DEFAULT_CSS = """
    ModelTable {
        height: auto;
        max-height: 16;
        border: round $surface-lighten-2;
    }
    ModelTable:focus {
        border: round $accent;
    }
    """

    def __init__(self, task_name: str = "", **kwargs):
        super().__init__(**kwargs)
        self._task_name = task_name

    def on_mount(self) -> None:
        self.cursor_type = "row"
        self.zebra_stripes = True
        self.add_columns("Model", "Use", "VRAM", "Catalog", "Local", "License")
        if self._task_name:
            self.load_models(self._task_name)

    def load_models(self, task_name: str) -> None:
        """Load models for a given task into the table."""
        self._task_name = task_name
        self.clear()

        try:
            from hftool.core.models import get_models_for_task
            from hftool.core.download import get_model_download_status

            models = get_models_for_task(task_name)
            for short_name, info in models.items():
                download_status = get_model_download_status(info)
                status_display = {
                    "downloaded": "[green]✓[/green]",
                    "partial": "[yellow]~[/yellow]",
                }.get(download_status, "[dim]✗[/dim]")

                default_tag = " [cyan](default)[/cyan]" if info.is_default else ""
                self.add_row(
                    f"{short_name}{default_tag}",
                    info.use_case or "—",
                    f"{info.min_vram_gb:g}+ GB" if info.min_vram_gb is not None else "—",
                    info.status_label,
                    status_display,
                    info.license or "unknown",
                    key=short_name,
                )
        except Exception:
            self.add_row("Error loading models", "", "", "")

    def get_selected_model(self) -> str | None:
        """Return the short name of the selected model, or None."""
        if self.row_count == 0:
            return None
        try:
            row_key, _ = self.coordinate_to_cell_key(self.cursor_coordinate)
            return str(row_key.value)
        except Exception:
            return None

    def get_selected_info(self):
        """Return catalog information for the selected model."""
        model = self.get_selected_model()
        if model is None:
            return None
        from hftool.core.models import get_model_info

        return get_model_info(self._task_name, model)
