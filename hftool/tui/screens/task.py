"""TaskScreen — model selection, input, parameters, and generate button."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, Static, TextArea

from hftool.tui.widgets.model_table import ModelTable


class TaskScreen(Screen):
    """Configure and launch a task."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("ctrl+g", "generate", "Generate"),
    ]

    DEFAULT_CSS = """
    TaskScreen {
        layout: vertical;
    }
    #task-config {
        padding: 1 2;
        height: 1fr;
    }
    #model-section {
        border: round $accent 40%;
        border-title-color: $text-muted;
        border-title-style: bold;
        padding: 1 2;
        margin: 0 0 1 0;
        height: auto;
    }
    #model-section:focus-within {
        border: round $accent 100%;
        border-title-color: $text;
    }
    #input-section {
        border: round $accent 40%;
        border-title-color: $text-muted;
        border-title-style: bold;
        padding: 1 2;
        margin: 0 0 1 0;
        height: auto;
    }
    #input-section:focus-within {
        border: round $accent 100%;
        border-title-color: $text;
    }
    #params-section {
        border: round $accent 40%;
        border-title-color: $text-muted;
        border-title-style: bold;
        padding: 1 2;
        margin: 0 0 1 0;
        height: auto;
    }
    #params-section:focus-within {
        border: round $accent 100%;
        border-title-color: $text;
    }
    #prompt-input {
        height: 5;
    }
    #params-row {
        height: auto;
    }
    #params-row > Input {
        width: 1fr;
        margin: 0 1 0 0;
    }
    #generate-btn {
        dock: bottom;
        width: 100%;
        margin: 1 0 0 0;
    }
    """

    def __init__(self, task_name: str):
        super().__init__()
        self._task_name = task_name

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="task-config"):
            yield Label(f"[bold]{self._task_name}[/bold]", classes="section-title")

            with Vertical(id="model-section"):
                yield ModelTable(self._task_name, id="model-table")

            with Vertical(id="input-section"):
                yield TextArea(id="prompt-input")

            with Vertical(id="params-section"):
                yield Input(placeholder="Output path (auto if empty)", id="output-path")
                with Horizontal(id="params-row"):
                    yield Input(placeholder="Seed", id="param-seed")
                    yield Input(placeholder="Device", id="param-device", value="auto")
                    yield Input(placeholder="Dtype", id="param-dtype")

            yield Button("Generate", variant="success", id="generate-btn")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#model-section").border_title = "Model"
        self.query_one("#input-section").border_title = "Input"
        self.query_one("#params-section").border_title = "Parameters"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "generate-btn":
            self._launch_generation()

    def action_generate(self) -> None:
        self._launch_generation()

    def action_back(self) -> None:
        self.app.pop_screen()

    def _launch_generation(self) -> None:
        """Gather parameters and push GenerationScreen."""
        from hftool.core.executor import TaskRequest
        from hftool.tui.screens.generation import GenerationScreen

        model_table = self.query_one("#model-table", ModelTable)
        prompt_area = self.query_one("#prompt-input", TextArea)
        output_input = self.query_one("#output-path", Input)
        seed_input = self.query_one("#param-seed", Input)
        device_input = self.query_one("#param-device", Input)
        dtype_input = self.query_one("#param-dtype", Input)

        model = model_table.get_selected_model()
        prompt = prompt_area.text.strip()
        if not prompt:
            self.notify("Please enter input text", severity="error")
            return

        seed = None
        seed_str = seed_input.value.strip()
        if seed_str:
            try:
                seed = int(seed_str)
            except ValueError:
                self.notify("Seed must be a number", severity="error")
                return

        request = TaskRequest(
            task_name=self._task_name,
            model=model,
            input_data=prompt,
            output_path=output_input.value.strip() or None,
            device=device_input.value.strip() or "auto",
            dtype=dtype_input.value.strip() or None,
            seed=seed,
        )

        self.app.push_screen(GenerationScreen(request))
