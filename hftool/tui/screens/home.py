"""HomeScreen — task list + system info."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, ListItem, ListView, Static

from hftool.tui.widgets.system_info import SystemInfo


class HomeScreen(Screen):
    """Main screen showing available tasks and system info."""

    BINDINGS = [
        Binding("m", "models", "Models"),
        Binding("s", "settings", "Settings"),
        Binding("q", "quit", "Quit"),
    ]

    DEFAULT_CSS = """
    HomeScreen {
        layout: horizontal;
    }
    #task-panel {
        width: 1fr;
        min-width: 30;
        padding: 1 2;
    }
    #info-panel {
        width: 42;
        padding: 1 2;
    }
    #task-list {
        height: 1fr;
        border: round $accent 40%;
        border-title-color: $text-muted;
        border-title-style: bold;
    }
    #task-list:focus {
        border: round $accent 100%;
        border-title-color: $text;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="task-panel"):
                yield ListView(id="task-list")
            with Vertical(id="info-panel"):
                yield SystemInfo()
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#task-list", ListView).border_title = "Tasks"
        self._load_tasks()

    def _load_tasks(self) -> None:
        """Populate the task list from the registry."""
        from hftool.core.registry import list_tasks

        task_list = self.query_one("#task-list", ListView)
        tasks = list_tasks()

        # Group tasks by category
        categories = {
            "Generation": ["text-to-image", "image-to-image", "text-to-video", "image-to-video"],
            "Audio": ["text-to-speech", "automatic-speech-recognition"],
            "Voiceover": ["voiceover"],
            "Vision": ["vision-language-model"],
            "NLP": ["text-generation", "summarization", "text-classification",
                     "question-answering", "translation"],
        }

        shown = set()
        for category, task_names in categories.items():
            has_tasks = False
            for name in task_names:
                if name in tasks:
                    if not has_tasks:
                        task_list.append(ListItem(
                            Label(f"[bold $accent]  {category}[/]"),
                            disabled=True,
                        ))
                        has_tasks = True
                    item = ListItem(
                        Label(f"    {name}\n    [dim]{tasks[name]}[/dim]"),
                    )
                    item._task_name = name
                    task_list.append(item)
                    shown.add(name)

        # Show remaining tasks under "Other"
        remaining = {k: v for k, v in tasks.items() if k not in shown}
        if remaining:
            task_list.append(ListItem(
                Label("[bold $accent]  Other[/]"),
                disabled=True,
            ))
            for name, desc in sorted(remaining.items()):
                item = ListItem(Label(f"    {name}\n    [dim]{desc}[/dim]"))
                item._task_name = name
                task_list.append(item)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        task_name = getattr(event.item, "_task_name", None)
        if task_name:
            if task_name == "voiceover":
                from hftool.tui.screens.voiceover import VoiceoverScreen
                self.app.push_screen(VoiceoverScreen())
            else:
                from hftool.tui.screens.task import TaskScreen
                self.app.push_screen(TaskScreen(task_name))

    def action_models(self) -> None:
        from hftool.tui.screens.models import ModelBrowserScreen
        self.app.push_screen(ModelBrowserScreen())

    def action_settings(self) -> None:
        from hftool.tui.screens.settings import SettingsScreen
        self.app.push_screen(SettingsScreen())

    def action_quit(self) -> None:
        self.app.exit()
