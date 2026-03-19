"""HFToolApp — main Textual application."""

from textual.app import App

from hftool.tui.screens.home import HomeScreen


class HFToolApp(App):
    """hftool Terminal User Interface."""

    TITLE = "hftool"
    SUB_TITLE = "Hugging Face Model Runner"

    CSS = """
    /* ── Global ─────────────────────────────────────────────── */
    Screen {
        background: $background;
        scrollbar-color: $primary 10%;
        scrollbar-color-hover: $primary 80%;
        scrollbar-color-active: $primary;
        scrollbar-background: $surface-darken-1;
        scrollbar-size-vertical: 1;
    }

    /* ── Sections ───────────────────────────────────────────── */
    .section {
        border: round $accent 40%;
        border-title-color: $text-muted;
        border-title-style: bold;
        padding: 1 2;
        margin: 0 0 1 0;
        height: auto;
    }
    .section:focus-within {
        border: round $accent 100%;
        border-title-color: $text;
    }

    .section-title {
        text-style: bold;
        color: $accent;
        margin: 0 0 1 0;
    }

    /* ── Inputs ─────────────────────────────────────────────── */
    Input {
        border: round $surface-lighten-2;
        padding: 0 1;
        height: 3;
        margin: 0 0 1 0;
    }
    Input:focus {
        border: round $accent;
    }

    TextArea {
        border: round $surface-lighten-2;
        margin: 0 0 1 0;
    }
    TextArea:focus {
        border: round $accent;
    }

    /* ── Buttons ────────────────────────────────────────────── */
    Button {
        margin: 0 1 0 0;
    }

    /* ── Select ─────────────────────────────────────────────── */
    Select {
        margin: 0 0 1 0;
    }

    /* ── Labels ─────────────────────────────────────────────── */
    .field-label {
        margin: 1 0 0 0;
        text-style: bold;
        color: $text;
    }
    .field-help {
        color: $text-muted;
        margin: 0 0 1 0;
    }

    /* ── DataTable ──────────────────────────────────────────── */
    DataTable {
        border: round $surface-lighten-2;
        height: auto;
        max-height: 20;
    }
    DataTable:focus {
        border: round $accent;
    }

    /* ── RichLog ────────────────────────────────────────────── */
    RichLog {
        border: round $surface-lighten-2;
        height: 1fr;
        min-height: 6;
    }

    /* ── ProgressBar ────────────────────────────────────────── */
    ProgressBar {
        margin: 0 0 1 0;
    }

    /* ── ListView ───────────────────────────────────────────── */
    ListView {
        border: round $surface-lighten-2;
        height: 1fr;
    }
    ListView:focus {
        border: round $accent;
    }
    ListView > ListItem {
        padding: 0 1;
    }
    ListView > ListItem.-highlight {
        background: $accent 20%;
    }
    """

    SCREENS = {"home": HomeScreen}

    def on_mount(self) -> None:
        self.push_screen("home")


def main():
    """Entry point for hftool-tui command."""
    app = HFToolApp()
    app.run()


if __name__ == "__main__":
    main()
