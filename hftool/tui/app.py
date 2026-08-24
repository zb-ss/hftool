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

    /* ── Error state ────────────────────────────────────────── */
    .-error {
        border: round $error;
        border-title-color: $error;
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

    def on_text_selected(self) -> None:
        """Auto-copy selected text to clipboard when user highlights with mouse.

        Inside Docker the container is isolated from the host clipboard,
        so we skip the attempt and rely on Shift+mouse for native terminal
        selection instead.
        """
        import os
        import re
        from textual.widgets import TextArea

        # Inside Docker, clipboard tools can't reach the host — don't
        # show a misleading "Copied" notification.
        if os.environ.get("HFTOOL_IN_DOCKER"):
            return

        # Try screen-level selection (Static, Label, RichLog, etc.)
        text = self.screen.get_selected_text()

        # Fall back to TextArea selections
        if not text or not text.strip():
            try:
                for ta in self.screen.query(TextArea):
                    sel = ta.selected_text
                    if sel and sel.strip():
                        text = sel
                        break
            except Exception:
                pass

        if not text or not text.strip():
            return

        # Strip ANSI escape codes
        clean = re.sub(r"\x1b\[[0-9;]*m", "", text).strip()
        if not clean:
            return

        from hftool.utils.clipboard import copy_to_clipboard

        if copy_to_clipboard(clean):
            lines = len(clean.splitlines())
            label = f"{lines} lines" if lines > 1 else f"{len(clean)} chars"
            self.notify(f"Copied {label}", severity="information", timeout=2)


def main():
    """Entry point for hftool-tui command."""
    app = HFToolApp()
    app.run()


if __name__ == "__main__":
    main()
