"""Cross-platform clipboard support."""

import platform
import shutil
import subprocess


def copy_to_clipboard(text: str) -> bool:
    """Copy text to system clipboard.

    Uses platform-appropriate clipboard command:
    - macOS: pbcopy
    - Linux: wl-copy (Wayland), xclip, or xsel (X11)
    - Windows: clip

    Args:
        text: Text to copy to clipboard.

    Returns:
        True if copy succeeded, False otherwise.
    """
    system = platform.system().lower()

    try:
        if system == "darwin":
            subprocess.run(["pbcopy"], input=text.encode(), check=True)
            return True
        elif system == "linux":
            for cmd in (
                ["wl-copy"],
                ["xclip", "-selection", "clipboard"],
                ["xsel", "--clipboard", "--input"],
            ):
                if shutil.which(cmd[0]):
                    subprocess.run(cmd, input=text.encode(), check=True)
                    return True
            return False
        elif system == "windows":
            subprocess.run(["clip"], input=text.encode(), check=True)
            return True
        return False
    except (subprocess.SubprocessError, FileNotFoundError):
        return False
