"""Path utilities for the voiceover TUI screen.

Docker container ↔ host path translation and cross-platform file opening.
"""

import os
import platform
import shutil
import subprocess
from typing import Callable, Optional


def container_to_host_path(container_path: str) -> str:
    """Translate a Docker container path back to the host filesystem path.

    Uses HFTOOL_REAL_HOME, HFTOOL_HOST_HOME, and HFTOOL_HOST_CWD
    environment variables set by the Docker launcher.
    """
    real_home = os.environ.get("HFTOOL_REAL_HOME", "")
    host_home = os.environ.get("HFTOOL_HOST_HOME", "/home/host")

    if real_home and container_path.startswith(host_home):
        return real_home + container_path[len(host_home):]

    if container_path.startswith("/workspace/"):
        cwd = os.environ.get("HFTOOL_HOST_CWD", "")
        if cwd:
            return os.path.join(cwd, container_path[len("/workspace/"):])

    return container_path


def open_path(
    path: str,
    notify_fn: Callable[[str, str], None],
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """Open a file or directory with the system's default application.

    When running inside Docker, translates the container path back to
    the host path and shows it in a notification (no desktop environment
    is available inside the container).

    Args:
        path: File or directory path to open.
        notify_fn: Callback for notifications — ``notify_fn(message, severity)``.
        log_fn: Optional callback for log output.
    """
    if os.environ.get("HFTOOL_IN_DOCKER"):
        host_path = container_to_host_path(path)
        notify_fn(f"Open on host: {host_path}", "information")
        if log_fn:
            log_fn(f"  [dim]Host path:[/dim] {host_path}")
        return

    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.Popen(
                ["open", path],
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif system == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            for opener in ("xdg-open", "gnome-open", "kde-open"):
                if shutil.which(opener):
                    subprocess.Popen(
                        [opener, path],
                        start_new_session=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return
            notify_fn(f"No file opener found. Path: {path}", "warning")
    except Exception as e:
        notify_fn(f"Could not open: {e}", "error")
