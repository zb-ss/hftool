"""Script review module for the voiceover pipeline.

Handles the review and edit step for generated voiceover scripts, supporting
Docker mode (no interactive editor), interactive $EDITOR mode, and non-interactive
(skip review) mode.
"""

import os
import shutil
import subprocess
from typing import Optional

from hftool.io.script_parser import ScriptData, parse_json
from hftool.utils.errors import HFToolError


def _is_docker() -> bool:
    """Return True if running inside a Docker container.

    Checks the HFTOOL_IN_DOCKER environment variable.
    """
    return os.environ.get("HFTOOL_IN_DOCKER", "").lower() in ("1", "true", "yes")


def _find_editor() -> Optional[str]:
    """Resolve the editor to use for interactive script editing.

    Resolution order:
        1. $EDITOR environment variable
        2. $VISUAL environment variable
        3. nano, vim, vi (first found on PATH)

    Returns:
        Absolute path to editor executable, or None if none found.
    """
    for var in ("EDITOR", "VISUAL"):
        value = os.environ.get(var, "").strip()
        if value:
            return value

    for candidate in ("nano", "vim", "vi"):
        path = shutil.which(candidate)
        if path:
            return path

    return None


def open_in_editor(file_path: str) -> bool:
    """Open a file in the user's preferred editor.

    Args:
        file_path: Absolute or relative path to the file to edit.

    Returns:
        True if the editor exited successfully (return code 0), False otherwise.
    """
    editor = _find_editor()
    if editor is None:
        return False

    try:
        return_code = subprocess.call([editor, file_path])
        return return_code == 0
    except OSError:
        return False


def review_script(
    script: ScriptData,
    work_dir: str,
    no_edit: bool = False,
    save_path: Optional[str] = None,
) -> ScriptData:
    """Review and optionally edit a generated voiceover script.

    Behaviour depends on the execution context:

    - ``no_edit=True``: Returns the script unchanged; saves to ``save_path`` if
      provided.
    - Docker mode (``HFTOOL_IN_DOCKER`` is set): Writes the script to
      ``save_path`` or ``{work_dir}/generated_script.json``, prints the path with
      instructions to edit on the host, pauses for user confirmation, then
      re-parses and returns the (potentially edited) file.
    - Interactive mode: Saves to a temp file in ``work_dir``, opens the file in
      ``$EDITOR``, re-parses and returns the result.

    In all edit-capable paths the final script is also written to ``save_path``
    when that argument is provided.

    Args:
        script: The ``ScriptData`` to review.
        work_dir: Working directory used for temporary files.
        no_edit: When True, skip all editing and return the script as-is.
        save_path: Optional explicit output path for the JSON script file.

    Returns:
        The reviewed (and potentially edited) ``ScriptData``.

    Raises:
        HFToolError: If the edited file cannot be re-parsed.
    """
    # --- no-edit fast path ---
    if no_edit:
        if save_path:
            _write_json(script, save_path)
        return script

    # --- docker mode ---
    if _is_docker():
        target_path = save_path if save_path else os.path.join(work_dir, "generated_script.json")
        _write_json(script, target_path)
        print(f"Script saved to {target_path}. Edit it on your host, then re-run with --script {target_path}")
        try:
            input("Press Enter to continue after editing...")
        except EOFError:
            pass  # Non-interactive stdin; treat as confirmed

        updated = parse_json(target_path)
        if save_path and save_path != target_path:
            _write_json(updated, save_path)
        return updated

    # --- interactive mode ---
    temp_path = os.path.join(work_dir, "script_review.json")
    _write_json(script, temp_path)

    success = open_in_editor(temp_path)
    if not success:
        raise HFToolError(
            "Could not open a text editor for script review.",
            suggestion=(
                "Set the $EDITOR environment variable to your preferred editor "
                "(e.g., export EDITOR=nano), or pass --no-edit to skip review."
            ),
        )

    updated = parse_json(temp_path)

    if save_path:
        _write_json(updated, save_path)

    return updated


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_json(script: ScriptData, path: str) -> None:
    """Write *script* as a JSON file to *path*, creating parent dirs as needed.

    Args:
        script: Script to serialise.
        path: Destination file path.

    Raises:
        HFToolError: If the file cannot be written.
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(script.to_json())
    except OSError as exc:
        raise HFToolError(
            f"Cannot write script file: {exc}",
            suggestion="Check that the destination directory is writable.",
            original_error=exc,
        ) from exc
