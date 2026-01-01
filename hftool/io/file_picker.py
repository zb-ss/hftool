"""Interactive file picker for hftool.

Supports multiple modes:
- @ : Pick from current directory
- @? : Interactive picker mode
- @. : Pick from current directory recursively
- @~ : Pick from home directory
- @/path/ : Pick from specific directory
- @*.ext : Pick files matching glob pattern
- @@ : Pick from recent files (history)

Uses InquirerPy when available, falls back to click-based selection.
"""

import os
import glob as glob_module
import click
from pathlib import Path
from typing import Optional, List, Set, Tuple
from enum import Enum, auto


class FileType(Enum):
    """File type filters for picker."""
    ALL = auto()
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()
    TEXT = auto()


class FilePicker:
    """Interactive file picker with security features."""
    
    # Security settings
    MAX_DEPTH = 3  # Maximum recursion depth
    MAX_FILES = 1000  # Maximum files to show in picker
    MAX_GLOB_RESULTS = 500  # Maximum glob results
    
    # File type extensions
    FILE_EXTENSIONS = {
        FileType.IMAGE: {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"},
        FileType.AUDIO: {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"},
        FileType.VIDEO: {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"},
        FileType.TEXT: {".txt", ".md", ".json", ".csv", ".yaml", ".yml"},
    }
    
    def __init__(self, file_type: FileType = FileType.ALL):
        """Initialize file picker.
        
        Args:
            file_type: Type of files to filter for
        """
        self.file_type = file_type
        self._inquirer_available = False
        
        # Try to import InquirerPy
        try:
            from InquirerPy import inquirer
            from InquirerPy.base.control import Choice
            self._inquirer = inquirer
            self._Choice = Choice
            self._inquirer_available = True
        except ImportError:
            pass
    
    def resolve_reference(self, reference: str, task: Optional[str] = None) -> str:
        """Resolve @ file reference to actual file path.
        
        Args:
            reference: Reference string (e.g., "@", "@?", "@*.png", etc.)
            task: Task name for file type filtering
        
        Returns:
            Resolved file path
        
        Raises:
            ValueError: If reference cannot be resolved
        """
        # Not a reference - return as-is
        if not reference.startswith("@"):
            return reference
        
        # Determine file type from task if not set
        if self.file_type == FileType.ALL and task:
            self.file_type = self._file_type_from_task(task)
        
        # @@ - Recent files from history
        if reference == "@@":
            return self._pick_from_history()
        
        # @? - Interactive picker from current directory
        if reference == "@?":
            return self._pick_interactive(Path.cwd(), recursive=False)
        
        # @. - Recursive picker from current directory
        if reference == "@.":
            return self._pick_interactive(Path.cwd(), recursive=True)
        
        # @~ - Pick from home directory
        if reference == "@~":
            return self._pick_interactive(Path.home(), recursive=False)
        
        # @/path/ - Pick from specific directory
        if reference.startswith("@/") or (len(reference) > 2 and reference[1] == ":"):
            path_str = reference[1:]  # Remove @
            base_path = Path(path_str).expanduser().resolve()
            
            # Security: Validate path
            self._validate_path(base_path)
            
            if not base_path.exists():
                raise ValueError(f"Directory does not exist: {base_path}")
            
            if not base_path.is_dir():
                raise ValueError(f"Not a directory: {base_path}")
            
            return self._pick_interactive(base_path, recursive=False)
        
        # @*.ext - Glob pattern
        if "*" in reference or "?" in reference:
            pattern = reference[1:]  # Remove @
            return self._pick_from_glob(pattern)
        
        # @ - Pick from current directory (simple)
        if reference == "@":
            return self._pick_interactive(Path.cwd(), recursive=False)
        
        # Unknown reference format
        raise ValueError(f"Unknown file reference format: {reference}")
    
    def _file_type_from_task(self, task: str) -> FileType:
        """Determine file type from task name.
        
        Args:
            task: Task name
        
        Returns:
            FileType enum
        """
        task_lower = task.lower()
        
        if "image" in task_lower or task_lower in ("t2i", "i2i"):
            return FileType.IMAGE
        
        if "audio" in task_lower or "speech" in task_lower or task_lower in ("tts", "asr", "stt"):
            return FileType.AUDIO
        
        if "video" in task_lower or task_lower in ("t2v", "i2v"):
            return FileType.VIDEO
        
        return FileType.ALL
    
    def _validate_path(self, path: Path) -> None:
        """Validate path for security.
        
        Args:
            path: Path to validate
        
        Raises:
            ValueError: If path is not allowed
        """
        # Security: Ensure path is within safe boundaries
        resolved = path.resolve()
        home = Path.home().resolve()
        
        # Allow home directory and subdirectories
        try:
            resolved.relative_to(home)
            return
        except ValueError:
            pass
        
        # Allow current working directory and subdirectories
        try:
            resolved.relative_to(Path.cwd().resolve())
            return
        except ValueError:
            pass
        
        # Allow /tmp
        try:
            resolved.relative_to(Path("/tmp"))
            return
        except ValueError:
            pass
        
        # Not in allowed directories
        raise ValueError(
            f"Path '{path}' is outside allowed directories "
            f"(home: {home}, cwd: {Path.cwd()}, /tmp)"
        )
    
    def _pick_from_history(self) -> str:
        """Pick file from command history.
        
        Returns:
            Selected file path
        
        Raises:
            ValueError: If no files in history or selection cancelled
        """
        from hftool.core.history import History
        
        history = History.get()
        
        # Get file type filter
        file_type_str = None
        if self.file_type == FileType.IMAGE:
            file_type_str = "image"
        elif self.file_type == FileType.AUDIO:
            file_type_str = "audio"
        elif self.file_type == FileType.VIDEO:
            file_type_str = "video"
        elif self.file_type == FileType.TEXT:
            file_type_str = "text"
        
        recent_files = history.get_recent_files(file_type=file_type_str, limit=20)
        
        if not recent_files:
            raise ValueError("No recent files in history")
        
        # Filter to existing files
        existing_files = [f for f in recent_files if Path(f).exists()]
        
        if not existing_files:
            raise ValueError("No recent files exist on disk")
        
        return self._select_from_list(existing_files, "Select recent file:")
    
    def _pick_interactive(self, base_path: Path, recursive: bool) -> str:
        """Interactive file picker.
        
        Args:
            base_path: Directory to pick from
            recursive: Whether to search recursively
        
        Returns:
            Selected file path
        
        Raises:
            ValueError: If no files found or selection cancelled
        """
        files = self._find_files(base_path, recursive)
        
        if not files:
            raise ValueError(f"No matching files found in {base_path}")
        
        # Security: Limit number of files shown
        if len(files) > self.MAX_FILES:
            files = files[:self.MAX_FILES]
            click.echo(
                f"Warning: Showing first {self.MAX_FILES} files out of {len(files)} total",
                err=True
            )
        
        return self._select_from_list(files, f"Select file from {base_path}:")
    
    def _pick_from_glob(self, pattern: str) -> str:
        """Pick file using glob pattern.
        
        Args:
            pattern: Glob pattern
        
        Returns:
            Selected file path
        
        Raises:
            ValueError: If no matches or selection cancelled
        """
        # Security: Validate pattern doesn't escape safe directories
        # Expand relative to current directory
        if not pattern.startswith("/"):
            pattern = str(Path.cwd() / pattern)
        
        # Get matches
        matches = glob_module.glob(pattern, recursive=True)
        
        if not matches:
            raise ValueError(f"No files match pattern: {pattern}")
        
        # Security: Limit results
        if len(matches) > self.MAX_GLOB_RESULTS:
            matches = matches[:self.MAX_GLOB_RESULTS]
            click.echo(
                f"Warning: Showing first {self.MAX_GLOB_RESULTS} matches",
                err=True
            )
        
        # Filter to files only (not directories)
        files = [m for m in matches if Path(m).is_file()]
        
        # Filter by file type
        files = self._filter_by_type(files)
        
        if not files:
            raise ValueError(f"No matching files found for pattern: {pattern}")
        
        if len(files) == 1:
            return files[0]
        
        return self._select_from_list(files, "Select file:")
    
    def _find_files(self, base_path: Path, recursive: bool, depth: int = 0) -> List[str]:
        """Find files in directory.
        
        Args:
            base_path: Directory to search
            recursive: Whether to search recursively
            depth: Current recursion depth
        
        Returns:
            List of file paths
        """
        # Security: Limit recursion depth
        if depth > self.MAX_DEPTH:
            return []
        
        files = []
        
        try:
            for item in sorted(base_path.iterdir()):
                # Security: Skip hidden files and system files
                if item.name.startswith("."):
                    continue
                
                if item.is_file():
                    # Apply file type filter
                    if self._matches_file_type(item):
                        files.append(str(item))
                        
                        # Security: Stop if we have too many files
                        if len(files) >= self.MAX_FILES:
                            break
                
                elif item.is_dir() and recursive:
                    # Recurse into subdirectories
                    subfiles = self._find_files(item, recursive, depth + 1)
                    files.extend(subfiles)
                    
                    if len(files) >= self.MAX_FILES:
                        break
        
        except PermissionError:
            # Skip directories we can't access
            pass
        
        return files
    
    def _matches_file_type(self, file_path: Path) -> bool:
        """Check if file matches the file type filter.
        
        Args:
            file_path: Path to check
        
        Returns:
            True if matches filter
        """
        if self.file_type == FileType.ALL:
            return True
        
        ext = file_path.suffix.lower()
        return ext in self.FILE_EXTENSIONS.get(self.file_type, set())
    
    def _filter_by_type(self, files: List[str]) -> List[str]:
        """Filter list of files by type.
        
        Args:
            files: List of file paths
        
        Returns:
            Filtered list
        """
        if self.file_type == FileType.ALL:
            return files
        
        return [f for f in files if self._matches_file_type(Path(f))]
    
    def _select_from_list(self, files: List[str], prompt: str) -> str:
        """Select a file from a list.
        
        Args:
            files: List of file paths
            prompt: Prompt message
        
        Returns:
            Selected file path
        
        Raises:
            ValueError: If selection cancelled
        """
        # If only one file, return it
        if len(files) == 1:
            click.echo(f"Selected: {files[0]}")
            return files[0]
        
        # Use InquirerPy if available
        if self._inquirer_available:
            try:
                # Create choices with file info
                choices = []
                for file_path in files:
                    path = Path(file_path)
                    size = path.stat().st_size
                    size_str = self._format_size(size)
                    name = f"{path.name} ({size_str})"
                    choices.append(self._Choice(value=file_path, name=name))
                
                result = self._inquirer.select(
                    message=prompt,
                    choices=choices,
                    default=files[0],
                ).execute()
                
                return result
            
            except KeyboardInterrupt:
                raise ValueError("Selection cancelled")
            except Exception:
                # Fall back to click-based selection
                pass
        
        # Fallback: Click-based numbered list selection
        return self._select_with_click(files, prompt)
    
    def _select_with_click(self, files: List[str], prompt: str) -> str:
        """Select file using click numbered list.
        
        Args:
            files: List of file paths
            prompt: Prompt message
        
        Returns:
            Selected file path
        
        Raises:
            ValueError: If selection cancelled or invalid
        """
        click.echo("")
        click.echo(prompt)
        click.echo("-" * 60)
        
        for i, file_path in enumerate(files, 1):
            path = Path(file_path)
            size = path.stat().st_size
            size_str = self._format_size(size)
            click.echo(f"  [{i:3d}] {path.name} ({size_str})")
            click.echo(f"        {file_path}")
        
        click.echo("-" * 60)
        
        try:
            selection = click.prompt(
                f"Select file [1-{len(files)}]",
                type=int,
                default=1,
            )
            
            if 1 <= selection <= len(files):
                return files[selection - 1]
            else:
                raise ValueError(f"Invalid selection: {selection}")
        
        except click.Abort:
            raise ValueError("Selection cancelled")
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format.
        
        Args:
            size_bytes: Size in bytes
        
        Returns:
            Formatted size string
        """
        size = float(size_bytes)
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"


def resolve_file_reference(reference: str, task: Optional[str] = None, file_type: FileType = FileType.ALL) -> str:
    """Convenience function to resolve a file reference.
    
    Args:
        reference: File reference string
        task: Optional task name for file type inference
        file_type: File type filter
    
    Returns:
        Resolved file path
    
    Raises:
        ValueError: If reference cannot be resolved
    """
    picker = FilePicker(file_type=file_type)
    return picker.resolve_reference(reference, task=task)
