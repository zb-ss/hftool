"""Command history tracking for hftool.

Tracks command history in ~/.hftool/history.json for easy re-running and reference.
Supports interactive history browsing and automatic pruning.
"""

import json
import time
import click
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class HistoryEntry:
    """A single command history entry."""
    
    id: int
    timestamp: float
    task: str
    model: Optional[str]
    input_data: str
    output_file: Optional[str]
    device: str
    dtype: Optional[str]
    seed: Optional[int]
    extra_args: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    
    def to_command(self) -> str:
        """Reconstruct the full command line.
        
        Returns:
            Command string that can be executed
        """
        parts = ["hftool"]
        
        parts.extend(["-t", self.task])
        
        if self.model:
            parts.extend(["-m", self.model])
        
        parts.extend(["-i", f'"{self.input_data}"'])
        
        if self.output_file:
            parts.extend(["-o", self.output_file])
        
        if self.device != "auto":
            parts.extend(["-d", self.device])
        
        if self.dtype:
            parts.extend(["--dtype", self.dtype])
        
        if self.seed is not None:
            parts.extend(["--seed", str(self.seed)])
        
        # Add extra args if any
        if self.extra_args:
            parts.append("--")
            for key, value in self.extra_args.items():
                if isinstance(value, bool):
                    if value:
                        parts.append(f"--{key.replace('_', '-')}")
                else:
                    parts.append(f"--{key.replace('_', '-')}")
                    parts.append(str(value))
        
        return " ".join(parts)
    
    def get_timestamp_str(self) -> str:
        """Get formatted timestamp string.
        
        Returns:
            Human-readable timestamp
        """
        from datetime import datetime
        dt = datetime.fromtimestamp(self.timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")


class History:
    """Command history manager (singleton)."""
    
    _instance: Optional["History"] = None
    
    # Maximum number of history entries
    MAX_ENTRIES = 1000
    
    # Maximum history file size (10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024
    
    def __init__(self):
        """Initialize history manager."""
        self._history_file = Path.home() / ".hftool" / "history.json"
        self._entries: List[HistoryEntry] = []
        self._next_id = 1
        self._loaded = False
    
    @classmethod
    def get(cls) -> "History":
        """Get the singleton History instance.
        
        Returns:
            History instance
        """
        if cls._instance is None:
            cls._instance = History()
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        cls._instance = None
    
    def _ensure_loaded(self) -> None:
        """Load history from file if not already loaded."""
        if self._loaded:
            return
        
        if not self._history_file.exists():
            self._loaded = True
            return
        
        try:
            # Security: Check file size before loading
            file_size = self._history_file.stat().st_size
            if file_size > self.MAX_FILE_SIZE:
                click.echo(
                    f"Warning: History file is too large ({file_size} bytes, max {self.MAX_FILE_SIZE}). "
                    f"Truncating to last {self.MAX_ENTRIES} entries.",
                    err=True
                )
                # Load and truncate
                with open(self._history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    entries_data = data.get("entries", [])[-self.MAX_ENTRIES:]
                    self._next_id = data.get("next_id", 1)
                
                # Save truncated version
                self._save_truncated(entries_data)
            else:
                with open(self._history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    entries_data = data.get("entries", [])
                    self._next_id = data.get("next_id", 1)
            
            # Convert dicts to HistoryEntry objects
            for entry_dict in entries_data:
                # Handle missing fields for backward compatibility
                entry = HistoryEntry(
                    id=entry_dict.get("id", 0),
                    timestamp=entry_dict.get("timestamp", time.time()),
                    task=entry_dict.get("task", ""),
                    model=entry_dict.get("model"),
                    input_data=entry_dict.get("input_data", ""),
                    output_file=entry_dict.get("output_file"),
                    device=entry_dict.get("device", "auto"),
                    dtype=entry_dict.get("dtype"),
                    seed=entry_dict.get("seed"),
                    extra_args=entry_dict.get("extra_args", {}),
                    success=entry_dict.get("success", True),
                    error_message=entry_dict.get("error_message"),
                )
                self._entries.append(entry)
            
        except (json.JSONDecodeError, ValueError) as e:
            # Corrupted history file - start fresh but keep backup
            click.echo(f"Warning: Failed to load history ({e}). Starting fresh.", err=True)
            backup_path = self._history_file.with_suffix(".json.backup")
            if self._history_file.exists():
                import shutil
                shutil.copy2(self._history_file, backup_path)
                click.echo(f"Backup saved to: {backup_path}", err=True)
            self._entries = []
            self._next_id = 1
        
        self._loaded = True
    
    def _save_truncated(self, entries_data: List[Dict[str, Any]]) -> None:
        """Save truncated history.
        
        Args:
            entries_data: List of entry dictionaries
        """
        data = {
            "version": "1.0",
            "next_id": self._next_id,
            "entries": entries_data,
        }
        
        # Ensure directory exists
        self._history_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temp file first, then rename (atomic)
        temp_file = self._history_file.with_suffix(".json.tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        temp_file.replace(self._history_file)
    
    def _save(self) -> None:
        """Save history to file."""
        if not self._loaded:
            return
        
        # Truncate if needed
        if len(self._entries) > self.MAX_ENTRIES:
            self._entries = self._entries[-self.MAX_ENTRIES:]
        
        data = {
            "version": "1.0",
            "next_id": self._next_id,
            "entries": [asdict(entry) for entry in self._entries],
        }
        
        # Ensure directory exists
        self._history_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temp file first, then rename (atomic)
        temp_file = self._history_file.with_suffix(".json.tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        temp_file.replace(self._history_file)
    
    def add(
        self,
        task: str,
        model: Optional[str],
        input_data: str,
        output_file: Optional[str],
        device: str,
        dtype: Optional[str],
        seed: Optional[int],
        extra_args: Dict[str, Any],
        success: bool,
        error_message: Optional[str] = None,
    ) -> int:
        """Add a new history entry.
        
        Args:
            task: Task name
            model: Model name or path
            input_data: Input data
            output_file: Output file path
            device: Device used
            dtype: Data type used
            seed: Random seed (if any)
            extra_args: Extra arguments passed
            success: Whether command succeeded
            error_message: Error message if failed
        
        Returns:
            Entry ID
        """
        self._ensure_loaded()
        
        entry = HistoryEntry(
            id=self._next_id,
            timestamp=time.time(),
            task=task,
            model=model,
            input_data=input_data,
            output_file=output_file,
            device=device,
            dtype=dtype,
            seed=seed,
            extra_args=extra_args,
            success=success,
            error_message=error_message,
        )
        
        self._entries.append(entry)
        self._next_id += 1
        
        self._save()
        
        return entry.id
    
    def get_recent(self, limit: int = 10) -> List[HistoryEntry]:
        """Get recent history entries.
        
        Args:
            limit: Maximum number of entries to return
        
        Returns:
            List of recent entries (newest first)
        """
        self._ensure_loaded()
        return list(reversed(self._entries[-limit:]))
    
    def get_by_id(self, entry_id: int) -> Optional[HistoryEntry]:
        """Get history entry by ID.
        
        Args:
            entry_id: Entry ID to find
        
        Returns:
            HistoryEntry or None if not found
        """
        self._ensure_loaded()
        for entry in self._entries:
            if entry.id == entry_id:
                return entry
        return None
    
    def get_all(self) -> List[HistoryEntry]:
        """Get all history entries.
        
        Returns:
            List of all entries (oldest first)
        """
        self._ensure_loaded()
        return self._entries.copy()
    
    def clear(self) -> None:
        """Clear all history."""
        self._ensure_loaded()
        self._entries = []
        self._next_id = 1
        self._save()
    
    def get_recent_files(self, file_type: Optional[str] = None, limit: int = 10) -> List[str]:
        """Get recent input/output files from history.
        
        Args:
            file_type: Filter by file type (image, audio, video, text) or None for all
            limit: Maximum number of files to return
        
        Returns:
            List of recent file paths
        """
        self._ensure_loaded()
        
        files = []
        seen = set()
        
        # Iterate through history in reverse (newest first)
        for entry in reversed(self._entries):
            # Check output file
            if entry.output_file and entry.output_file not in seen:
                if file_type is None or self._matches_file_type(entry.output_file, file_type):
                    files.append(entry.output_file)
                    seen.add(entry.output_file)
            
            # Check input file (if it's a file path)
            # Skip if it looks like a prompt (no file extension, too long, or JSON)
            if entry.input_data and entry.input_data not in seen:
                from pathlib import Path
                # Skip JSON data
                if entry.input_data.startswith("{"):
                    continue
                # Skip if too long to be a valid path (filesystem limit is typically 255-4096 chars)
                if len(entry.input_data) > 255:
                    continue
                # Skip if it doesn't look like a file path (no extension or contains newlines)
                if "\n" in entry.input_data:
                    continue
                try:
                    path = Path(entry.input_data)
                    # Skip if no file extension (likely a prompt)
                    if not path.suffix:
                        continue
                    if path.exists():
                        if file_type is None or self._matches_file_type(entry.input_data, file_type):
                            files.append(entry.input_data)
                            seen.add(entry.input_data)
                except OSError:
                    # Path is invalid (too long, contains null bytes, etc.)
                    continue
            
            if len(files) >= limit:
                break
        
        return files
    
    def _matches_file_type(self, file_path: str, file_type: str) -> bool:
        """Check if file matches the specified type.
        
        Args:
            file_path: Path to file
            file_type: Type to check (image, audio, video, text)
        
        Returns:
            True if matches
        """
        from pathlib import Path
        ext = Path(file_path).suffix.lower()
        
        type_extensions = {
            "image": {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"},
            "audio": {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"},
            "video": {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"},
            "text": {".txt", ".md", ".json", ".csv"},
        }
        
        return ext in type_extensions.get(file_type, set())
