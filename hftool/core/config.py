"""Configuration management for hftool.

Supports TOML config files with priority chain:
1. CLI arguments (highest priority)
2. Environment variables (HFTOOL_*)
3. Project config (./.hftool/config.toml)
4. User config (~/.hftool/config.toml)
5. Built-in defaults (lowest priority)
"""

import os
import re
from pathlib import Path
from typing import Any, Optional, Dict

# Try to use stdlib tomllib (Python 3.11+), fallback to tomli
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        tomllib = None  # type: ignore


class Config:
    """Configuration manager for hftool (singleton)."""
    
    _instance: Optional["Config"] = None
    
    # Security: Maximum config file size (1MB)
    MAX_CONFIG_SIZE = 1024 * 1024
    
    # Security: Environment variable name pattern (must start with HFTOOL_)
    ENV_VAR_PATTERN = re.compile(r'^HFTOOL_[A-Z0-9_]+$')
    
    def __init__(self):
        """Initialize configuration manager."""
        self._config: Dict[str, Any] = {}
        self._user_config_path: Optional[Path] = None
        self._project_config_path: Optional[Path] = None
        self._loaded = False
    
    @classmethod
    def get(cls) -> "Config":
        """Get the singleton Config instance.
        
        Returns:
            Config instance
        """
        if cls._instance is None:
            cls._instance = Config()
            cls._instance._load_config()
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        cls._instance = None
    
    def _load_config(self) -> None:
        """Load config from all sources (user + project)."""
        if self._loaded:
            return
        
        if tomllib is None:
            # Config files not supported - continue without them
            self._loaded = True
            return
        
        # Load user config first
        user_config_path = Path.home() / ".hftool" / "config.toml"
        if user_config_path.exists():
            try:
                # Security: Check file size before parsing (M-1)
                file_size = user_config_path.stat().st_size
                if file_size > self.MAX_CONFIG_SIZE:
                    import sys
                    print(
                        f"Warning: Config file {user_config_path} is too large "
                        f"({file_size} bytes, max {self.MAX_CONFIG_SIZE})",
                        file=sys.stderr
                    )
                else:
                    with open(user_config_path, "rb") as f:
                        self._config = tomllib.load(f)
                    self._user_config_path = user_config_path
            except Exception as e:
                # Log error but don't fail - config is optional
                import sys
                print(f"Warning: Failed to load user config from {user_config_path}: {e}", file=sys.stderr)
        
        # Load project config (overrides user config)
        project_config_path = Path.cwd() / ".hftool" / "config.toml"
        if project_config_path.exists():
            try:
                # Security: Check file size before parsing (M-1)
                file_size = project_config_path.stat().st_size
                if file_size > self.MAX_CONFIG_SIZE:
                    import sys
                    print(
                        f"Warning: Config file {project_config_path} is too large "
                        f"({file_size} bytes, max {self.MAX_CONFIG_SIZE})",
                        file=sys.stderr
                    )
                else:
                    with open(project_config_path, "rb") as f:
                        project_config = tomllib.load(f)
                        self._merge(project_config)
                    self._project_config_path = project_config_path
            except Exception as e:
                import sys
                print(f"Warning: Failed to load project config from {project_config_path}: {e}", file=sys.stderr)
        
        self._loaded = True
    
    def _merge(self, new_config: Dict[str, Any]) -> None:
        """Merge new config into existing config (new values override).
        
        Args:
            new_config: Configuration dict to merge
        """
        for key, value in new_config.items():
            if isinstance(value, dict) and key in self._config and isinstance(self._config[key], dict):
                # Merge nested dicts
                self._config[key] = {**self._config[key], **value}
            else:
                # Override value
                self._config[key] = value
    
    def get_value(
        self,
        key: str,
        task: Optional[str] = None,
        default: Any = None,
        env_var: Optional[str] = None,
    ) -> Any:
        """Get config value with fallback chain.
        
        Priority order:
        1. Environment variable (if env_var specified)
        2. Task-specific config (if task specified)
        3. Defaults section in config
        4. Provided default value
        
        Args:
            key: Configuration key to look up
            task: Task name for task-specific config
            default: Default value if not found
            env_var: Environment variable name to check (auto-generated if None)
        
        Returns:
            Configuration value
        """
        # Check environment variable first
        if env_var is None:
            env_var = f"HFTOOL_{key.upper()}"
        
        # Security: Validate environment variable name (M-2)
        if not self.ENV_VAR_PATTERN.match(env_var):
            # Invalid env var name - skip and continue to next priority
            env_value = None
        else:
            env_value = os.environ.get(env_var)
        
        if env_value is not None:
            return self._parse_env_value(env_value)
        
        # Check task-specific config
        if task and task in self._config:
            task_config = self._config[task]
            if isinstance(task_config, dict) and key in task_config:
                return task_config[key]
        
        # Check defaults section
        if "defaults" in self._config and isinstance(self._config["defaults"], dict):
            if key in self._config["defaults"]:
                return self._config["defaults"][key]
        
        # Return default
        return default
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type.
        
        Args:
            value: String value from environment
        
        Returns:
            Parsed value (bool, int, float, or str)
        """
        # Boolean
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
        
        # Integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        
        # String
        return value
    
    def resolve_model_alias(self, alias: str) -> str:
        """Resolve a model alias to repo_id.
        
        Args:
            alias: Short name or alias for a model
        
        Returns:
            Resolved repo_id (or original alias if not found)
        """
        aliases = self._config.get("aliases", {})
        if not isinstance(aliases, dict):
            return alias
        
        return aliases.get(alias, alias)
    
    def get_path(self, key: str, default: Optional[str] = None) -> Optional[Path]:
        """Get a path from config and expand it.
        
        Args:
            key: Path key in config
            default: Default path if not found
        
        Returns:
            Expanded Path object or None
        
        Raises:
            ValueError: If resolved path is outside user's home directory
        """
        path_str = self.get_value(key, default=default)
        if path_str is None:
            return None
        
        # Expand user path
        resolved_path = Path(path_str).expanduser().resolve()
        
        # Security: Validate path is within safe boundaries (H-1)
        # Allow paths within home directory or /tmp
        home_dir = Path.home().resolve()
        tmp_dir = Path("/tmp").resolve()
        
        try:
            # Check if path is relative to home or /tmp
            is_in_home = resolved_path.is_relative_to(home_dir)
        except ValueError:
            is_in_home = False
        
        try:
            is_in_tmp = resolved_path.is_relative_to(tmp_dir)
        except ValueError:
            is_in_tmp = False
        
        if not (is_in_home or is_in_tmp):
            raise ValueError(
                f"Path '{path_str}' resolves to '{resolved_path}' which is outside "
                f"allowed directories (home: {home_dir}, /tmp: {tmp_dir})"
            )
        
        return resolved_path
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get the entire config dict.
        
        Returns:
            Complete configuration dictionary
        """
        return self._config.copy()
    
    def has_config_file(self) -> bool:
        """Check if any config file was loaded.
        
        Returns:
            True if a config file was loaded
        """
        return self._user_config_path is not None or self._project_config_path is not None
    
    def get_config_paths(self) -> Dict[str, Optional[Path]]:
        """Get paths to loaded config files.
        
        Returns:
            Dict with 'user' and 'project' keys containing Path objects or None
        """
        return {
            "user": self._user_config_path,
            "project": self._project_config_path,
        }
