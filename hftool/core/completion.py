"""Shell completion utilities for hftool.

Provides custom completers for Click's built-in shell completion system.
Supports bash, zsh, and fish shells.
"""

import os
from typing import List, Optional

import click


def get_task_names() -> List[str]:
    """Get list of available task names for completion.
    
    Returns:
        List of task names and aliases
    """
    from hftool.core.registry import TASK_REGISTRY, TASK_ALIASES
    
    tasks = list(TASK_REGISTRY.keys())
    aliases = list(TASK_ALIASES.keys())
    return sorted(set(tasks + aliases))


def get_model_names(task: Optional[str] = None) -> List[str]:
    """Get list of available model names for completion.
    
    Args:
        task: Optional task name to filter models
    
    Returns:
        List of model short names and repo_ids
    """
    from hftool.core.models import MODEL_REGISTRY, get_models_for_task
    from hftool.core.registry import TASK_ALIASES
    
    if task:
        # Resolve alias
        task = TASK_ALIASES.get(task, task)
        if task in MODEL_REGISTRY:
            models_dict = get_models_for_task(task)
            return list(models_dict.keys())
    
    # All models
    all_models = []
    for task_models in MODEL_REGISTRY.values():
        all_models.extend(task_models.keys())
    
    return sorted(set(all_models))


def get_device_options() -> List[str]:
    """Get list of device options for completion.
    
    Returns:
        List of device names
    """
    return ["auto", "cuda", "mps", "cpu", "cuda:0", "cuda:1"]


def get_dtype_options() -> List[str]:
    """Get list of dtype options for completion.
    
    Returns:
        List of dtype names
    """
    return ["float32", "float16", "bfloat16"]


def get_shell_name() -> Optional[str]:
    """Detect the current shell.
    
    Returns:
        Shell name ("bash", "zsh", "fish") or None if unknown
    """
    # Check SHELL environment variable
    shell_path = os.environ.get("SHELL", "")
    if shell_path:
        shell = os.path.basename(shell_path)
        if shell in ("bash", "zsh", "fish"):
            return shell
    
    # Check if running in specific shell
    if os.environ.get("ZSH_VERSION"):
        return "zsh"
    elif os.environ.get("BASH_VERSION"):
        return "bash"
    elif os.environ.get("FISH_VERSION"):
        return "fish"
    
    return None


def get_completion_script(shell: str, prog_name: str = "hftool") -> str:
    """Get shell completion activation script.
    
    Args:
        shell: Shell name ("bash", "zsh", "fish")
        prog_name: Program name (default: "hftool")
    
    Returns:
        Shell-specific completion activation script
    
    Raises:
        ValueError: If shell is not supported
    """
    if shell == "bash":
        return f"""# hftool completion for bash
# Add this to your ~/.bashrc:
eval "$(_HFTOOL_COMPLETE=bash_source {prog_name})"
"""
    elif shell == "zsh":
        return f"""# hftool completion for zsh
# Add this to your ~/.zshrc:
eval "$(_HFTOOL_COMPLETE=zsh_source {prog_name})"
"""
    elif shell == "fish":
        return f"""# hftool completion for fish
# Add this to your ~/.config/fish/config.fish:
eval (env _HFTOOL_COMPLETE=fish_source {prog_name})
"""
    else:
        raise ValueError(f"Unsupported shell: {shell}")


def install_completion(shell: str, prog_name: str = "hftool") -> bool:
    """Install shell completion for the current user.
    
    Args:
        shell: Shell name ("bash", "zsh", "fish")
        prog_name: Program name (default: "hftool")
    
    Returns:
        True if installation succeeded
    
    Raises:
        ValueError: If shell is not supported
    """
    import subprocess
    from pathlib import Path
    
    if shell not in ("bash", "zsh", "fish"):
        raise ValueError(f"Unsupported shell: {shell}")
    
    # Get completion script
    script = get_completion_script(shell, prog_name)
    
    # Determine config file
    home = Path.home()
    if shell == "bash":
        config_file = home / ".bashrc"
    elif shell == "zsh":
        config_file = home / ".zshrc"
    elif shell == "fish":
        config_file = home / ".config" / "fish" / "config.fish"
    else:
        raise ValueError(f"Unsupported shell: {shell}")
    
    # Check if already installed
    marker = f"# hftool completion for {shell}"
    if config_file.exists():
        content = config_file.read_text()
        if marker in content:
            return False  # Already installed
    
    # Create parent directory if needed (for fish)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Append completion script
    with config_file.open("a") as f:
        f.write("\n")
        f.write(script)
        f.write("\n")
    
    return True


# Custom completers for Click commands

class TaskCompleter:
    """Custom completer for task names."""
    
    def __call__(self, ctx: click.Context, param: click.Parameter, incomplete: str) -> List[str]:
        """Complete task names.
        
        Args:
            ctx: Click context
            param: Parameter being completed
            incomplete: Incomplete value being typed
        
        Returns:
            List of matching task names
        """
        tasks = get_task_names()
        return [t for t in tasks if t.startswith(incomplete)]


class ModelCompleter:
    """Custom completer for model names."""
    
    def __call__(self, ctx: click.Context, param: click.Parameter, incomplete: str) -> List[str]:
        """Complete model names.
        
        Args:
            ctx: Click context
            param: Parameter being completed
            incomplete: Incomplete value being typed
        
        Returns:
            List of matching model names
        """
        # Try to get task from context
        task = ctx.params.get("task")
        models = get_model_names(task)
        return [m for m in models if m.startswith(incomplete)]


class DeviceCompleter:
    """Custom completer for device names."""
    
    def __call__(self, ctx: click.Context, param: click.Parameter, incomplete: str) -> List[str]:
        """Complete device names.
        
        Args:
            ctx: Click context
            param: Parameter being completed
            incomplete: Incomplete value being typed
        
        Returns:
            List of matching device names
        """
        devices = get_device_options()
        return [d for d in devices if d.startswith(incomplete)]


class DtypeCompleter:
    """Custom completer for dtype names."""
    
    def __call__(self, ctx: click.Context, param: click.Parameter, incomplete: str) -> List[str]:
        """Complete dtype names.
        
        Args:
            ctx: Click context
            param: Parameter being completed
            incomplete: Incomplete value being typed
        
        Returns:
            List of matching dtype names
        """
        dtypes = get_dtype_options()
        return [d for d in dtypes if d.startswith(incomplete)]


class FilePickerCompleter:
    """Custom completer for @ file picker syntax."""
    
    def __call__(self, ctx: click.Context, param: click.Parameter, incomplete: str) -> List[str]:
        """Complete @ file picker syntax.
        
        Args:
            ctx: Click context
            param: Parameter being completed
            incomplete: Incomplete value being typed
        
        Returns:
            List of matching file picker options
        """
        if not incomplete.startswith("@"):
            return []
        
        # Provide @ syntax options
        options = [
            "@",      # Interactive picker
            "@?",     # Fuzzy search
            "@.",     # Current directory
            "@~",     # Home directory
            "@@",     # Recent files
        ]
        
        return [o for o in options if o.startswith(incomplete)]
