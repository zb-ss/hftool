"""Interactive JSON input builder for hftool.

Provides guided prompts for building structured JSON input for complex tasks.
Uses InquirerPy when available, falls back to click-based prompts.
"""

import json
import click
from typing import Any, Dict, Optional
from pathlib import Path

from hftool.core.parameters import ParameterSchema, ParameterType, TaskParameterSchema, get_task_schema
from hftool.io.file_picker import FilePicker, FileType


class InteractiveInputBuilder:
    """Build JSON input interactively."""
    
    def __init__(self, task_name: str):
        """Initialize builder.
        
        Args:
            task_name: Task name to build input for
        
        Raises:
            ValueError: If task doesn't have a schema
        """
        self.task_name = task_name
        self.schema = get_task_schema(task_name)
        
        if self.schema is None:
            raise ValueError(
                f"Task '{task_name}' does not have an interactive schema. "
                f"Use plain text input or JSON directly."
            )
        
        self._inquirer_available = False
        
        # Try to import InquirerPy
        try:
            from InquirerPy import inquirer
            self._inquirer = inquirer
            self._inquirer_available = True
        except ImportError:
            pass
    
    def build(self) -> Dict[str, Any]:
        """Build input dictionary interactively.
        
        Returns:
            Dictionary of parameter values
        
        Raises:
            ValueError: If user cancels or input is invalid
        """
        click.echo("")
        click.echo(f"=== Interactive Input Builder: {self.schema.description} ===")
        click.echo("")
        
        data = {}
        
        # Prompt for each parameter
        for param in self.schema.parameters:
            value = self._prompt_parameter(param)
            
            # Only include non-None values or required params
            if value is not None or param.required:
                data[param.name] = value
        
        # Show final JSON
        click.echo("")
        click.echo("=== Final Input ===")
        click.echo(json.dumps(data, indent=2))
        click.echo("")
        
        # Confirm
        if not click.confirm("Use this input?", default=True):
            raise ValueError("Input cancelled by user")
        
        # Validate
        is_valid, errors = self.schema.validate(data)
        if not is_valid:
            click.echo("Validation errors:", err=True)
            for error in errors:
                click.echo(f"  - {error}", err=True)
            raise ValueError("Input validation failed")
        
        return data
    
    def _prompt_parameter(self, param: ParameterSchema) -> Any:
        """Prompt for a single parameter.
        
        Args:
            param: Parameter schema
        
        Returns:
            Parameter value
        """
        # Show description
        required_str = " (required)" if param.required else " (optional)"
        click.echo(f"{param.name}{required_str}: {param.description}")
        
        # Handle different parameter types
        if param.type == ParameterType.FILE_PATH:
            return self._prompt_file_path(param)
        
        elif param.type == ParameterType.BOOLEAN:
            return self._prompt_boolean(param)
        
        elif param.type == ParameterType.CHOICE:
            return self._prompt_choice(param)
        
        elif param.type == ParameterType.INTEGER:
            return self._prompt_integer(param)
        
        elif param.type == ParameterType.FLOAT:
            return self._prompt_float(param)
        
        elif param.type == ParameterType.STRING:
            return self._prompt_string(param)
        
        else:
            # Fallback to string
            return self._prompt_string(param)
    
    def _prompt_file_path(self, param: ParameterSchema) -> Optional[str]:
        """Prompt for file path with file picker support.
        
        Args:
            param: Parameter schema
        
        Returns:
            File path or None
        """
        # Show @ reference help
        click.echo("  (Tip: Use @ references like @, @?, @@, or enter path directly)")
        
        try:
            value = click.prompt(
                f"  Enter {param.name}",
                default=param.default if param.default else "",
                type=str,
            )
        except click.Abort:
            if param.required:
                raise ValueError("Input cancelled")
            return None
        
        if not value and not param.required:
            return None
        
        if not value and param.required:
            click.echo("  Error: This parameter is required", err=True)
            return self._prompt_file_path(param)
        
        # Resolve @ references
        if value.startswith("@"):
            # Determine file type from extensions
            file_type = FileType.ALL
            if param.file_extensions:
                if any(ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"] for ext in param.file_extensions):
                    file_type = FileType.IMAGE
                elif any(ext in [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"] for ext in param.file_extensions):
                    file_type = FileType.AUDIO
                elif any(ext in [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"] for ext in param.file_extensions):
                    file_type = FileType.VIDEO
            
            picker = FilePicker(file_type=file_type)
            try:
                value = picker.resolve_reference(value, task=self.task_name)
            except ValueError as e:
                click.echo(f"  Error: {e}", err=True)
                return self._prompt_file_path(param)
        
        # Validate file exists
        if not Path(value).exists():
            click.echo(f"  Error: File not found: {value}", err=True)
            return self._prompt_file_path(param)
        
        # Validate extension
        if param.file_extensions:
            ext = Path(value).suffix.lower()
            if ext not in param.file_extensions:
                click.echo(
                    f"  Error: File must have extension: {', '.join(param.file_extensions)}",
                    err=True
                )
                return self._prompt_file_path(param)
        
        click.echo(f"  ✓ Using: {value}")
        return value
    
    def _prompt_boolean(self, param: ParameterSchema) -> Optional[bool]:
        """Prompt for boolean value.
        
        Args:
            param: Parameter schema
        
        Returns:
            Boolean value or None
        """
        default = param.default if param.default is not None else True
        
        try:
            value = click.confirm(f"  Enable {param.name}?", default=default)
            return value
        except click.Abort:
            if param.required:
                raise ValueError("Input cancelled")
            return None
    
    def _prompt_choice(self, param: ParameterSchema) -> Optional[Any]:
        """Prompt for choice from list.
        
        Args:
            param: Parameter schema
        
        Returns:
            Selected choice or None
        """
        if not param.choices:
            return self._prompt_string(param)
        
        # Use InquirerPy if available
        if self._inquirer_available:
            try:
                value = self._inquirer.select(
                    message=f"  Select {param.name}:",
                    choices=param.choices,
                    default=param.default if param.default else param.choices[0],
                ).execute()
                return value
            except KeyboardInterrupt:
                if param.required:
                    raise ValueError("Input cancelled")
                return None
            except Exception:
                # Fall back to click
                pass
        
        # Click-based selection
        click.echo(f"  Choices:")
        for i, choice in enumerate(param.choices, 1):
            default_marker = " (default)" if choice == param.default else ""
            click.echo(f"    [{i}] {choice}{default_marker}")
        
        try:
            selection = click.prompt(
                f"  Select {param.name} [1-{len(param.choices)}]",
                type=int,
                default=1,
            )
            
            if 1 <= selection <= len(param.choices):
                return param.choices[selection - 1]
            else:
                click.echo("  Invalid selection", err=True)
                return self._prompt_choice(param)
        
        except click.Abort:
            if param.required:
                raise ValueError("Input cancelled")
            return None
    
    def _prompt_integer(self, param: ParameterSchema) -> Optional[int]:
        """Prompt for integer value.
        
        Args:
            param: Parameter schema
        
        Returns:
            Integer value or None
        """
        # Build range description
        range_str = ""
        if param.min_value is not None and param.max_value is not None:
            range_str = f" [{param.min_value}-{param.max_value}]"
        elif param.min_value is not None:
            range_str = f" [>={param.min_value}]"
        elif param.max_value is not None:
            range_str = f" [<={param.max_value}]"
        
        try:
            value = click.prompt(
                f"  Enter {param.name}{range_str}",
                type=int,
                default=param.default if param.default is not None else None,
            )
            
            # Validate range
            if param.min_value is not None and value < param.min_value:
                click.echo(f"  Error: Value must be >= {param.min_value}", err=True)
                return self._prompt_integer(param)
            
            if param.max_value is not None and value > param.max_value:
                click.echo(f"  Error: Value must be <= {param.max_value}", err=True)
                return self._prompt_integer(param)
            
            return value
        
        except click.Abort:
            if param.required:
                raise ValueError("Input cancelled")
            return None
    
    def _prompt_float(self, param: ParameterSchema) -> Optional[float]:
        """Prompt for float value.
        
        Args:
            param: Parameter schema
        
        Returns:
            Float value or None
        """
        # Build range description
        range_str = ""
        if param.min_value is not None and param.max_value is not None:
            range_str = f" [{param.min_value}-{param.max_value}]"
        elif param.min_value is not None:
            range_str = f" [>={param.min_value}]"
        elif param.max_value is not None:
            range_str = f" [<={param.max_value}]"
        
        try:
            value = click.prompt(
                f"  Enter {param.name}{range_str}",
                type=float,
                default=param.default if param.default is not None else None,
            )
            
            # Validate range
            if param.min_value is not None and value < param.min_value:
                click.echo(f"  Error: Value must be >= {param.min_value}", err=True)
                return self._prompt_float(param)
            
            if param.max_value is not None and value > param.max_value:
                click.echo(f"  Error: Value must be <= {param.max_value}", err=True)
                return self._prompt_float(param)
            
            return value
        
        except click.Abort:
            if param.required:
                raise ValueError("Input cancelled")
            return None
    
    def _prompt_string(self, param: ParameterSchema) -> Optional[str]:
        """Prompt for string value.
        
        Args:
            param: Parameter schema
        
        Returns:
            String value or None
        """
        try:
            value = click.prompt(
                f"  Enter {param.name}",
                type=str,
                default=param.default if param.default else "",
            )
            
            if not value and param.required:
                click.echo("  Error: This parameter is required", err=True)
                return self._prompt_string(param)
            
            return value if value else None
        
        except click.Abort:
            if param.required:
                raise ValueError("Input cancelled")
            return None


def build_interactive_input(task_name: str) -> str:
    """Build JSON input interactively for a task.
    
    Args:
        task_name: Task name
    
    Returns:
        JSON string of input data
    
    Raises:
        ValueError: If task doesn't support interactive input or user cancels
    """
    builder = InteractiveInputBuilder(task_name)
    data = builder.build()
    return json.dumps(data)
