"""Parameter schemas for hftool tasks.

Defines schemas for tasks that require structured JSON input,
with types, validation, and interactive prompting support.
"""

from enum import Enum, auto
from typing import Any, Optional, List, Dict, Union
from dataclasses import dataclass


class ParameterType(Enum):
    """Types of parameters."""
    STRING = auto()
    INTEGER = auto()
    FLOAT = auto()
    BOOLEAN = auto()
    FILE_PATH = auto()
    CHOICE = auto()  # Select from list of choices


@dataclass
class ParameterSchema:
    """Schema for a single parameter."""
    
    name: str
    type: ParameterType
    description: str
    required: bool = True
    default: Optional[Any] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    choices: Optional[List[Any]] = None
    file_extensions: Optional[List[str]] = None
    
    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate a value against this schema.
        
        Args:
            value: Value to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Required check
        if self.required and value is None:
            return False, f"Parameter '{self.name}' is required"
        
        if value is None:
            return True, None
        
        # Type validation
        if self.type == ParameterType.STRING:
            if not isinstance(value, str):
                return False, f"Parameter '{self.name}' must be a string"
        
        elif self.type == ParameterType.INTEGER:
            if not isinstance(value, int):
                return False, f"Parameter '{self.name}' must be an integer"
            
            if self.min_value is not None and value < self.min_value:
                return False, f"Parameter '{self.name}' must be >= {self.min_value}"
            
            if self.max_value is not None and value > self.max_value:
                return False, f"Parameter '{self.name}' must be <= {self.max_value}"
        
        elif self.type == ParameterType.FLOAT:
            if not isinstance(value, (int, float)):
                return False, f"Parameter '{self.name}' must be a number"
            
            if self.min_value is not None and value < self.min_value:
                return False, f"Parameter '{self.name}' must be >= {self.min_value}"
            
            if self.max_value is not None and value > self.max_value:
                return False, f"Parameter '{self.name}' must be <= {self.max_value}"
        
        elif self.type == ParameterType.BOOLEAN:
            if not isinstance(value, bool):
                return False, f"Parameter '{self.name}' must be a boolean"
        
        elif self.type == ParameterType.FILE_PATH:
            if not isinstance(value, str):
                return False, f"Parameter '{self.name}' must be a file path string"
            
            # Validate file extension if specified
            if self.file_extensions:
                from pathlib import Path
                ext = Path(value).suffix.lower()
                if ext not in self.file_extensions:
                    return False, f"Parameter '{self.name}' must have extension: {', '.join(self.file_extensions)}"
        
        elif self.type == ParameterType.CHOICE:
            if self.choices and value not in self.choices:
                return False, f"Parameter '{self.name}' must be one of: {', '.join(map(str, self.choices))}"
        
        return True, None


@dataclass
class TaskParameterSchema:
    """Complete parameter schema for a task."""
    
    task_name: str
    description: str
    parameters: List[ParameterSchema]
    
    def validate(self, data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate a complete data dict against this schema.
        
        Args:
            data: Dictionary of parameter values
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate each parameter
        for param in self.parameters:
            value = data.get(param.name)
            is_valid, error_msg = param.validate(value)
            
            if not is_valid and error_msg:
                errors.append(error_msg)
        
        # Check for unexpected parameters
        known_params = {p.name for p in self.parameters}
        for key in data.keys():
            if key not in known_params:
                errors.append(f"Unknown parameter: '{key}'")
        
        return len(errors) == 0, errors
    
    def get_parameter(self, name: str) -> Optional[ParameterSchema]:
        """Get parameter schema by name.
        
        Args:
            name: Parameter name
        
        Returns:
            ParameterSchema or None
        """
        for param in self.parameters:
            if param.name == name:
                return param
        return None


# =============================================================================
# Task-specific schemas
# =============================================================================

# Image-to-Image schema
IMAGE_TO_IMAGE_SCHEMA = TaskParameterSchema(
    task_name="image-to-image",
    description="Transform an image using AI guidance",
    parameters=[
        ParameterSchema(
            name="image",
            type=ParameterType.FILE_PATH,
            description="Input image file path",
            required=True,
            file_extensions=[".jpg", ".jpeg", ".png", ".webp", ".bmp"],
        ),
        ParameterSchema(
            name="prompt",
            type=ParameterType.STRING,
            description="Text prompt describing desired changes",
            required=True,
        ),
        ParameterSchema(
            name="negative_prompt",
            type=ParameterType.STRING,
            description="What to avoid in the output",
            required=False,
            default="",
        ),
        ParameterSchema(
            name="strength",
            type=ParameterType.FLOAT,
            description="How much to transform the image (0.0-1.0)",
            required=False,
            default=0.75,
            min_value=0.0,
            max_value=1.0,
        ),
        ParameterSchema(
            name="guidance_scale",
            type=ParameterType.FLOAT,
            description="How closely to follow the prompt (1.0-20.0)",
            required=False,
            default=7.5,
            min_value=1.0,
            max_value=20.0,
        ),
        ParameterSchema(
            name="num_inference_steps",
            type=ParameterType.INTEGER,
            description="Number of denoising steps (10-100)",
            required=False,
            default=50,
            min_value=10,
            max_value=100,
        ),
    ],
)

# Image-to-Video schema
IMAGE_TO_VIDEO_SCHEMA = TaskParameterSchema(
    task_name="image-to-video",
    description="Generate video from image using AI",
    parameters=[
        ParameterSchema(
            name="image",
            type=ParameterType.FILE_PATH,
            description="Input image file path",
            required=True,
            file_extensions=[".jpg", ".jpeg", ".png", ".webp"],
        ),
        ParameterSchema(
            name="prompt",
            type=ParameterType.STRING,
            description="Text prompt describing desired motion/animation",
            required=True,
        ),
        ParameterSchema(
            name="negative_prompt",
            type=ParameterType.STRING,
            description="What to avoid in the video",
            required=False,
            default="",
        ),
        ParameterSchema(
            name="num_frames",
            type=ParameterType.INTEGER,
            description="Number of frames to generate (8-120)",
            required=False,
            default=49,
            min_value=8,
            max_value=120,
        ),
        ParameterSchema(
            name="num_inference_steps",
            type=ParameterType.INTEGER,
            description="Number of denoising steps (10-100)",
            required=False,
            default=50,
            min_value=10,
            max_value=100,
        ),
        ParameterSchema(
            name="guidance_scale",
            type=ParameterType.FLOAT,
            description="How closely to follow the prompt (1.0-20.0)",
            required=False,
            default=6.0,
            min_value=1.0,
            max_value=20.0,
        ),
    ],
)

# Multi-image input schema (for tasks that accept multiple images)
MULTI_IMAGE_INPUT_SCHEMA = TaskParameterSchema(
    task_name="multi-image-input",
    description="Process multiple images",
    parameters=[
        ParameterSchema(
            name="images",
            type=ParameterType.STRING,  # Will be parsed as list
            description="List of image file paths",
            required=True,
        ),
        ParameterSchema(
            name="prompt",
            type=ParameterType.STRING,
            description="Text prompt for processing",
            required=True,
        ),
    ],
)


# Registry of all task schemas
TASK_SCHEMAS: Dict[str, TaskParameterSchema] = {
    "image-to-image": IMAGE_TO_IMAGE_SCHEMA,
    "i2i": IMAGE_TO_IMAGE_SCHEMA,
    "image-to-video": IMAGE_TO_VIDEO_SCHEMA,
    "i2v": IMAGE_TO_VIDEO_SCHEMA,
}


def get_task_schema(task_name: str) -> Optional[TaskParameterSchema]:
    """Get parameter schema for a task.
    
    Args:
        task_name: Task name
    
    Returns:
        TaskParameterSchema or None if task doesn't have a schema
    """
    return TASK_SCHEMAS.get(task_name)


def validate_task_input(task_name: str, data: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Validate input data for a task.
    
    Args:
        task_name: Task name
        data: Input data dictionary
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    schema = get_task_schema(task_name)
    
    if schema is None:
        # No schema defined - allow all input
        return True, []
    
    return schema.validate(data)
