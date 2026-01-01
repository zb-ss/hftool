"""Tests for interactive input and parameter schemas."""

import pytest

from hftool.core.parameters import (
    ParameterType,
    ParameterSchema,
    TaskParameterSchema,
    get_task_schema,
    validate_task_input,
    IMAGE_TO_IMAGE_SCHEMA,
)


class TestParameterSchema:
    """Tests for ParameterSchema class."""
    
    def test_string_validation(self):
        """Test string parameter validation."""
        param = ParameterSchema(
            name="test",
            type=ParameterType.STRING,
            description="Test parameter",
            required=True,
        )
        
        is_valid, error = param.validate("test value")
        assert is_valid
        assert error is None
        
        is_valid, error = param.validate(123)
        assert not is_valid
        assert "must be a string" in error
    
    def test_integer_validation(self):
        """Test integer parameter validation."""
        param = ParameterSchema(
            name="steps",
            type=ParameterType.INTEGER,
            description="Number of steps",
            required=True,
            min_value=1,
            max_value=100,
        )
        
        is_valid, error = param.validate(50)
        assert is_valid
        
        is_valid, error = param.validate(0)
        assert not is_valid
        assert ">= 1" in error
        
        is_valid, error = param.validate(101)
        assert not is_valid
        assert "<= 100" in error
    
    def test_float_validation(self):
        """Test float parameter validation."""
        param = ParameterSchema(
            name="strength",
            type=ParameterType.FLOAT,
            description="Strength",
            required=True,
            min_value=0.0,
            max_value=1.0,
        )
        
        is_valid, error = param.validate(0.5)
        assert is_valid
        
        is_valid, error = param.validate(-0.1)
        assert not is_valid
        
        is_valid, error = param.validate(1.5)
        assert not is_valid
    
    def test_boolean_validation(self):
        """Test boolean parameter validation."""
        param = ParameterSchema(
            name="flag",
            type=ParameterType.BOOLEAN,
            description="A flag",
            required=False,
        )
        
        is_valid, error = param.validate(True)
        assert is_valid
        
        is_valid, error = param.validate(False)
        assert is_valid
        
        is_valid, error = param.validate("true")
        assert not is_valid
    
    def test_choice_validation(self):
        """Test choice parameter validation."""
        param = ParameterSchema(
            name="mode",
            type=ParameterType.CHOICE,
            description="Mode selection",
            required=True,
            choices=["fast", "balanced", "quality"],
        )
        
        is_valid, error = param.validate("fast")
        assert is_valid
        
        is_valid, error = param.validate("invalid")
        assert not is_valid
        assert "must be one of" in error
    
    def test_file_path_validation(self):
        """Test file path parameter validation."""
        param = ParameterSchema(
            name="image",
            type=ParameterType.FILE_PATH,
            description="Image file",
            required=True,
            file_extensions=[".png", ".jpg"],
        )
        
        is_valid, error = param.validate("test.png")
        assert is_valid
        
        is_valid, error = param.validate("test.txt")
        assert not is_valid
        assert "extension" in error
    
    def test_required_validation(self):
        """Test required parameter validation."""
        param = ParameterSchema(
            name="required_param",
            type=ParameterType.STRING,
            description="Required",
            required=True,
        )
        
        is_valid, error = param.validate(None)
        assert not is_valid
        assert "required" in error
    
    def test_optional_validation(self):
        """Test optional parameter validation."""
        param = ParameterSchema(
            name="optional_param",
            type=ParameterType.STRING,
            description="Optional",
            required=False,
        )
        
        is_valid, error = param.validate(None)
        assert is_valid


class TestTaskParameterSchema:
    """Tests for TaskParameterSchema class."""
    
    def test_schema_validation_success(self):
        """Test successful schema validation."""
        data = {
            "image": "test.png",
            "prompt": "A beautiful sunset",
            "strength": 0.75,
            "num_inference_steps": 50,
        }
        
        is_valid, errors = IMAGE_TO_IMAGE_SCHEMA.validate(data)
        assert is_valid
        assert len(errors) == 0
    
    def test_schema_validation_missing_required(self):
        """Test validation with missing required field."""
        data = {
            "image": "test.png",
            # Missing required 'prompt'
        }
        
        is_valid, errors = IMAGE_TO_IMAGE_SCHEMA.validate(data)
        assert not is_valid
        assert any("prompt" in err and "required" in err for err in errors)
    
    def test_schema_validation_invalid_value(self):
        """Test validation with invalid value."""
        data = {
            "image": "test.png",
            "prompt": "Test",
            "strength": 2.0,  # Invalid: should be 0.0-1.0
        }
        
        is_valid, errors = IMAGE_TO_IMAGE_SCHEMA.validate(data)
        assert not is_valid
        assert any("strength" in err for err in errors)
    
    def test_schema_validation_unknown_parameter(self):
        """Test validation with unknown parameter."""
        data = {
            "image": "test.png",
            "prompt": "Test",
            "unknown_param": "value",
        }
        
        is_valid, errors = IMAGE_TO_IMAGE_SCHEMA.validate(data)
        assert not is_valid
        assert any("unknown_param" in err for err in errors)
    
    def test_get_parameter(self):
        """Test getting parameter by name."""
        param = IMAGE_TO_IMAGE_SCHEMA.get_parameter("prompt")
        assert param is not None
        assert param.name == "prompt"
        assert param.type == ParameterType.STRING
        
        param = IMAGE_TO_IMAGE_SCHEMA.get_parameter("nonexistent")
        assert param is None


class TestTaskSchemaRegistry:
    """Tests for task schema registry."""
    
    def test_get_task_schema_image_to_image(self):
        """Test getting image-to-image schema."""
        schema = get_task_schema("image-to-image")
        assert schema is not None
        assert schema.task_name == "image-to-image"
        assert len(schema.parameters) > 0
    
    def test_get_task_schema_alias(self):
        """Test getting schema by alias."""
        schema = get_task_schema("i2i")
        assert schema is not None
        assert schema.task_name == "image-to-image"
    
    def test_get_task_schema_no_schema(self):
        """Test getting schema for task without schema."""
        schema = get_task_schema("text-to-image")
        assert schema is None
    
    def test_validate_task_input(self):
        """Test task input validation."""
        data = {
            "image": "test.png",
            "prompt": "Test prompt",
        }
        
        is_valid, errors = validate_task_input("image-to-image", data)
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_task_input_no_schema(self):
        """Test validation for task without schema (should pass)."""
        data = {"anything": "goes"}
        
        is_valid, errors = validate_task_input("text-to-image", data)
        assert is_valid
        assert len(errors) == 0
    
    def test_image_to_image_schema_complete(self):
        """Test that image-to-image schema has all expected parameters."""
        schema = IMAGE_TO_IMAGE_SCHEMA
        
        param_names = {p.name for p in schema.parameters}
        
        assert "image" in param_names
        assert "prompt" in param_names
        assert "negative_prompt" in param_names
        assert "strength" in param_names
        assert "guidance_scale" in param_names
        assert "num_inference_steps" in param_names
    
    def test_parameter_defaults(self):
        """Test parameter default values."""
        strength_param = IMAGE_TO_IMAGE_SCHEMA.get_parameter("strength")
        assert strength_param.default == 0.75
        
        negative_prompt_param = IMAGE_TO_IMAGE_SCHEMA.get_parameter("negative_prompt")
        assert negative_prompt_param.default == ""
        assert not negative_prompt_param.required
