"""Base task class for hftool task handlers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseTask(ABC):
    """Abstract base class for all task handlers.
    
    Each task handler must implement:
    - load_pipeline(): Load the model/pipeline
    - run_inference(): Run inference on input
    - save_output(): Save the result to file
    
    Optionally override:
    - validate_input(): Validate input before inference
    - get_default_kwargs(): Get default inference kwargs
    """
    
    def __init__(self, device: str = "auto", dtype: Optional[str] = None):
        """Initialize the task handler.
        
        Args:
            device: Device to run on ("auto", "cuda", "mps", "cpu")
            dtype: Data type ("bfloat16", "float16", "float32", or None for auto)
        """
        self.device = device
        self.dtype = dtype
        self._pipeline = None
    
    @abstractmethod
    def load_pipeline(self, model: str, **kwargs) -> Any:
        """Load the model/pipeline.
        
        Args:
            model: Model name or path (HuggingFace repo or local path)
            **kwargs: Additional arguments for loading
        
        Returns:
            Loaded pipeline object
        """
        pass
    
    @abstractmethod
    def run_inference(self, pipeline: Any, input_data: Any, **kwargs) -> Any:
        """Run inference on the input data.
        
        Args:
            pipeline: The loaded pipeline
            input_data: Input data (text, image, audio, etc.)
            **kwargs: Additional inference arguments
        
        Returns:
            Inference result
        """
        pass
    
    @abstractmethod
    def save_output(self, result: Any, output_path: str, **kwargs) -> str:
        """Save the inference result to a file.
        
        Args:
            result: Inference result
            output_path: Path to save the result
            **kwargs: Additional arguments (e.g., fps for video, sample_rate for audio)
        
        Returns:
            Path to the saved file
        """
        pass
    
    def validate_input(self, input_data: Any) -> Any:
        """Validate and preprocess input data.
        
        Override this method to add input validation.
        
        Args:
            input_data: Raw input data
        
        Returns:
            Validated/preprocessed input data
        
        Raises:
            ValueError: If input is invalid
        """
        return input_data
    
    def get_default_kwargs(self) -> Dict[str, Any]:
        """Get default inference kwargs for this task.
        
        Override this method to provide task-specific defaults.
        
        Returns:
            Dictionary of default kwargs
        """
        return {}
    
    def execute(
        self,
        model: str,
        input_data: Any,
        output_path: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Execute the full pipeline: load -> validate -> infer -> save.
        
        Args:
            model: Model name or path
            input_data: Input data
            output_path: Optional output path (if None, returns result without saving)
            **kwargs: Additional arguments passed to load/run/save
        
        Returns:
            Inference result (and saves to file if output_path is provided)
        """
        # Extract kwargs for each stage
        load_kwargs = kwargs.pop("load_kwargs", {})
        infer_kwargs = kwargs.pop("infer_kwargs", {})
        save_kwargs = kwargs.pop("save_kwargs", {})
        
        # Merge remaining kwargs into infer_kwargs
        infer_kwargs = {**self.get_default_kwargs(), **infer_kwargs, **kwargs}
        
        # Load pipeline if not already loaded
        if self._pipeline is None:
            self._pipeline = self.load_pipeline(model, **load_kwargs)
        
        # Validate input
        validated_input = self.validate_input(input_data)
        
        # Run inference
        result = self.run_inference(self._pipeline, validated_input, **infer_kwargs)
        
        # Save output if path provided
        if output_path:
            self.save_output(result, output_path, **save_kwargs)
        
        return result
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.cleanup()
        return False
    
    def cleanup(self):
        """Clean up resources (e.g., unload model from GPU).
        
        Override this method to add custom cleanup logic.
        """
        self._pipeline = None


class TextInputMixin:
    """Mixin for tasks that accept text input."""
    
    def validate_input(self, input_data: Any) -> str:
        """Validate text input.
        
        Args:
            input_data: Input that should be text
        
        Returns:
            Validated text string
        
        Raises:
            ValueError: If input is not a valid string
        """
        if not isinstance(input_data, str):
            raise ValueError(f"Expected string input, got {type(input_data).__name__}")
        if not input_data.strip():
            raise ValueError("Input text cannot be empty")
        return input_data.strip()


class ImageInputMixin:
    """Mixin for tasks that accept image input."""
    
    def validate_input(self, input_data: Any) -> Any:
        """Validate image input.
        
        Args:
            input_data: Input that should be an image path or PIL Image
        
        Returns:
            PIL Image object
        
        Raises:
            ValueError: If input is not a valid image
        """
        from hftool.io.input_loader import load_input, InputType
        
        # If it's already a PIL Image, return it
        try:
            from PIL import Image
            if isinstance(input_data, Image.Image):
                return input_data
        except ImportError:
            pass
        
        # Otherwise, try to load it
        if isinstance(input_data, str):
            return load_input(input_data, InputType.IMAGE)
        
        raise ValueError(f"Expected image path or PIL Image, got {type(input_data).__name__}")


class AudioInputMixin:
    """Mixin for tasks that accept audio input."""
    
    def validate_input(self, input_data: Any) -> str:
        """Validate audio input.
        
        Args:
            input_data: Input that should be an audio file path
        
        Returns:
            Path to audio file
        
        Raises:
            ValueError: If input is not a valid audio file
        """
        import os
        
        if not isinstance(input_data, str):
            raise ValueError(f"Expected audio file path, got {type(input_data).__name__}")
        
        if not os.path.exists(input_data):
            raise ValueError(f"Audio file not found: {input_data}")
        
        # Check extension
        valid_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}
        ext = os.path.splitext(input_data)[1].lower()
        if ext not in valid_extensions:
            raise ValueError(f"Unsupported audio format: {ext}. Supported: {valid_extensions}")
        
        return input_data
