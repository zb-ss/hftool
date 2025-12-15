"""Text-to-speech task handler.

Supports Bark, MMS-TTS, and GLM-TTS (via external wrapper).
"""

import os
import subprocess
from typing import Any, Dict, Optional
from hftool.tasks.base import BaseTask, TextInputMixin
from hftool.io.output_handler import save_audio


class TextToSpeechTask(TextInputMixin, BaseTask):
    """Handler for text-to-speech synthesis.
    
    Supported models:
    - Bark (suno/bark, suno/bark-small) - default, high quality
    - MMS-TTS (facebook/mms-tts-*) - lightweight, multilingual
    - GLM-TTS (zai-org/GLM-TTS) - requires external setup
    """
    
    # Model-specific default configurations
    MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
        "bark": {
            "sample_rate": 24000,
        },
        "mms-tts": {
            "sample_rate": 16000,
        },
    }
    
    def __init__(self, device: str = "auto", dtype: Optional[str] = None):
        super().__init__(device, dtype)
        self._model_name: Optional[str] = None
        self._sample_rate: int = 24000
    
    def _get_model_config(self, model: str) -> Dict[str, Any]:
        """Get default config for a specific model."""
        model_lower = model.lower()
        
        for key, config in self.MODEL_CONFIGS.items():
            if key.lower() in model_lower:
                return config.copy()
        
        return {"sample_rate": 24000}
    
    def load_pipeline(self, model: str, **kwargs) -> Any:
        """Load a TTS pipeline.
        
        Args:
            model: HuggingFace model name or local path
            **kwargs: Additional arguments for loading
        
        Returns:
            Loaded pipeline or model components
        """
        from hftool.utils.deps import check_dependencies
        check_dependencies(["transformers", "torch"], extra="with_tts")
        
        from hftool.core.device import detect_device, get_optimal_dtype
        
        self._model_name = model
        model_lower = model.lower()
        
        # Determine device
        device = self.device if self.device != "auto" else detect_device()
        
        # Get model config
        config = self._get_model_config(model)
        self._sample_rate = config.get("sample_rate", 24000)
        
        # Handle GLM-TTS separately (requires external setup)
        if "glm-tts" in model_lower:
            return self._load_glmtts(model)
        
        # Try transformers pipeline for standard models
        try:
            from transformers import pipeline
            
            pipe = pipeline(
                "text-to-speech",
                model=model,
                device=device if device != "mps" else -1,  # MPS needs special handling
                **kwargs
            )
            return pipe
        except Exception as e:
            # If pipeline fails, try loading components directly
            return self._load_model_components(model, device, **kwargs)
    
    def _load_model_components(self, model: str, device: str, **kwargs) -> Dict[str, Any]:
        """Load model components directly for models that don't support pipeline."""
        from transformers import AutoProcessor, AutoModel
        import torch
        
        if self.dtype:
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(self.dtype, torch.float32)
        else:
            dtype = torch.float32
        
        processor = AutoProcessor.from_pretrained(model, **kwargs)
        model_obj = AutoModel.from_pretrained(model, torch_dtype=dtype, **kwargs)
        
        if device != "cpu":
            model_obj = model_obj.to(device)
        
        return {
            "type": "components",
            "processor": processor,
            "model": model_obj,
            "device": device,
        }
    
    def _load_glmtts(self, model: str) -> Dict[str, Any]:
        """Load GLM-TTS (requires external setup).
        
        GLM-TTS requires cloning their repository and installing dependencies.
        """
        glmtts_path = os.environ.get("GLMTTS_PATH", "./GLM-TTS")
        
        if not os.path.exists(glmtts_path):
            raise RuntimeError(
                "GLM-TTS requires manual setup. Please run:\n"
                "  git clone https://github.com/zai-org/GLM-TTS.git\n"
                "  cd GLM-TTS && pip install -r requirements.txt\n"
                "Then set GLMTTS_PATH environment variable to the path."
            )
        
        return {
            "type": "glmtts",
            "path": glmtts_path,
        }
    
    def get_default_kwargs(self) -> Dict[str, Any]:
        """Get default inference kwargs."""
        return {}
    
    def run_inference(self, pipeline: Any, text: str, **kwargs) -> Any:
        """Run TTS inference.
        
        Args:
            pipeline: Loaded TTS pipeline or components
            text: Text to synthesize
            **kwargs: Additional inference arguments
        
        Returns:
            Audio data (format depends on model)
        """
        # Handle GLM-TTS
        if isinstance(pipeline, dict) and pipeline.get("type") == "glmtts":
            return self._run_glmtts(pipeline, text, **kwargs)
        
        # Handle component-based loading
        if isinstance(pipeline, dict) and pipeline.get("type") == "components":
            return self._run_components(pipeline, text, **kwargs)
        
        # Standard transformers pipeline
        result = pipeline(text, **kwargs)
        
        # Normalize output format
        if isinstance(result, dict):
            return result
        elif hasattr(result, "audio"):
            return {"audio": result.audio, "sampling_rate": self._sample_rate}
        else:
            return {"audio": result, "sampling_rate": self._sample_rate}
    
    def _run_components(self, components: Dict[str, Any], text: str, **kwargs) -> Dict[str, Any]:
        """Run inference with component-based model."""
        import torch
        
        processor = components["processor"]
        model = components["model"]
        device = components["device"]
        
        inputs = processor(text=text, return_tensors="pt")
        
        if device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output = model.generate(**inputs, **kwargs)
        
        # Convert to numpy
        if hasattr(output, "cpu"):
            audio = output.cpu().numpy()
        else:
            audio = output
        
        return {"audio": audio, "sampling_rate": self._sample_rate}
    
    def _run_glmtts(self, pipeline: Dict[str, Any], text: str, **kwargs) -> Dict[str, Any]:
        """Run GLM-TTS inference via their script."""
        import tempfile
        import json
        
        glmtts_path = pipeline["path"]
        
        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"text": text}, f)
            input_file = f.name
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as output_dir:
            # Run GLM-TTS inference script
            cmd = [
                "python", os.path.join(glmtts_path, "glmtts_inference.py"),
                "--text", text,
                "--output_dir", output_dir,
            ]
            
            try:
                subprocess.run(cmd, cwd=glmtts_path, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"GLM-TTS inference failed: {e.stderr.decode()}")
            
            # Find output audio file
            audio_files = [f for f in os.listdir(output_dir) if f.endswith(".wav")]
            if not audio_files:
                raise RuntimeError("GLM-TTS did not produce output audio")
            
            # Load the audio
            from hftool.io.input_loader import load_audio_array
            audio, sr = load_audio_array(os.path.join(output_dir, audio_files[0]))
            
            return {"audio": audio, "sampling_rate": sr}
    
    def save_output(self, result: Any, output_path: str, **kwargs) -> str:
        """Save generated audio to file.
        
        Args:
            result: Audio data from inference
            output_path: Path to save audio
            **kwargs: Additional save arguments
        
        Returns:
            Path to saved file
        """
        sample_rate = kwargs.pop("sample_rate", self._sample_rate)
        
        # Handle different result formats
        if isinstance(result, dict):
            sample_rate = result.get("sampling_rate", result.get("sample_rate", sample_rate))
        
        return save_audio(result, output_path, sample_rate=sample_rate, **kwargs)


def create_task(device: str = "auto", dtype: Optional[str] = None) -> TextToSpeechTask:
    """Factory function to create a TextToSpeechTask.
    
    Args:
        device: Device to run on ("auto", "cuda", "mps", "cpu")
        dtype: Data type ("bfloat16", "float16", "float32")
    
    Returns:
        Configured TextToSpeechTask instance
    """
    return TextToSpeechTask(device=device, dtype=dtype)
