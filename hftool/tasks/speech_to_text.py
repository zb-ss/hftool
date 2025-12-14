"""Speech-to-text task handler.

Supports Whisper and other ASR models via transformers.
"""

from typing import Any, Dict, Optional
from hftool.tasks.base import BaseTask, AudioInputMixin
from hftool.io.output_handler import save_text


class SpeechToTextTask(AudioInputMixin, BaseTask):
    """Handler for speech-to-text (ASR) using transformers.
    
    Supported models:
    - Whisper (openai/whisper-large-v3, whisper-medium, whisper-small)
    - And other ASR models compatible with transformers pipeline
    
    Note: Step-Audio-R1 requires custom vLLM backend and is documented
    as "advanced/experimental" in the README.
    """
    
    # Model-specific default configurations
    MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
        "whisper-large-v3": {
            "chunk_length_s": 30,
            "batch_size": 24,
        },
        "whisper-large": {
            "chunk_length_s": 30,
            "batch_size": 16,
        },
        "whisper-medium": {
            "chunk_length_s": 30,
            "batch_size": 16,
        },
        "whisper-small": {
            "chunk_length_s": 30,
            "batch_size": 16,
        },
    }
    
    def __init__(self, device: str = "auto", dtype: Optional[str] = None):
        super().__init__(device, dtype)
        self._model_name: Optional[str] = None
    
    def _get_model_config(self, model: str) -> Dict[str, Any]:
        """Get default config for a specific model."""
        model_lower = model.lower()
        
        for key, config in self.MODEL_CONFIGS.items():
            if key.lower() in model_lower:
                return config.copy()
        
        return {"chunk_length_s": 30}
    
    def load_pipeline(self, model: str, **kwargs) -> Any:
        """Load an ASR pipeline.
        
        Args:
            model: HuggingFace model name or local path
            **kwargs: Additional arguments for loading
        
        Returns:
            Loaded transformers pipeline
        """
        from hftool.utils.deps import check_dependencies
        check_dependencies(["transformers", "torch"], extra="with_audio")
        
        from hftool.core.device import detect_device, get_optimal_dtype
        from transformers import pipeline
        import torch
        
        self._model_name = model
        
        # Determine device
        device = self.device if self.device != "auto" else detect_device()
        
        # Determine dtype
        if self.dtype:
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(self.dtype, torch.float16)
        else:
            dtype = get_optimal_dtype(device)
        
        # Load pipeline
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            torch_dtype=dtype,
            device=device if device != "mps" else "cpu",  # MPS support varies
            **kwargs
        )
        
        return pipe
    
    def get_default_kwargs(self) -> Dict[str, Any]:
        """Get default inference kwargs based on loaded model."""
        if self._model_name:
            return self._get_model_config(self._model_name)
        return {}
    
    def run_inference(self, pipeline: Any, audio_path: str, **kwargs) -> Dict[str, Any]:
        """Run ASR inference.
        
        Args:
            pipeline: Loaded ASR pipeline
            audio_path: Path to audio file
            **kwargs: Additional inference arguments
                - return_timestamps: Whether to return word/chunk timestamps
                - chunk_length_s: Chunk length for long audio
                - language: Force specific language
        
        Returns:
            Dict with 'text' key and optionally 'chunks' with timestamps
        """
        # Get model-specific defaults and merge with kwargs
        defaults = self.get_default_kwargs()
        inference_kwargs = {**defaults, **kwargs}
        
        # Run inference
        result = pipeline(audio_path, **inference_kwargs)
        
        # Normalize output format
        if isinstance(result, str):
            return {"text": result}
        elif isinstance(result, dict):
            return result
        else:
            return {"text": str(result)}
    
    def save_output(self, result: Dict[str, Any], output_path: str, **kwargs) -> str:
        """Save transcription to file.
        
        Args:
            result: Transcription result dict
            output_path: Path to save transcription
            **kwargs: Additional save arguments
                - format: "text" (default), "json", or "srt"
        
        Returns:
            Path to saved file
        """
        import json
        
        output_format = kwargs.get("format", "text")
        
        if output_format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        elif output_format == "srt" and "chunks" in result:
            # Generate SRT format from timestamps
            srt_content = self._generate_srt(result["chunks"])
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(srt_content)
        else:
            # Plain text
            text = result.get("text", str(result))
            return save_text(text, output_path)
        
        return output_path
    
    def _generate_srt(self, chunks: list) -> str:
        """Generate SRT subtitle format from timestamped chunks."""
        srt_lines = []
        
        for i, chunk in enumerate(chunks, 1):
            if "timestamp" in chunk:
                start, end = chunk["timestamp"]
                start_str = self._format_srt_time(start)
                end_str = self._format_srt_time(end)
            else:
                continue
            
            text = chunk.get("text", "").strip()
            
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start_str} --> {end_str}")
            srt_lines.append(text)
            srt_lines.append("")
        
        return "\n".join(srt_lines)
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format seconds to SRT timestamp format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def create_task(device: str = "auto", dtype: Optional[str] = None) -> SpeechToTextTask:
    """Factory function to create a SpeechToTextTask.
    
    Args:
        device: Device to run on ("auto", "cuda", "mps", "cpu")
        dtype: Data type ("bfloat16", "float16", "float32")
    
    Returns:
        Configured SpeechToTextTask instance
    """
    return SpeechToTextTask(device=device, dtype=dtype)
