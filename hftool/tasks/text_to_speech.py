"""Text-to-speech task handler.

Supports Kokoro, Chatterbox, Bark, and MMS-TTS.
"""

import os
import subprocess
from typing import Any, Dict, Optional
from hftool.tasks.base import BaseTask, TextInputMixin
from hftool.io.output_handler import save_audio


class TextToSpeechTask(TextInputMixin, BaseTask):
    """Handler for text-to-speech synthesis.

    Supported models:
    - Kokoro (hexgrad/Kokoro-82M) - default, lightweight, CPU-capable
    - Chatterbox (ResembleAI/chatterbox) - voice cloning, emotion control
    - Bark (suno/bark, suno/bark-small) - high quality
    - MMS-TTS (facebook/mms-tts-*) - lightweight, multilingual
    """

    # Model-specific default configurations
    MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
        "kokoro": {
            "sample_rate": 24000,
        },
        "chatterbox": {
            "sample_rate": 24000,
            "exaggeration": 0.4,
            "cfg_weight": 0.5,
            "temperature": 0.7,
        },
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
        self._model_type: Optional[str] = None  # "kokoro", "chatterbox", "bark", "mms-tts", etc.
        self._sample_rate: int = 24000

    def _get_model_config(self, model: str) -> Dict[str, Any]:
        """Get default config for a specific model."""
        model_lower = model.lower()

        for key, config in self.MODEL_CONFIGS.items():
            if key.lower() in model_lower:
                return config.copy()

        return {"sample_rate": 24000}

    def _detect_model_type(self, model: str) -> str:
        """Detect model type from model name/path."""
        model_lower = model.lower()
        if "kokoro" in model_lower:
            return "kokoro"
        if "chatterbox" in model_lower:
            return "chatterbox"
        if "mms-tts" in model_lower:
            return "mms-tts"
        if "bark" in model_lower:
            return "bark"
        if "glm-tts" in model_lower:
            return "glmtts"
        return "transformers"

    def load_pipeline(self, model: str, **kwargs) -> Any:
        """Load a TTS pipeline.

        Args:
            model: HuggingFace model name or local path
            **kwargs: Additional arguments for loading
                - voice_ref: Path to reference audio for voice cloning (Chatterbox)

        Returns:
            Loaded pipeline or model components
        """
        from hftool.core.device import configure_rocm_env, detect_device

        configure_rocm_env()

        self._model_name = model
        self._model_type = self._detect_model_type(model)

        # Get model config
        config = self._get_model_config(model)
        self._sample_rate = config.get("sample_rate", 24000)

        # Determine device
        device = self.device if self.device != "auto" else detect_device()

        if self._model_type == "kokoro":
            return self._load_kokoro(device, **kwargs)
        elif self._model_type == "chatterbox":
            return self._load_chatterbox(device, **kwargs)
        elif self._model_type == "glmtts":
            return self._load_glmtts(model)
        else:
            return self._load_transformers(model, device, **kwargs)

    def _load_kokoro(self, device: str, **kwargs) -> Dict[str, Any]:
        """Load Kokoro TTS pipeline."""
        from hftool.utils.deps import check_dependency
        check_dependency("kokoro", extra="with_tts_kokoro")

        from kokoro import KPipeline

        lang = kwargs.get("lang", "a")  # 'a' = American English
        # kokoro 0.9+ renamed 'lang' to 'lang_code'
        import inspect
        sig = inspect.signature(KPipeline.__init__)
        if "lang_code" in sig.parameters:
            pipe = KPipeline(lang_code=lang)
        else:
            pipe = KPipeline(lang=lang)

        return {
            "type": "kokoro",
            "pipeline": pipe,
            "device": device,
        }

    def _load_chatterbox(self, device: str, **kwargs) -> Dict[str, Any]:
        """Load Chatterbox TTS model."""
        from hftool.utils.deps import check_dependency
        check_dependency("chatterbox", extra="with_tts_chatterbox", pip_name="chatterbox-tts")

        from hftool.core.device import configure_rocm_env
        configure_rocm_env()

        from chatterbox.tts import ChatterboxTTS

        model = ChatterboxTTS.from_pretrained(device=device)
        self._sample_rate = model.sr

        voice_ref = kwargs.get("voice_ref")

        return {
            "type": "chatterbox",
            "model": model,
            "device": device,
            "voice_ref": voice_ref,
        }

    def _load_transformers(self, model: str, device: str, **kwargs) -> Any:
        """Load a transformers-based TTS pipeline (Bark, MMS-TTS, etc.)."""
        from hftool.utils.deps import check_dependencies
        check_dependencies(["transformers", "torch"], extra="with_tts")

        try:
            from transformers import pipeline

            pipe = pipeline(
                "text-to-speech",
                model=model,
                device=device if device != "mps" else -1,
                **kwargs
            )
            return pipe
        except Exception:
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
        """Load GLM-TTS (requires external setup)."""
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
                - exaggeration: Chatterbox emotion control (default 0.4)
                - cfg_weight: Chatterbox pacing control (default 0.5)
                - temperature: Chatterbox consistency (default 0.7)
                - voice_ref: Path to reference audio for voice cloning

        Returns:
            Audio data dict with 'audio' and 'sampling_rate' keys
        """
        if isinstance(pipeline, dict):
            pipeline_type = pipeline.get("type", "")
            if pipeline_type == "kokoro":
                return self._run_kokoro(pipeline, text, **kwargs)
            elif pipeline_type == "chatterbox":
                return self._run_chatterbox(pipeline, text, **kwargs)
            elif pipeline_type == "glmtts":
                return self._run_glmtts(pipeline, text, **kwargs)
            elif pipeline_type == "components":
                return self._run_components(pipeline, text, **kwargs)

        # Standard transformers pipeline
        result = pipeline(text, **kwargs)

        if isinstance(result, dict):
            return result
        elif hasattr(result, "audio"):
            return {"audio": result.audio, "sampling_rate": self._sample_rate}
        else:
            return {"audio": result, "sampling_rate": self._sample_rate}

    def _run_kokoro(self, pipeline: Dict[str, Any], text: str, **kwargs) -> Dict[str, Any]:
        """Run Kokoro TTS inference."""
        import numpy as np

        pipe = pipeline["pipeline"]
        voice = kwargs.get("voice", "af_heart")  # Default voice
        speed = kwargs.get("speed", 1.0)

        # Kokoro generates audio in chunks via a generator
        audio_chunks = []
        for chunk in pipe(text, voice=voice, speed=speed):
            if chunk.audio is not None:
                audio_chunks.append(chunk.audio)

        if not audio_chunks:
            raise RuntimeError("Kokoro produced no audio output")

        audio = np.concatenate(audio_chunks)
        return {"audio": audio, "sampling_rate": 24000}

    def _run_chatterbox(self, pipeline: Dict[str, Any], text: str, **kwargs) -> Dict[str, Any]:
        """Run Chatterbox TTS inference."""
        model = pipeline["model"]
        voice_ref = kwargs.get("voice_ref") or pipeline.get("voice_ref")

        config = self._get_model_config("chatterbox")
        exaggeration = kwargs.get("exaggeration", config.get("exaggeration", 0.4))
        cfg_weight = kwargs.get("cfg_weight", config.get("cfg_weight", 0.5))
        temperature = kwargs.get("temperature", config.get("temperature", 0.7))

        generate_kwargs = {
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature,
        }

        if voice_ref:
            generate_kwargs["audio_prompt_path"] = voice_ref

        wav = model.generate(text, **generate_kwargs)

        # wav is a torch tensor of shape [1, samples]
        audio = wav.cpu().numpy()

        return {"audio": audio, "sampling_rate": model.sr}

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

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"text": text}, f)
            input_file = f.name

        with tempfile.TemporaryDirectory() as output_dir:
            cmd = [
                "python", os.path.join(glmtts_path, "glmtts_inference.py"),
                "--text", text,
                "--output_dir", output_dir,
            ]

            try:
                subprocess.run(cmd, cwd=glmtts_path, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"GLM-TTS inference failed: {e.stderr.decode()}")
            finally:
                os.unlink(input_file)

            audio_files = [f for f in os.listdir(output_dir) if f.endswith(".wav")]
            if not audio_files:
                raise RuntimeError("GLM-TTS did not produce output audio")

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
