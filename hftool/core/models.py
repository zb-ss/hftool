"""Model registry with download metadata for hftool.

Defines available models for each task with HuggingFace repo IDs, sizes, and download info.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ModelType(Enum):
    """Type of model/pipeline."""
    DIFFUSERS = "diffusers"
    TRANSFORMERS = "transformers"
    CUSTOM = "custom"


@dataclass
class ModelInfo:
    """Information about a downloadable model."""
    
    # HuggingFace repository ID (e.g., "openai/whisper-large-v3")
    repo_id: str
    
    # Human-readable name
    name: str
    
    # Model type (diffusers, transformers, custom)
    model_type: ModelType
    
    # Approximate size in GB (for user information)
    size_gb: float
    
    # Whether this is the default model for the task
    is_default: bool = False
    
    # Short description
    description: str = ""
    
    # Specific revision/commit to download (optional)
    revision: Optional[str] = None
    
    # Files to exclude from download (to reduce size)
    ignore_patterns: List[str] = field(default_factory=list)
    
    # Additional pip packages required for this model
    # Will be installed when model is downloaded/used
    pip_dependencies: List[str] = field(default_factory=list)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def short_name(self) -> str:
        """Get short name for CLI display (last part of repo_id)."""
        return self.repo_id.split("/")[-1]
    
    @property
    def size_str(self) -> str:
        """Get human-readable size string."""
        if self.size_gb >= 1:
            return f"{self.size_gb:.1f} GB"
        return f"{int(self.size_gb * 1024)} MB"


# =============================================================================
# MODEL REGISTRY
# =============================================================================

MODEL_REGISTRY: Dict[str, Dict[str, ModelInfo]] = {
    # =========================================================================
    # TEXT-TO-IMAGE MODELS
    # =========================================================================
    "text-to-image": {
        "z-image-turbo": ModelInfo(
            repo_id="Tongyi-MAI/Z-Image-Turbo",
            name="Z-Image Turbo",
            model_type=ModelType.DIFFUSERS,
            size_gb=6.0,
            is_default=True,
            description="Fast high-quality image generation (9 steps)",
            metadata={"num_inference_steps": 9, "guidance_scale": 0.0},
        ),
        "z-image": ModelInfo(
            repo_id="Tongyi-MAI/Z-Image",
            name="Z-Image",
            model_type=ModelType.DIFFUSERS,
            size_gb=6.0,
            description="High-quality image generation",
            metadata={"num_inference_steps": 9, "guidance_scale": 0.0},
        ),
        "sdxl": ModelInfo(
            repo_id="stabilityai/stable-diffusion-xl-base-1.0",
            name="Stable Diffusion XL",
            model_type=ModelType.DIFFUSERS,
            size_gb=6.5,
            description="Stable Diffusion XL base model (1024x1024)",
            metadata={"num_inference_steps": 30, "guidance_scale": 7.5},
        ),
        "flux-schnell": ModelInfo(
            repo_id="black-forest-labs/FLUX.1-schnell",
            name="FLUX.1 Schnell",
            model_type=ModelType.DIFFUSERS,
            size_gb=23.0,
            description="Fast FLUX model (4 steps)",
            metadata={"num_inference_steps": 4, "guidance_scale": 0.0},
        ),
        "flux-dev": ModelInfo(
            repo_id="black-forest-labs/FLUX.1-dev",
            name="FLUX.1 Dev",
            model_type=ModelType.DIFFUSERS,
            size_gb=23.0,
            description="High-quality FLUX model",
            metadata={"num_inference_steps": 28, "guidance_scale": 3.5},
        ),
    },
    
    # =========================================================================
    # IMAGE-TO-IMAGE MODELS
    # =========================================================================
    "image-to-image": {
        "qwen-image-edit": ModelInfo(
            repo_id="Qwen/Qwen-Image-Edit-2511",
            name="Qwen Image Edit",
            model_type=ModelType.DIFFUSERS,
            size_gb=25.0,
            is_default=True,
            description="Qwen image editing with character consistency and multi-image support",
            pip_dependencies=["diffusers>=0.36.0", "transformers>=4.45.0"],
            metadata={
                "num_inference_steps": 40,
                "guidance_scale": 1.0,
                "true_cfg_scale": 4.0,
                "pipeline_class": "QwenImageEditPlusPipeline",
            },
        ),
        "sdxl-refiner": ModelInfo(
            repo_id="stabilityai/stable-diffusion-xl-refiner-1.0",
            name="SDXL Refiner",
            model_type=ModelType.DIFFUSERS,
            size_gb=6.2,
            description="SDXL refiner for img2img enhancement and style transfer",
            metadata={"num_inference_steps": 30, "strength": 0.3},
        ),
        "sdxl": ModelInfo(
            repo_id="stabilityai/stable-diffusion-xl-base-1.0",
            name="SDXL Base (img2img)",
            model_type=ModelType.DIFFUSERS,
            size_gb=6.5,
            description="SDXL base model for stronger style transfer",
            metadata={"num_inference_steps": 30, "strength": 0.7},
        ),
    },
    
    # =========================================================================
    # TEXT-TO-VIDEO MODELS
    # =========================================================================
    "text-to-video": {
        "hunyuanvideo-1.5-480p": ModelInfo(
            repo_id="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
            name="HunyuanVideo 1.5 (480p)",
            model_type=ModelType.DIFFUSERS,
            size_gb=25.0,
            is_default=True,
            description="High-quality text-to-video (480p)",
        ),
        "hunyuanvideo-1.5-720p": ModelInfo(
            repo_id="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
            name="HunyuanVideo 1.5 (720p)",
            model_type=ModelType.DIFFUSERS,
            size_gb=25.0,
            description="High-quality text-to-video (720p)",
        ),
        "cogvideox-5b": ModelInfo(
            repo_id="THUDM/CogVideoX-5b",
            name="CogVideoX 5B",
            model_type=ModelType.DIFFUSERS,
            size_gb=20.0,
            description="CogVideo text-to-video model",
        ),
        "wan2.1-1.3b": ModelInfo(
            repo_id="Wan-AI/Wan2.1-T2V-1.3B",
            name="Wan 2.1 (1.3B)",
            model_type=ModelType.DIFFUSERS,
            size_gb=5.0,
            description="Lightweight text-to-video model",
        ),
        "wan2.1-14b": ModelInfo(
            repo_id="Wan-AI/Wan2.1-T2V-14B",
            name="Wan 2.1 (14B)",
            model_type=ModelType.DIFFUSERS,
            size_gb=28.0,
            description="High-quality Wan text-to-video model",
        ),
    },
    
    # =========================================================================
    # IMAGE-TO-VIDEO MODELS
    # =========================================================================
    "image-to-video": {
        "hunyuanvideo-1.5-480p-i2v": ModelInfo(
            repo_id="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v",
            name="HunyuanVideo 1.5 I2V (480p)",
            model_type=ModelType.DIFFUSERS,
            size_gb=25.0,
            is_default=True,
            description="Image-to-video generation (480p)",
        ),
        "cogvideox-5b-i2v": ModelInfo(
            repo_id="THUDM/CogVideoX-5b-I2V",
            name="CogVideoX 5B I2V",
            model_type=ModelType.DIFFUSERS,
            size_gb=20.0,
            description="CogVideo image-to-video model",
        ),
    },
    
    # =========================================================================
    # TEXT-TO-SPEECH MODELS
    # =========================================================================
    "text-to-speech": {
        "bark-small": ModelInfo(
            repo_id="suno/bark-small",
            name="Bark Small",
            model_type=ModelType.TRANSFORMERS,
            size_gb=1.5,
            is_default=True,
            description="Suno Bark TTS (small, fast)",
        ),
        "bark": ModelInfo(
            repo_id="suno/bark",
            name="Bark",
            model_type=ModelType.TRANSFORMERS,
            size_gb=5.0,
            description="Suno Bark TTS (full quality)",
        ),
        "mms-tts-eng": ModelInfo(
            repo_id="facebook/mms-tts-eng",
            name="MMS-TTS English",
            model_type=ModelType.TRANSFORMERS,
            size_gb=0.3,
            description="Facebook's multilingual TTS (English, lightweight)",
        ),
        # VibeVoice disabled - streaming API not compatible with current inference code
        # "vibevoice": ModelInfo(
        #     repo_id="microsoft/VibeVoice-Realtime-0.5B",
        #     name="VibeVoice Realtime",
        #     model_type=ModelType.CUSTOM,
        #     size_gb=1.0,
        #     description="Microsoft's realtime TTS model",
        #     pip_dependencies=["vibevoice"],
        # ),
    },
    
    # =========================================================================
    # SPEECH-TO-TEXT / ASR MODELS
    # =========================================================================
    "automatic-speech-recognition": {
        "whisper-large-v3": ModelInfo(
            repo_id="openai/whisper-large-v3",
            name="Whisper Large v3",
            model_type=ModelType.TRANSFORMERS,
            size_gb=3.1,
            is_default=True,
            description="OpenAI Whisper large (best quality)",
        ),
        "whisper-large-v3-turbo": ModelInfo(
            repo_id="openai/whisper-large-v3-turbo",
            name="Whisper Large v3 Turbo",
            model_type=ModelType.TRANSFORMERS,
            size_gb=1.6,
            description="Fast Whisper large variant",
        ),
        "whisper-medium": ModelInfo(
            repo_id="openai/whisper-medium",
            name="Whisper Medium",
            model_type=ModelType.TRANSFORMERS,
            size_gb=1.5,
            description="OpenAI Whisper medium (balanced)",
        ),
        "whisper-small": ModelInfo(
            repo_id="openai/whisper-small",
            name="Whisper Small",
            model_type=ModelType.TRANSFORMERS,
            size_gb=0.5,
            description="OpenAI Whisper small (fast)",
        ),
        "whisper-base": ModelInfo(
            repo_id="openai/whisper-base",
            name="Whisper Base",
            model_type=ModelType.TRANSFORMERS,
            size_gb=0.15,
            description="OpenAI Whisper base (lightweight)",
        ),
    },
    
    # =========================================================================
    # TEXT GENERATION MODELS
    # =========================================================================
    "text-generation": {
        "llama-3.2-1b": ModelInfo(
            repo_id="meta-llama/Llama-3.2-1B-Instruct",
            name="Llama 3.2 (1B)",
            model_type=ModelType.TRANSFORMERS,
            size_gb=2.5,
            is_default=True,
            description="Meta Llama 3.2 1B instruction-tuned",
        ),
        "llama-3.2-3b": ModelInfo(
            repo_id="meta-llama/Llama-3.2-3B-Instruct",
            name="Llama 3.2 (3B)",
            model_type=ModelType.TRANSFORMERS,
            size_gb=6.5,
            description="Meta Llama 3.2 3B instruction-tuned",
        ),
        "mistral-7b": ModelInfo(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            name="Mistral 7B",
            model_type=ModelType.TRANSFORMERS,
            size_gb=14.0,
            description="Mistral 7B instruction-tuned",
        ),
        "qwen2.5-0.5b": ModelInfo(
            repo_id="Qwen/Qwen2.5-0.5B-Instruct",
            name="Qwen 2.5 (0.5B)",
            model_type=ModelType.TRANSFORMERS,
            size_gb=1.0,
            description="Qwen 2.5 0.5B instruction-tuned",
        ),
    },
    
    # =========================================================================
    # TEXT CLASSIFICATION MODELS
    # =========================================================================
    "text-classification": {
        "distilbert-sst2": ModelInfo(
            repo_id="distilbert-base-uncased-finetuned-sst-2-english",
            name="DistilBERT SST-2",
            model_type=ModelType.TRANSFORMERS,
            size_gb=0.27,
            is_default=True,
            description="Sentiment analysis (positive/negative)",
        ),
    },
    
    # =========================================================================
    # QUESTION ANSWERING MODELS
    # =========================================================================
    "question-answering": {
        "distilbert-squad": ModelInfo(
            repo_id="distilbert-base-cased-distilled-squad",
            name="DistilBERT SQuAD",
            model_type=ModelType.TRANSFORMERS,
            size_gb=0.26,
            is_default=True,
            description="Question answering from context",
        ),
    },
    
    # =========================================================================
    # SUMMARIZATION MODELS
    # =========================================================================
    "summarization": {
        "bart-cnn": ModelInfo(
            repo_id="facebook/bart-large-cnn",
            name="BART CNN",
            model_type=ModelType.TRANSFORMERS,
            size_gb=1.6,
            is_default=True,
            description="Text summarization",
        ),
    },
    
    # =========================================================================
    # TRANSLATION MODELS
    # =========================================================================
    "translation": {
        "opus-mt-en-de": ModelInfo(
            repo_id="Helsinki-NLP/opus-mt-en-de",
            name="OPUS MT EN-DE",
            model_type=ModelType.TRANSFORMERS,
            size_gb=0.3,
            is_default=True,
            description="English to German translation",
        ),
        "opus-mt-en-fr": ModelInfo(
            repo_id="Helsinki-NLP/opus-mt-en-fr",
            name="OPUS MT EN-FR",
            model_type=ModelType.TRANSFORMERS,
            size_gb=0.3,
            description="English to French translation",
        ),
        "opus-mt-en-es": ModelInfo(
            repo_id="Helsinki-NLP/opus-mt-en-es",
            name="OPUS MT EN-ES",
            model_type=ModelType.TRANSFORMERS,
            size_gb=0.3,
            description="English to Spanish translation",
        ),
    },
    
    # =========================================================================
    # IMAGE CLASSIFICATION MODELS
    # =========================================================================
    "image-classification": {
        "vit-base": ModelInfo(
            repo_id="google/vit-base-patch16-224",
            name="ViT Base",
            model_type=ModelType.TRANSFORMERS,
            size_gb=0.35,
            is_default=True,
            description="Vision Transformer for image classification",
        ),
    },
    
    # =========================================================================
    # OBJECT DETECTION MODELS
    # =========================================================================
    "object-detection": {
        "detr-resnet-50": ModelInfo(
            repo_id="facebook/detr-resnet-50",
            name="DETR ResNet-50",
            model_type=ModelType.TRANSFORMERS,
            size_gb=0.17,
            is_default=True,
            description="Object detection with DETR",
        ),
    },
    
    # =========================================================================
    # IMAGE-TO-TEXT MODELS
    # =========================================================================
    "image-to-text": {
        "blip-captioning": ModelInfo(
            repo_id="Salesforce/blip-image-captioning-base",
            name="BLIP Captioning",
            model_type=ModelType.TRANSFORMERS,
            size_gb=0.9,
            is_default=True,
            description="Image captioning with BLIP",
        ),
        "blip-captioning-large": ModelInfo(
            repo_id="Salesforce/blip-image-captioning-large",
            name="BLIP Captioning Large",
            model_type=ModelType.TRANSFORMERS,
            size_gb=1.8,
            description="Image captioning with BLIP (large)",
        ),
    },
}


def get_models_for_task(task: str) -> Dict[str, ModelInfo]:
    """Get all available models for a task.
    
    Args:
        task: Task name (e.g., "text-to-image")
    
    Returns:
        Dictionary mapping model short names to ModelInfo
    
    Raises:
        ValueError: If task is not found
    """
    from hftool.core.registry import TASK_ALIASES
    
    # Resolve alias
    resolved_task = TASK_ALIASES.get(task, task)
    
    if resolved_task not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown task: '{task}'. Available: {', '.join(available)}")
    
    return MODEL_REGISTRY[resolved_task]


def get_model_info(task: str, model: str) -> ModelInfo:
    """Get model info by task and model name.
    
    Args:
        task: Task name
        model: Model short name or repo_id
    
    Returns:
        ModelInfo for the model
    
    Raises:
        ValueError: If model is not found
    """
    models = get_models_for_task(task)
    
    # Try exact match on short name
    if model in models:
        return models[model]
    
    # Try match on repo_id
    model_lower = model.lower()
    for info in models.values():
        if info.repo_id.lower() == model_lower:
            return info
        # Also match on last part of repo_id
        if info.short_name.lower() == model_lower:
            return info
    
    available = list(models.keys())
    raise ValueError(f"Unknown model '{model}' for task '{task}'. Available: {', '.join(available)}")


def get_default_model_info(task: str) -> ModelInfo:
    """Get the default model for a task.
    
    Args:
        task: Task name
    
    Returns:
        ModelInfo for the default model
    
    Raises:
        ValueError: If no default model is configured
    """
    models = get_models_for_task(task)
    
    for info in models.values():
        if info.is_default:
            return info
    
    # If no default, return the first one
    if models:
        return next(iter(models.values()))
    
    raise ValueError(f"No models configured for task '{task}'")


def find_model_by_repo_id(repo_id: str) -> Optional[tuple]:
    """Find model info by repo_id across all tasks.
    
    Args:
        repo_id: HuggingFace repository ID
    
    Returns:
        Tuple of (task, short_name, ModelInfo) or None if not found
    """
    repo_lower = repo_id.lower()
    
    for task, models in MODEL_REGISTRY.items():
        for short_name, info in models.items():
            if info.repo_id.lower() == repo_lower:
                return (task, short_name, info)
    
    return None
