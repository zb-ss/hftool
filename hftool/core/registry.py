"""Task registry system for hftool.

Defines available tasks, their handlers, and required dependencies.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class TaskConfig:
    """Configuration for a task handler."""
    
    # Handler module path (e.g., "hftool.tasks.text_to_image")
    handler: str
    
    # Primary library: "diffusers", "transformers", or "custom"
    library: str
    
    # Input type: "text", "image", "audio", "video"
    input_type: str
    
    # Output type: "text", "image", "audio", "video"
    output_type: str
    
    # Required Python dependencies
    required_deps: List[str] = field(default_factory=list)
    
    # Whether ffmpeg is required (for video/audio processing)
    requires_ffmpeg: bool = False
    
    # Default models for this task (first is the primary default)
    default_models: List[str] = field(default_factory=list)
    
    # Description for help text
    description: str = ""
    
    # Additional task-specific configuration
    config: Dict[str, Any] = field(default_factory=dict)


# Task registry - maps task names to their configurations
TASK_REGISTRY: Dict[str, TaskConfig] = {
    # ============================================
    # TEXT-TO-IMAGE (diffusers)
    # ============================================
    "text-to-image": TaskConfig(
        handler="hftool.tasks.text_to_image",
        library="diffusers",
        input_type="text",
        output_type="image",
        required_deps=["diffusers", "PIL", "accelerate", "torch"],
        default_models=[
            "Tongyi-MAI/Z-Image-Turbo",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "black-forest-labs/FLUX.1-schnell",
        ],
        description="Generate images from text prompts (Z-Image, SDXL, FLUX, etc.)",
    ),
    
    # ============================================
    # TEXT-TO-VIDEO (diffusers)
    # ============================================
    "text-to-video": TaskConfig(
        handler="hftool.tasks.text_to_video",
        library="diffusers",
        input_type="text",
        output_type="video",
        required_deps=["diffusers", "PIL", "accelerate", "torch"],
        requires_ffmpeg=True,
        default_models=[
            "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
            "THUDM/CogVideoX-5b",
            "Wan-AI/Wan2.1-T2V-1.3B",
        ],
        description="Generate videos from text prompts (HunyuanVideo, CogVideoX, Wan2.2)",
    ),
    
    # ============================================
    # IMAGE-TO-IMAGE (diffusers)
    # ============================================
    "image-to-image": TaskConfig(
        handler="hftool.tasks.image_to_image",
        library="diffusers",
        input_type="image",
        output_type="image",
        required_deps=["diffusers", "PIL", "accelerate", "torch"],
        default_models=[
            "stabilityai/stable-diffusion-xl-refiner-1.0",
        ],
        description="Transform images with text guidance (style transfer, editing)",
    ),
    
    # ============================================
    # IMAGE-TO-VIDEO (diffusers)
    # ============================================
    "image-to-video": TaskConfig(
        handler="hftool.tasks.text_to_video",
        library="diffusers",
        input_type="image",
        output_type="video",
        required_deps=["diffusers", "PIL", "accelerate", "torch"],
        requires_ffmpeg=True,
        default_models=[
            "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v",
            "THUDM/CogVideoX-5b-I2V",
        ],
        description="Generate videos from images (HunyuanVideo I2V, CogVideoX I2V)",
        config={"mode": "i2v"},
    ),
    
    # ============================================
    # TEXT-TO-SPEECH (transformers/custom)
    # ============================================
    "text-to-speech": TaskConfig(
        handler="hftool.tasks.text_to_speech",
        library="transformers",
        input_type="text",
        output_type="audio",
        required_deps=["transformers", "soundfile", "torch"],
        requires_ffmpeg=True,  # For MP3 conversion
        default_models=[
            "microsoft/VibeVoice-Realtime-0.5B",
            "suno/bark-small",
            "facebook/mms-tts-eng",
        ],
        description="Generate speech from text (VibeVoice, Bark, MMS-TTS)",
    ),
    
    # ============================================
    # SPEECH-TO-TEXT / ASR (transformers)
    # ============================================
    "automatic-speech-recognition": TaskConfig(
        handler="hftool.tasks.speech_to_text",
        library="transformers",
        input_type="audio",
        output_type="text",
        required_deps=["transformers", "soundfile", "torch"],
        default_models=[
            "openai/whisper-large-v3",
            "openai/whisper-medium",
            "openai/whisper-small",
        ],
        description="Transcribe speech to text (Whisper)",
    ),
    
    # ============================================
    # TEXT GENERATION (transformers)
    # ============================================
    "text-generation": TaskConfig(
        handler="hftool.tasks.transformers_generic",
        library="transformers",
        input_type="text",
        output_type="text",
        required_deps=["transformers", "torch"],
        default_models=[
            "meta-llama/Llama-3.2-1B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ],
        description="Generate text from prompts (LLMs)",
    ),
    
    # ============================================
    # TEXT CLASSIFICATION (transformers)
    # ============================================
    "text-classification": TaskConfig(
        handler="hftool.tasks.transformers_generic",
        library="transformers",
        input_type="text",
        output_type="text",
        required_deps=["transformers", "torch"],
        default_models=[
            "distilbert-base-uncased-finetuned-sst-2-english",
        ],
        description="Classify text into categories",
    ),
    
    # ============================================
    # QUESTION ANSWERING (transformers)
    # ============================================
    "question-answering": TaskConfig(
        handler="hftool.tasks.transformers_generic",
        library="transformers",
        input_type="text",
        output_type="text",
        required_deps=["transformers", "torch"],
        default_models=[
            "distilbert-base-cased-distilled-squad",
        ],
        description="Answer questions from context",
    ),
    
    # ============================================
    # SUMMARIZATION (transformers)
    # ============================================
    "summarization": TaskConfig(
        handler="hftool.tasks.transformers_generic",
        library="transformers",
        input_type="text",
        output_type="text",
        required_deps=["transformers", "torch"],
        default_models=[
            "facebook/bart-large-cnn",
        ],
        description="Summarize long text",
    ),
    
    # ============================================
    # TRANSLATION (transformers)
    # ============================================
    "translation": TaskConfig(
        handler="hftool.tasks.transformers_generic",
        library="transformers",
        input_type="text",
        output_type="text",
        required_deps=["transformers", "torch"],
        default_models=[
            "Helsinki-NLP/opus-mt-en-de",
        ],
        description="Translate text between languages",
    ),
    
    # ============================================
    # IMAGE CLASSIFICATION (transformers)
    # ============================================
    "image-classification": TaskConfig(
        handler="hftool.tasks.transformers_generic",
        library="transformers",
        input_type="image",
        output_type="text",
        required_deps=["transformers", "PIL", "torch"],
        default_models=[
            "google/vit-base-patch16-224",
        ],
        description="Classify images into categories",
    ),
    
    # ============================================
    # OBJECT DETECTION (transformers)
    # ============================================
    "object-detection": TaskConfig(
        handler="hftool.tasks.transformers_generic",
        library="transformers",
        input_type="image",
        output_type="text",
        required_deps=["transformers", "PIL", "torch"],
        default_models=[
            "facebook/detr-resnet-50",
        ],
        description="Detect objects in images",
    ),
    
    # ============================================
    # IMAGE-TO-TEXT (transformers)
    # ============================================
    "image-to-text": TaskConfig(
        handler="hftool.tasks.transformers_generic",
        library="transformers",
        input_type="image",
        output_type="text",
        required_deps=["transformers", "PIL", "torch"],
        default_models=[
            "Salesforce/blip-image-captioning-base",
        ],
        description="Generate captions for images",
    ),
}

# Task aliases for convenience
TASK_ALIASES: Dict[str, str] = {
    "t2i": "text-to-image",
    "i2i": "image-to-image",
    "img2img": "image-to-image",
    "t2v": "text-to-video",
    "i2v": "image-to-video",
    "tts": "text-to-speech",
    "asr": "automatic-speech-recognition",
    "stt": "automatic-speech-recognition",
    "speech-to-text": "automatic-speech-recognition",
    "llm": "text-generation",
    "qa": "question-answering",
}


def get_task_config(task: str) -> TaskConfig:
    """Get the configuration for a task.
    
    Args:
        task: Task name or alias
    
    Returns:
        TaskConfig for the task
    
    Raises:
        ValueError: If the task is not found
    """
    # Resolve alias
    resolved = TASK_ALIASES.get(task, task)
    
    if resolved not in TASK_REGISTRY:
        available = list(TASK_REGISTRY.keys()) + list(TASK_ALIASES.keys())
        raise ValueError(
            f"Unknown task: '{task}'. Available tasks: {', '.join(sorted(set(available)))}"
        )
    
    return TASK_REGISTRY[resolved]


def list_tasks() -> Dict[str, str]:
    """List all available tasks with descriptions.
    
    Returns:
        Dictionary mapping task names to descriptions
    """
    return {name: config.description for name, config in TASK_REGISTRY.items()}


def get_default_model(task: str) -> str:
    """Get the default model for a task.
    
    Args:
        task: Task name or alias
    
    Returns:
        Default model name
    
    Raises:
        ValueError: If the task has no default models
    """
    config = get_task_config(task)
    if not config.default_models:
        raise ValueError(f"Task '{task}' has no default models configured")
    return config.default_models[0]
