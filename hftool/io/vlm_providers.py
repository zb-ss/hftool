"""VLM provider abstraction for the voiceover pipeline.

Supports local models (via VisionLanguageTask) and cloud API providers
(OpenAI, Google Gemini).  Provider is selected by model name prefix:

    openai/gpt-4o          → OpenAI API
    google/gemini-2.5-flash → Google Gemini API
    qwen3-vl-8b            → Local model (no prefix)
"""

from __future__ import annotations

import base64
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Auto-install helper
# ---------------------------------------------------------------------------


def _auto_install(package: str) -> None:
    """Install a pip package at runtime (used for cloud SDKs in Docker).

    Temporarily restores stderr (fd 2) if it was redirected, since pip
    needs a working stderr for error reporting and progress output.
    """
    import subprocess
    import sys

    print(f"  Installing {package}...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired as exc:
        from hftool.utils.errors import HFToolError

        raise HFToolError(
            f"Timed out installing {package}.",
            suggestion="Check network access from the container or rebuild the Docker image with cloud dependencies preinstalled.",
            original_error=exc,
        ) from exc
    if result.returncode != 0:
        from hftool.utils.errors import HFToolError
        err = result.stderr.strip().split("\n")[-3:] if result.stderr else ["unknown error"]
        raise HFToolError(
            f"Failed to install {package}: {' '.join(err)}",
            suggestion=f"Try manually: pip install {package}",
        )


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

# Maps prefix → (provider class name, env var for API key)
PROVIDER_PREFIXES: Dict[str, tuple] = {
    "openai": ("OpenAIVLMProvider", "OPENAI_API_KEY"),
    "google": ("GoogleVLMProvider", "GOOGLE_API_KEY"),
    "gemini": ("GoogleVLMProvider", "GOOGLE_API_KEY"),
}

DEFAULT_CLOUD_VLM_MODELS: Dict[str, list[str]] = {
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
    ],
    "google": [
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
    ],
}

DEFAULT_ONLINE_VLM_MODEL = "gemini-3-flash-preview"


def parse_vlm_model(model_spec: str) -> tuple:
    """Parse a model spec into (provider_prefix, model_name).

    Args:
        model_spec: e.g. "openai/gpt-4o", "google/gemini-2.5-flash", "qwen3-vl-8b"

    Returns:
        Tuple of (prefix_or_None, model_name).
        For local models the prefix is None.
    """
    cleaned = model_spec.strip()
    if "/" in cleaned:
        prefix_raw, _, model_name = cleaned.partition("/")
        prefix = prefix_raw.strip().lower()
        if prefix in PROVIDER_PREFIXES:
            return prefix, model_name.strip()
    return None, cleaned


def create_vlm_provider(
    model_spec: str,
    device: str = "auto",
    dtype: Optional[str] = None,
) -> VLMProvider:
    """Create the appropriate VLM provider for a model spec.

    Args:
        model_spec: Model identifier, optionally prefixed with provider.
        device: Device for local models (ignored for API providers).
        dtype: Data type for local models (ignored for API providers).

    Returns:
        Configured VLMProvider instance.

    Raises:
        HFToolError: If the provider requires an API key that isn't set.
    """
    from hftool.utils.errors import HFToolError

    prefix, model_name = parse_vlm_model(model_spec)

    if prefix is None:
        return LocalVLMProvider(model_name=model_name, device=device, dtype=dtype)

    if not model_name:
        raise HFToolError(
            f"Invalid cloud VLM model specification: '{model_spec}'.",
            suggestion="Use provider/model format, e.g. openai/gpt-4o or google/gemini-2.5-flash.",
        )

    provider_class_name, env_var = PROVIDER_PREFIXES[prefix]
    api_key = os.environ.get(env_var, "").strip()
    if not api_key:
        raise HFToolError(
            f"API key required for {prefix} provider.",
            suggestion=f"Set the {env_var} environment variable.",
        )

    provider_classes = {
        "OpenAIVLMProvider": OpenAIVLMProvider,
        "GoogleVLMProvider": GoogleVLMProvider,
    }
    cls = provider_classes.get(provider_class_name)
    if cls is None:
        raise HFToolError(f"Unknown VLM provider: {prefix}")

    return cls(model_name=model_name, api_key=api_key)


def get_available_vlm_options() -> list:
    """Build a list of (display_name, model_spec) tuples for the preset dropdown.

    Local models come from MODEL_REGISTRY["vision-language"].
    Cloud models are not listed — the user types any ``provider/model``
    in the input field (e.g. ``openai/gpt-4o``, ``google/gemini-2.5-flash``).

    Returns:
        List of (label, value) tuples suitable for a Select widget.
    """
    from hftool.core.models import MODEL_REGISTRY

    options: list = []

    vlm_models = MODEL_REGISTRY.get("vision-language", {})
    for short_name, info in vlm_models.items():
        size = info.size_str if hasattr(info, "size_str") else f"{info.size_gb:.0f}GB"
        label = f"{info.name} — local, {size}"
        options.append((label, short_name))

    return options


def get_default_cloud_vlm_models(provider: str) -> list[str]:
    """Return built-in fallback cloud model IDs for a provider."""
    normalized = provider.strip().lower()
    if normalized == "gemini":
        normalized = "google"
    return list(DEFAULT_CLOUD_VLM_MODELS.get(normalized, []))


def list_cloud_vlm_models(provider: str, timeout_s: int = 20) -> list[str]:
    """List VLM-capable cloud models from the provider API.

    Returns an empty list when models cannot be fetched.
    """
    normalized = provider.strip().lower()
    if normalized == "gemini":
        normalized = "google"

    if normalized == "openai":
        return _list_openai_vlm_models(timeout_s=timeout_s)
    if normalized == "google":
        return _list_google_vlm_models(timeout_s=timeout_s)
    return []


def _list_openai_vlm_models(timeout_s: int = 20) -> list[str]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return []

    try:
        from openai import OpenAI
    except ImportError:
        _auto_install("openai")
        from openai import OpenAI

    try:
        client = OpenAI(api_key=api_key, timeout=float(timeout_s), max_retries=1)
        model_ids = sorted({m.id for m in client.models.list() if getattr(m, "id", "")})
        return [mid for mid in model_ids if _is_openai_vlm_model(mid)]
    except Exception:
        return []


def _list_google_vlm_models(timeout_s: int = 20) -> list[str]:
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return []

    try:
        from google import genai
    except ImportError:
        _auto_install("google-genai")
        from google import genai

    try:
        client = genai.Client(api_key=api_key)
        models = []
        for model in client.models.list(config={"page_size": 100}):
            raw_name = getattr(model, "name", "") or ""
            model_id = raw_name.split("/", 1)[-1] if "/" in raw_name else raw_name
            if model_id and _is_google_vlm_model(model_id):
                models.append(model_id)
        return sorted(set(models))
    except Exception:
        return []


def _is_openai_vlm_model(model_id: str) -> bool:
    name = model_id.lower()

    if not (
        name.startswith("gpt-4")
        or name.startswith("gpt-4o")
        or name.startswith("gpt-4.1")
        or name.startswith("o1")
        or name.startswith("o3")
        or name.startswith("o4")
    ):
        return False

    blocked_markers = (
        "audio",
        "realtime",
        "transcribe",
        "tts",
        "embedding",
        "omni-moderation",
        "whisper",
    )
    return not any(marker in name for marker in blocked_markers)


def _is_google_vlm_model(model_id: str) -> bool:
    name = model_id.lower()
    if not name.startswith("gemini"):
        return False

    blocked_markers = (
        "embedding",
        "aqa",
        "tts",
        "transcribe",
        "imagen",
        "veo",
        "gemma",
    )
    return not any(marker in name for marker in blocked_markers)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class VLMProvider(ABC):
    """Abstract interface for VLM providers used by the voiceover pipeline."""

    @abstractmethod
    def load(self) -> None:
        """Load / initialise the provider (download model, create client, etc.)."""

    @abstractmethod
    def analyze_frame(
        self,
        image_path: str,
        prompt: str,
        previous_context: str = "",
    ) -> str:
        """Analyze a single image with a text prompt.

        Args:
            image_path: Path to a PNG/JPEG image file.
            prompt: Prompt template; may contain ``{previous_description}``.
            previous_context: Description of the previous frame.

        Returns:
            Model response as plain text.
        """

    @abstractmethod
    def text_inference(self, prompt: str) -> str:
        """Run a text-only inference (no image).

        Used for scene grouping and script assembly.

        Args:
            prompt: The full text prompt.

        Returns:
            Model response as plain text.
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Release resources (VRAM, connections, etc.)."""

    @property
    def is_local(self) -> bool:
        """Whether this provider runs a local model (uses GPU/VRAM)."""
        return False


# ---------------------------------------------------------------------------
# Local provider (wraps existing VisionLanguageTask)
# ---------------------------------------------------------------------------

class LocalVLMProvider(VLMProvider):
    """Local VLM using VisionLanguageTask (Qwen VL, etc.)."""

    def __init__(self, model_name: str, device: str = "auto", dtype: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self._task = None

    @property
    def is_local(self) -> bool:
        return True

    def load(self) -> None:
        from hftool.tasks.vision_language import VisionLanguageTask
        from hftool.core.download import ensure_model_available
        from hftool.core.models import MODEL_REGISTRY

        self._task = VisionLanguageTask(device=self.device, dtype=self.dtype)
        model_id = self._resolve_model_id()

        vlm_models = MODEL_REGISTRY.get("vision-language", {})
        model_info = vlm_models.get(self.model_name)
        if model_info:
            ensure_model_available(
                repo_id=model_id,
                size_gb=model_info.size_gb,
                task_name="vision-language",
                model_name=model_info.name,
                auto_download=True,
                pip_dependencies=model_info.pip_dependencies,
                gated=model_info.gated,
            )

        self._task.load_pipeline(model_id)

    def analyze_frame(self, image_path: str, prompt: str, previous_context: str = "") -> str:
        return self._task.analyze_frame(image_path, prompt, previous_context)

    def text_inference(self, prompt: str) -> str:
        response = self._task.run_inference(self._task._pipeline, {"prompt": prompt})
        return response.get("text", str(response))

    def cleanup(self) -> None:
        if self._task:
            self._task.cleanup()
            self._task = None

    def _resolve_model_id(self) -> str:
        from hftool.core.models import MODEL_REGISTRY

        vlm_models = MODEL_REGISTRY.get("vision-language", {})
        if self.model_name in vlm_models:
            return vlm_models[self.model_name].repo_id
        for info in vlm_models.values():
            if info.repo_id == self.model_name:
                return self.model_name
        return self.model_name


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class OpenAIVLMProvider(VLMProvider):
    """OpenAI API provider (GPT-4o, etc.)."""

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self._api_key = api_key
        self._client = None

    def load(self) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            _auto_install("openai")
            from openai import OpenAI
        self._client = OpenAI(api_key=self._api_key, timeout=60.0, max_retries=1)

    def analyze_frame(self, image_path: str, prompt: str, previous_context: str = "") -> str:
        from hftool.utils.errors import HFToolError

        resolved = prompt.replace("{previous_description}", previous_context)

        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        ext = os.path.splitext(image_path)[1].lower().lstrip(".")
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(ext, "png")

        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": resolved},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/{mime};base64,{b64}",
                            "detail": "low",
                        }},
                    ],
                }],
                max_tokens=1024,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise HFToolError(
                f"OpenAI VLM request failed for model '{self.model_name}'.",
                suggestion="Check OPENAI_API_KEY, network connectivity, and model access.",
                original_error=exc,
            ) from exc

    def text_inference(self, prompt: str) -> str:
        from hftool.utils.errors import HFToolError

        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise HFToolError(
                f"OpenAI VLM text request failed for model '{self.model_name}'.",
                suggestion="Check OPENAI_API_KEY, network connectivity, and model access.",
                original_error=exc,
            ) from exc

    def cleanup(self) -> None:
        self._client = None


# ---------------------------------------------------------------------------
# Google Gemini provider
# ---------------------------------------------------------------------------

class GoogleVLMProvider(VLMProvider):
    """Google Gemini API provider."""

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self._api_key = api_key
        self._client = None
        self._fallback_attempted = False
        self._disable_thinking_config = False

    def load(self) -> None:
        try:
            from google import genai
        except ImportError:
            _auto_install("google-genai")
            from google import genai
        self._client = genai.Client(api_key=self._api_key)

    def _gen_config(self):
        """Config that disables thinking and sets max output tokens."""
        from google.genai import types

        timeout_ms = self._resolve_google_timeout_ms()
        config_kwargs = {
            "max_output_tokens": 4096,
            # google-genai expects timeout in milliseconds.
            "http_options": types.HttpOptions(timeout=timeout_ms),
        }
        # Disable thinking to save tokens/time; some models may not support it
        if not self._disable_thinking_config:
            try:
                config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            except (TypeError, AttributeError, ValueError):
                self._disable_thinking_config = True
        return types.GenerateContentConfig(**config_kwargs)

    @staticmethod
    def _resolve_google_timeout_ms() -> int:
        """Resolve Gemini timeout from env with safe minimums.

        Supported env vars:
        - HFTOOL_CLOUD_VLM_TIMEOUT_MS (preferred)
        - HFTOOL_CLOUD_VLM_TIMEOUT_S (backward compatible)
        """
        timeout_ms_raw = os.environ.get("HFTOOL_CLOUD_VLM_TIMEOUT_MS", "").strip()
        timeout_s_raw = os.environ.get("HFTOOL_CLOUD_VLM_TIMEOUT_S", "").strip()

        try:
            if timeout_ms_raw:
                timeout_ms = int(float(timeout_ms_raw))
            elif timeout_s_raw:
                timeout_ms = int(float(timeout_s_raw) * 1000)
            else:
                timeout_ms = 120_000
        except (TypeError, ValueError):
            timeout_ms = 120_000

        # Gemini enforces a minimum deadline of 10 seconds.
        return max(timeout_ms, 10_000)

    def analyze_frame(self, image_path: str, prompt: str, previous_context: str = "") -> str:
        from hftool.utils.errors import HFToolError
        from PIL import Image

        resolved = prompt.replace("{previous_description}", previous_context)
        img = Image.open(image_path)

        try:
            response = self._generate_with_fallback([resolved, img])
            return response.text or ""
        except Exception as exc:
            raise HFToolError(
                f"Google VLM request failed for model '{self.model_name}'. Details: {exc}",
                suggestion="Check GOOGLE_API_KEY, network connectivity, and Gemini model access.",
                original_error=exc,
            ) from exc

    def text_inference(self, prompt: str) -> str:
        from hftool.utils.errors import HFToolError

        try:
            response = self._generate_with_fallback(prompt)
            return response.text or ""
        except Exception as exc:
            raise HFToolError(
                f"Google VLM text request failed for model '{self.model_name}'. Details: {exc}",
                suggestion="Check GOOGLE_API_KEY, network connectivity, and Gemini model access.",
                original_error=exc,
            ) from exc

    def _generate_with_fallback(self, contents: Any):
        """Generate content with one-time fallback to stable Gemini models."""
        primary_model = self.model_name
        try:
            return self._client.models.generate_content(
                model=primary_model,
                contents=contents,
                config=self._gen_config(),
            )
        except Exception as primary_exc:
            # If thinking config might be the issue, retry without it
            if not self._disable_thinking_config:
                self._disable_thinking_config = True
                try:
                    return self._client.models.generate_content(
                        model=primary_model,
                        contents=contents,
                        config=self._gen_config(),
                    )
                except Exception as retry_exc:
                    primary_exc = retry_exc

            if self._fallback_attempted or not self._should_try_fallback(primary_exc):
                raise

            self._fallback_attempted = True
            fallback_models = [
                model
                for model in get_default_cloud_vlm_models("google")
                if model != primary_model
            ]

            errors: list[str] = []
            for fallback_model in fallback_models:
                try:
                    response = self._client.models.generate_content(
                        model=fallback_model,
                        contents=contents,
                        config=self._gen_config(),
                    )
                    self.model_name = fallback_model
                    return response
                except Exception as fallback_exc:
                    errors.append(f"{fallback_model}: {fallback_exc}")

            detail = "; ".join(errors[:2]) if errors else str(primary_exc)
            raise RuntimeError(
                f"Primary model '{primary_model}' failed and fallback models failed ({detail})."
            ) from primary_exc

    @staticmethod
    def _should_try_fallback(exc: Exception) -> bool:
        message = str(exc).lower()
        fallback_markers = (
            "not found",
            "404",
            "permission",
            "forbidden",
            "unsupported",
            "not allowed",
            "not supported",
            "invalid model",
            "invalid argument",
            "400",
            "deprecated",
            "decommissioned",
        )
        return any(marker in message for marker in fallback_markers)

    def cleanup(self) -> None:
        self._client = None
