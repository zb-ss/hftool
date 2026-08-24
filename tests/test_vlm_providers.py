"""Tests for VLM provider parsing and creation."""

import pytest


def test_parse_vlm_model_normalizes_cloud_prefix_and_whitespace():
    from hftool.io.vlm_providers import parse_vlm_model

    prefix, model_name = parse_vlm_model("  OpenAI/gpt-4o-mini  ")

    assert prefix == "openai"
    assert model_name == "gpt-4o-mini"


def test_parse_vlm_model_keeps_unknown_prefix_as_local_model_spec():
    from hftool.io.vlm_providers import parse_vlm_model

    prefix, model_name = parse_vlm_model("  Qwen/Qwen2.5-VL-7B-Instruct  ")

    assert prefix is None
    assert model_name == "Qwen/Qwen2.5-VL-7B-Instruct"


def test_create_vlm_provider_accepts_mixed_case_cloud_prefix(monkeypatch):
    from hftool.io.vlm_providers import OpenAIVLMProvider, create_vlm_provider

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    provider = create_vlm_provider(" OpenAI/gpt-4o ")

    assert isinstance(provider, OpenAIVLMProvider)
    assert provider.model_name == "gpt-4o"


def test_create_vlm_provider_rejects_empty_cloud_model_spec(monkeypatch):
    from hftool.io.vlm_providers import create_vlm_provider
    from hftool.utils.errors import HFToolError

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with pytest.raises(HFToolError, match="Invalid cloud VLM model specification"):
        create_vlm_provider("openai/")


def test_default_online_model_is_google_gemini_flash_preview():
    from hftool.io.vlm_providers import DEFAULT_ONLINE_VLM_MODEL

    assert DEFAULT_ONLINE_VLM_MODEL == "gemini-3-flash-preview"


def test_google_cloud_defaults_include_default_online_model():
    from hftool.io.vlm_providers import DEFAULT_ONLINE_VLM_MODEL, get_default_cloud_vlm_models

    models = get_default_cloud_vlm_models("google")

    assert DEFAULT_ONLINE_VLM_MODEL in models


def test_google_provider_should_try_fallback_for_missing_model_error():
    from hftool.io.vlm_providers import GoogleVLMProvider

    assert GoogleVLMProvider._should_try_fallback(RuntimeError("404 model not found"))


def test_google_provider_falls_back_to_stable_model(monkeypatch):
    from hftool.io.vlm_providers import GoogleVLMProvider

    class _FakeResponse:
        def __init__(self, text: str):
            self.text = text

    class _FakeModels:
        def generate_content(self, model, contents, config):
            if model == "gemini-3-flash-preview":
                raise RuntimeError("404 model not found")
            return _FakeResponse("ok")

    class _FakeClient:
        def __init__(self):
            self.models = _FakeModels()

    provider = GoogleVLMProvider(model_name="gemini-3-flash-preview", api_key="test-key")
    provider._client = _FakeClient()

    response = provider._generate_with_fallback("ping")

    assert response.text == "ok"
    assert provider.model_name != "gemini-3-flash-preview"


def test_google_timeout_env_seconds_is_converted_to_ms(monkeypatch):
    from hftool.io.vlm_providers import GoogleVLMProvider

    monkeypatch.delenv("HFTOOL_CLOUD_VLM_TIMEOUT_MS", raising=False)
    monkeypatch.setenv("HFTOOL_CLOUD_VLM_TIMEOUT_S", "30")

    assert GoogleVLMProvider._resolve_google_timeout_ms() == 30_000


def test_google_timeout_env_respects_minimum_deadline(monkeypatch):
    from hftool.io.vlm_providers import GoogleVLMProvider

    monkeypatch.setenv("HFTOOL_CLOUD_VLM_TIMEOUT_MS", "500")

    assert GoogleVLMProvider._resolve_google_timeout_ms() == 10_000
