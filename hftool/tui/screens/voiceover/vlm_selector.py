"""VLM model selector mixin for the voiceover TUI screen.

Handles switching between local and cloud (OpenAI/Google) VLM providers,
fetching available model lists from provider APIs, and populating the
Select widgets.
"""

from __future__ import annotations

from typing import Dict, List

from textual import work
from textual.widgets import Label, Select


class VlmSelectorMixin:
    """Mixin that manages the VLM source/model selection widgets.

    Expects the host Screen to contain these widget IDs:
    - ``#vlm-local-row``, ``#vlm-online-row``, ``#vlm-online-status``
    - ``#vlm-provider-select``, ``#vlm-online-model-select``

    State:
        _cloud_model_cache: Provider → model list cache (avoids re-fetching).
    """

    _cloud_model_cache: Dict[str, List[str]]

    def _init_vlm_selector(self) -> None:
        """Initialise VLM selector state. Call from ``__init__``."""
        self._cloud_model_cache = {}

    # ------------------------------------------------------------------
    # Visibility toggling
    # ------------------------------------------------------------------

    def _apply_vlm_source_visibility(self, source: str) -> None:
        is_online = source == "online"
        self.query_one("#vlm-local-row").display = not is_online
        self.query_one("#vlm-online-row").display = is_online
        self.query_one("#vlm-online-status").display = is_online

    # ------------------------------------------------------------------
    # Online model fetching
    # ------------------------------------------------------------------

    def _set_online_status(self, text: str) -> None:
        self.query_one("#vlm-online-status", Label).update(text)

    def _refresh_online_models(self, force_refresh: bool) -> None:
        provider = str(self.query_one("#vlm-provider-select", Select).value)
        if not provider:
            return
        self._load_online_models_worker(provider, force_refresh=force_refresh)

    @work(thread=True)
    def _load_online_models_worker(self, provider: str, force_refresh: bool = False) -> None:
        from hftool.io.vlm_providers import get_default_cloud_vlm_models, list_cloud_vlm_models

        provider = provider.strip().lower()

        if not force_refresh and provider in self._cloud_model_cache:
            models = self._cloud_model_cache[provider]
            self.app.call_from_thread(self._apply_online_model_options, provider, models, "cached")
            return

        self.app.call_from_thread(
            self._set_online_status,
            f"Online models: querying {provider} endpoint...",
        )
        models = list_cloud_vlm_models(provider)
        source = "provider endpoint"

        if not models:
            models = get_default_cloud_vlm_models(provider)
            source = "built-in defaults"

        self._cloud_model_cache[provider] = models
        self.app.call_from_thread(self._apply_online_model_options, provider, models, source)

    def _apply_online_model_options(
        self, provider: str, models: list[str], source: str,
    ) -> None:
        from hftool.io.vlm_providers import DEFAULT_ONLINE_VLM_MODEL

        current_provider = str(
            self.query_one("#vlm-provider-select", Select).value,
        ).strip().lower()
        if current_provider != provider:
            return

        model_select = self.query_one("#vlm-online-model-select", Select)
        options = (
            [(model, model) for model in models]
            or [(DEFAULT_ONLINE_VLM_MODEL, DEFAULT_ONLINE_VLM_MODEL)]
        )
        model_select.set_options(options)

        current_value = str(model_select.value) if model_select.value is not None else ""
        option_values = {value for _label, value in options}
        if current_value not in option_values:
            if provider == "google" and DEFAULT_ONLINE_VLM_MODEL in option_values:
                model_select.value = DEFAULT_ONLINE_VLM_MODEL
            else:
                model_select.value = options[0][1]

        self._set_online_status(f"Online models: loaded {len(options)} from {source}.")
