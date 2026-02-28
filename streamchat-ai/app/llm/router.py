"""StreamChat AI — Multi-Model Router."""

from __future__ import annotations

import logging

from app.llm.base import BaseLLMProvider
from app.llm.openai_client import OpenAIProvider
from app.llm.gemini_client import GeminiProvider
from app.config import get_settings

logger = logging.getLogger(__name__)


class ModelRouter:
    """Routes requests to the appropriate LLM provider based on model name.

    Supports automatic fallback if a provider is unavailable.
    """

    def __init__(self):
        settings = get_settings()
        self._providers: dict[str, BaseLLMProvider] = {}
        self._model_to_provider: dict[str, str] = {}

        # Register OpenAI models
        if settings.openai_api_key:
            openai_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
            for model in openai_models:
                provider = OpenAIProvider(model=model)
                self._providers[model] = provider
                self._model_to_provider[model] = "openai"
            logger.info(f"Registered OpenAI models: {openai_models}")

        # Register Gemini models
        if settings.gemini_api_key:
            gemini_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"]
            for model in gemini_models:
                provider = GeminiProvider(model=model)
                self._providers[model] = provider
                self._model_to_provider[model] = "gemini"
            logger.info(f"Registered Gemini models: {gemini_models}")

        self.default_model = settings.default_model

    def get_provider(self, model: str | None = None) -> BaseLLMProvider:
        """Get the provider for a specific model.

        Args:
            model: Model name (e.g., 'gpt-4o-mini', 'gemini-1.5-flash').
                   Falls back to default model if None.

        Returns:
            LLM provider instance.

        Raises:
            ValueError: If model is not registered.
        """
        model = model or self.default_model

        if model not in self._providers:
            available = list(self._providers.keys())
            raise ValueError(
                f"Model '{model}' not available. "
                f"Available models: {available}"
            )

        return self._providers[model]

    def list_models(self) -> list[dict]:
        """List all available models and their providers."""
        return [
            {
                "model": model,
                "provider": self._model_to_provider[model],
            }
            for model in sorted(self._providers.keys())
        ]
