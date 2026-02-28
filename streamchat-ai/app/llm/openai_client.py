"""StreamChat AI — OpenAI LLM Provider."""

from __future__ import annotations

import logging
from typing import AsyncIterator

import openai

from app.llm.base import BaseLLMProvider, ChatMessage, ChatResponse
from app.config import get_settings

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider."""

    def __init__(self, model: str | None = None):
        settings = get_settings()
        self.client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = model or settings.openai_model

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> ChatResponse:
        """Generate a non-streaming response."""
        formatted = [{"role": m.role, "content": m.content} for m in messages]

        response = await self.client.chat.completions.create(
            model=self._model,
            messages=formatted,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return ChatResponse(
            content=response.choices[0].message.content or "",
            model=self._model,
            usage=usage,
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        """Stream response tokens."""
        formatted = [{"role": m.role, "content": m.content} for m in messages]

        stream = await self.client.chat.completions.create(
            model=self._model,
            messages=formatted,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
