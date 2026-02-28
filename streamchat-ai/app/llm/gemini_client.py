"""StreamChat AI — Google Gemini LLM Provider."""

from __future__ import annotations

import logging
from typing import AsyncIterator

import openai

from app.llm.base import BaseLLMProvider, ChatMessage, ChatResponse
from app.config import get_settings

logger = logging.getLogger(__name__)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider (via OpenAI-compatible API)."""

    def __init__(self, model: str | None = None):
        settings = get_settings()
        self.client = openai.AsyncOpenAI(
            api_key=settings.gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self._model = model or settings.gemini_model

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> ChatResponse:
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
