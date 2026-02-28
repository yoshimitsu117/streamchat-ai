"""StreamChat AI — Abstract LLM Provider Base."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator
from dataclasses import dataclass


@dataclass
class ChatMessage:
    """A single chat message."""

    role: str  # system, user, assistant
    content: str


@dataclass
class ChatResponse:
    """Response from an LLM provider."""

    content: str
    model: str
    usage: dict | None = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of this provider."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier."""
        ...

    @abstractmethod
    async def generate(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> ChatResponse:
        """Generate a response (non-streaming)."""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        """Stream response tokens."""
        ...
