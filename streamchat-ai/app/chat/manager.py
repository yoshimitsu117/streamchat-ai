"""StreamChat AI — Chat Session Manager."""

from __future__ import annotations

import logging

from app.llm.base import ChatMessage, ChatResponse
from app.llm.router import ModelRouter
from app.chat.history import ConversationHistory

logger = logging.getLogger(__name__)


class ChatManager:
    """Manages chat sessions, message flow, and model routing.

    Coordinates between the model router, conversation history,
    and streaming infrastructure.
    """

    SYSTEM_PROMPT = (
        "You are StreamChat AI, a helpful and knowledgeable assistant. "
        "Provide clear, accurate, and concise responses. "
        "Be conversational and engaging."
    )

    def __init__(
        self,
        router: ModelRouter,
        history: ConversationHistory,
    ):
        self.router = router
        self.history = history

    async def chat(
        self,
        session_id: str,
        message: str,
        model: str | None = None,
        temperature: float = 0.7,
    ) -> ChatResponse:
        """Send a message and get a non-streaming response.

        Args:
            session_id: Chat session ID.
            message: User message.
            model: Model to use (default from config).
            temperature: Sampling temperature.

        Returns:
            ChatResponse with the assistant's reply.
        """
        # Save user message
        self.history.add_message(session_id, "user", message, model=model)

        # Build message list from history
        messages = self._build_messages(session_id)

        # Get provider & generate
        provider = self.router.get_provider(model)
        response = await provider.generate(
            messages=messages,
            temperature=temperature,
        )

        # Save assistant response
        self.history.add_message(
            session_id,
            "assistant",
            response.content,
            model=response.model,
            metadata={"usage": response.usage} if response.usage else None,
        )

        logger.info(
            f"Chat [{session_id}] via {response.model}: "
            f"{len(message)} → {len(response.content)} chars"
        )

        return response

    async def chat_stream(
        self,
        session_id: str,
        message: str,
        model: str | None = None,
        temperature: float = 0.7,
    ):
        """Send a message and get a streaming response.

        Yields tokens as they are generated.
        """
        # Save user message
        self.history.add_message(session_id, "user", message, model=model)

        # Build messages
        messages = self._build_messages(session_id)

        # Get provider & stream
        provider = self.router.get_provider(model)
        full_response = ""

        async for token in provider.stream(
            messages=messages,
            temperature=temperature,
        ):
            full_response += token
            yield token

        # Save full assistant response
        self.history.add_message(
            session_id,
            "assistant",
            full_response,
            model=provider.model_name,
        )

        logger.info(
            f"Stream [{session_id}] via {provider.model_name}: "
            f"{len(full_response)} chars"
        )

    def _build_messages(self, session_id: str) -> list[ChatMessage]:
        """Build the message list for LLM input from session history."""
        messages = [ChatMessage(role="system", content=self.SYSTEM_PROMPT)]

        # Get recent history (limit context window)
        history = self.history.get_history(session_id, limit=20)

        for msg in history:
            if msg["role"] in ("user", "assistant"):
                messages.append(
                    ChatMessage(role=msg["role"], content=msg["content"])
                )

        return messages

    def get_session_info(self, session_id: str) -> dict:
        """Get information about a chat session."""
        history = self.history.get_history(session_id)
        return {
            "session_id": session_id,
            "message_count": len(history),
            "history": history,
        }
