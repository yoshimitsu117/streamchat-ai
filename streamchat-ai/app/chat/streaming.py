"""StreamChat AI — SSE & WebSocket Streaming Utilities."""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

logger = logging.getLogger(__name__)


async def sse_stream(
    token_stream: AsyncIterator[str],
    session_id: str,
    model: str,
) -> AsyncIterator[str]:
    """Wrap a token stream as Server-Sent Events.

    Yields SSE-formatted strings:
        data: {"token": "...", "type": "token"}
        data: {"type": "done", "model": "..."}

    Args:
        token_stream: Async iterator of tokens from LLM.
        session_id: Chat session identifier.
        model: Model used for generation.
    """
    try:
        full_response = ""
        async for token in token_stream:
            full_response += token
            event = json.dumps({
                "token": token,
                "type": "token",
                "session_id": session_id,
            })
            yield f"data: {event}\n\n"

        # Send completion event
        done_event = json.dumps({
            "type": "done",
            "model": model,
            "session_id": session_id,
            "total_length": len(full_response),
        })
        yield f"data: {done_event}\n\n"

    except Exception as e:
        logger.error(f"SSE stream error: {e}")
        error_event = json.dumps({
            "type": "error",
            "message": str(e),
        })
        yield f"data: {error_event}\n\n"


class WebSocketStreamHandler:
    """Handles WebSocket streaming for chat sessions."""

    @staticmethod
    async def stream_to_websocket(
        websocket,
        token_stream: AsyncIterator[str],
        model: str,
    ) -> str:
        """Stream tokens to a WebSocket connection.

        Args:
            websocket: FastAPI WebSocket connection.
            token_stream: Async iterator of tokens.
            model: Model name.

        Returns:
            Full response text.
        """
        full_response = ""

        try:
            async for token in token_stream:
                full_response += token
                await websocket.send_json({
                    "type": "token",
                    "token": token,
                })

            await websocket.send_json({
                "type": "done",
                "model": model,
                "total_length": len(full_response),
            })

        except Exception as e:
            logger.error(f"WebSocket stream error: {e}")
            await websocket.send_json({
                "type": "error",
                "message": str(e),
            })

        return full_response
