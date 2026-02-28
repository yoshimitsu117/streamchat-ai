"""StreamChat AI — FastAPI Application.

Real-time LLM Chat Platform with Multi-Model Routing.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.llm.router import ModelRouter
from app.chat.history import ConversationHistory
from app.chat.manager import ChatManager
from app.chat.streaming import sse_stream, WebSocketStreamHandler
from app.middleware.auth import verify_api_key
from app.middleware.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# Global components
_router: ModelRouter | None = None
_history: ConversationHistory | None = None
_manager: ChatManager | None = None
_rate_limiter: RateLimiter | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _router, _history, _manager, _rate_limiter
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    _router = ModelRouter()
    _history = ConversationHistory()
    _manager = ChatManager(router=_router, history=_history)
    _rate_limiter = RateLimiter()

    logger.info("All components initialized")
    yield
    logger.info("Shutting down StreamChat AI")


app = FastAPI(
    title="StreamChat AI API",
    description="Real-time LLM Chat Platform with Multi-Model Routing",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    model: str | None = Field(default=None, description="Model to use")
    session_id: str = Field(default="default", description="Chat session ID")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    content: str
    model: str
    session_id: str
    usage: dict | None = None


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.get("/health")
async def health_check():
    settings = get_settings()
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "models": _router.list_models() if _router else [],
    }


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, api_key: str = Depends(verify_api_key)):
    """Send a message and get a response."""
    # Rate limit
    allowed, info = _rate_limiter.check_rate_limit(api_key)
    if not allowed:
        raise HTTPException(status_code=429, detail=info)

    try:
        response = await _manager.chat(
            session_id=request.session_id,
            message=request.message,
            model=request.model,
            temperature=request.temperature,
        )

        return ChatResponse(
            content=response.content,
            model=response.model,
            session_id=request.session_id,
            usage=response.usage,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/chat/stream")
async def chat_stream(request: ChatRequest, api_key: str = Depends(verify_api_key)):
    """Send a message and stream the response via SSE."""
    allowed, info = _rate_limiter.check_rate_limit(api_key)
    if not allowed:
        raise HTTPException(status_code=429, detail=info)

    try:
        model = request.model or _router.default_model
        token_stream = _manager.chat_stream(
            session_id=request.session_id,
            message=request.message,
            model=request.model,
            temperature=request.temperature,
        )

        return StreamingResponse(
            sse_stream(token_stream, request.session_id, model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat."""
    await websocket.accept()
    logger.info(f"WebSocket connected: session={session_id}")

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message_data = json.loads(data)

            user_message = message_data.get("message", "")
            model = message_data.get("model")

            if not user_message:
                await websocket.send_json({"type": "error", "message": "Empty message"})
                continue

            # Stream response
            token_stream = _manager.chat_stream(
                session_id=session_id,
                message=user_message,
                model=model,
            )

            await WebSocketStreamHandler.stream_to_websocket(
                websocket, token_stream, model or _router.default_model
            )

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: session={session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


@app.get("/api/v1/sessions")
async def list_sessions(api_key: str = Depends(verify_api_key)):
    """List all active chat sessions."""
    return _history.list_sessions()


@app.get("/api/v1/sessions/{session_id}/history")
async def get_session_history(
    session_id: str,
    limit: int = 50,
    api_key: str = Depends(verify_api_key),
):
    """Get conversation history for a session."""
    return _manager.get_session_info(session_id)


@app.delete("/api/v1/sessions/{session_id}")
async def delete_session(session_id: str, api_key: str = Depends(verify_api_key)):
    """Delete a chat session."""
    deleted = _history.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": f"Session {session_id} deleted"}


@app.get("/api/v1/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    """List all available LLM models."""
    return _router.list_models()
