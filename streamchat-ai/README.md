# 💬 StreamChat AI

**Real-time LLM Chat Platform with Multi-Model Routing**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![WebSocket](https://img.shields.io/badge/WebSocket-Real--time-010101?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![OpenAI](https://img.shields.io/badge/Multi--Model-GPT%20%7C%20Gemini-412991?style=for-the-badge)

---

## 🧠 Overview

StreamChat AI is a **production-grade real-time chat platform** that supports streaming LLM responses via WebSockets and Server-Sent Events. It features **multi-model routing** (OpenAI GPT, Google Gemini), persistent conversation history, token-based rate limiting, and API key authentication.

Built for teams needing a **self-hosted, scalable chat backend** with full control over model selection, rate limits, and conversation management.

---

## 🏗️ Architecture

```
┌───────────────────────────────────────────────────┐
│                 Client Layer                       │
│         WebSocket / SSE / REST API                 │
└───────────────────┬───────────────────────────────┘
                    │
┌───────────────────▼───────────────────────────────┐
│              FastAPI Application                   │
│                                                    │
│  ┌──────────────┐  ┌───────────────┐              │
│  │ Auth         │  │ Rate Limiter  │              │
│  │ Middleware   │  │ (Token-based) │              │
│  └──────┬───────┘  └──────┬────────┘              │
│         │                 │                        │
│  ┌──────▼─────────────────▼────────┐              │
│  │        Chat Manager             │              │
│  │  (Session + Stream Control)     │              │
│  └──────────────┬──────────────────┘              │
│                 │                                  │
│  ┌──────────────▼──────────────────┐              │
│  │        Model Router             │              │
│  │  ┌─────────┐  ┌──────────┐     │              │
│  │  │ OpenAI  │  │  Gemini  │     │              │
│  │  │ Client  │  │  Client  │     │              │
│  │  └─────────┘  └──────────┘     │              │
│  └─────────────────────────────────┘              │
│                                                    │
│  ┌─────────────────────────────────┐              │
│  │    Conversation History         │              │
│  │    (SQLite Persistence)         │              │
│  └─────────────────────────────────┘              │
└────────────────────────────────────────────────────┘
```

---

## ✨ Features

- **Real-time Streaming** — WebSocket and SSE support for token-by-token output
- **Multi-Model Router** — Switch between OpenAI GPT-4 and Google Gemini
- **Conversation History** — SQLite-backed persistent chat history
- **Rate Limiting** — Token-based rate limiting per API key
- **API Key Auth** — Bearer token authentication middleware
- **Session Management** — Multiple concurrent chat sessions
- **Async Architecture** — Fully async Python with FastAPI and asyncio
- **Docker Ready** — Production container deployment

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key and/or Google Gemini API key

### 1. Clone & Install
```bash
git clone https://github.com/yoshimitsu117/streamchat-ai.git
cd streamchat-ai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example .env
# Add your API keys
```

### 3. Run
```bash
uvicorn app.main:app --reload --port 8003
```

### 4. Chat via REST
```bash
curl -X POST http://localhost:8003/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "message": "Explain quantum computing in simple terms",
    "model": "gpt-4o-mini",
    "session_id": "demo-session"
  }'
```

### 5. Chat via WebSocket
```python
import asyncio
import websockets

async def main():
    async with websockets.connect("ws://localhost:8003/ws/chat/demo-session") as ws:
        await ws.send('{"message": "Hello!", "model": "gpt-4o-mini"}')
        async for token in ws:
            print(token, end="", flush=True)

asyncio.run(main())
```

---

## 🐳 Docker
```bash
docker-compose up --build
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/chat` | Send a message (JSON response) |
| `POST` | `/api/v1/chat/stream` | Send a message (SSE stream) |
| `WS`   | `/ws/chat/{session_id}` | WebSocket chat connection |
| `GET`  | `/api/v1/sessions` | List active sessions |
| `GET`  | `/api/v1/sessions/{id}/history` | Get session history |
| `DELETE` | `/api/v1/sessions/{id}` | Delete a session |
| `GET`  | `/api/v1/models` | List available models |
| `GET`  | `/health` | Health check |

---

## 📁 Project Structure

```
streamchat-ai/
├── app/
│   ├── main.py              # FastAPI app + WebSocket endpoints
│   ├── config.py             # Configuration
│   ├── chat/
│   │   ├── manager.py       # Chat session management
│   │   ├── streaming.py     # SSE & WebSocket streaming
│   │   └── history.py       # Conversation history (SQLite)
│   ├── llm/
│   │   ├── router.py        # Multi-model router
│   │   ├── openai_client.py # OpenAI provider
│   │   ├── gemini_client.py # Google Gemini provider
│   │   └── base.py          # Abstract LLM provider
│   └── middleware/
│       ├── rate_limiter.py   # Token-based rate limiting
│       └── auth.py           # API key authentication
├── tests/
│   └── test_chat.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Siddharth** — AI Engineer  
Building production-grade AI systems, not just demos.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/yoshimitsu117)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/yoshimitsu117)
