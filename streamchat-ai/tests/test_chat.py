"""Tests for StreamChat AI platform."""

import os
import pytest
import tempfile

from app.chat.history import ConversationHistory
from app.middleware.rate_limiter import RateLimiter
from app.llm.base import ChatMessage


class TestConversationHistory:
    """Tests for SQLite conversation history."""

    @pytest.fixture
    def history(self, tmp_path):
        db_path = str(tmp_path / "test_history.db")
        return ConversationHistory(db_path=db_path)

    def test_add_and_get_message(self, history):
        history.add_message("sess1", "user", "Hello!")
        msgs = history.get_history("sess1")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello!"

    def test_multiple_messages(self, history):
        history.add_message("sess1", "user", "Hi")
        history.add_message("sess1", "assistant", "Hello!")
        history.add_message("sess1", "user", "How are you?")
        msgs = history.get_history("sess1")
        assert len(msgs) == 3
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert msgs[2]["role"] == "user"

    def test_session_isolation(self, history):
        history.add_message("sess1", "user", "Session 1")
        history.add_message("sess2", "user", "Session 2")
        assert len(history.get_history("sess1")) == 1
        assert len(history.get_history("sess2")) == 1

    def test_limit_history(self, history):
        for i in range(10):
            history.add_message("sess1", "user", f"Message {i}")
        msgs = history.get_history("sess1", limit=3)
        assert len(msgs) == 3

    def test_list_sessions(self, history):
        history.add_message("a", "user", "1")
        history.add_message("b", "user", "2")
        sessions = history.list_sessions()
        ids = [s["session_id"] for s in sessions]
        assert "a" in ids
        assert "b" in ids

    def test_delete_session(self, history):
        history.add_message("sess1", "user", "Hi")
        assert history.delete_session("sess1")
        assert len(history.get_history("sess1")) == 0

    def test_delete_nonexistent_session(self, history):
        assert not history.delete_session("nope")

    def test_message_count(self, history):
        history.add_message("sess1", "user", "1")
        history.add_message("sess1", "user", "2")
        assert history.get_message_count("sess1") == 2

    def test_message_with_metadata(self, history):
        history.add_message(
            "sess1", "assistant", "Response",
            model="gpt-4o-mini",
            metadata={"usage": {"tokens": 100}},
        )
        msgs = history.get_history("sess1")
        assert msgs[0]["model"] == "gpt-4o-mini"
        assert msgs[0]["metadata"]["usage"]["tokens"] == 100


class TestRateLimiter:
    """Tests for token-bucket rate limiter."""

    def test_allows_initial_request(self):
        limiter = RateLimiter()
        allowed, info = limiter.check_rate_limit("key1")
        assert allowed
        assert info["remaining"] >= 0

    def test_tracks_remaining(self):
        limiter = RateLimiter()
        limiter.check_rate_limit("key1")
        _, info = limiter.check_rate_limit("key1")
        assert info["remaining"] < limiter.max_requests

    def test_different_keys_independent(self):
        limiter = RateLimiter()
        limiter.check_rate_limit("key1")
        _, info = limiter.check_rate_limit("key2")
        assert info["remaining"] == limiter.max_requests - 1

    def test_consume_tokens(self):
        limiter = RateLimiter()
        assert limiter.consume_tokens("key1", 100)

    def test_consume_too_many_tokens(self):
        limiter = RateLimiter()
        # Consume all tokens
        assert not limiter.consume_tokens("key1", limiter.max_tokens + 1)


class TestChatMessage:
    """Tests for ChatMessage model."""

    def test_create_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_system_message(self):
        msg = ChatMessage(role="system", content="You are helpful")
        assert msg.role == "system"
