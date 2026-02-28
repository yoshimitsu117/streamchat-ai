"""StreamChat AI — Token-Based Rate Limiter."""

from __future__ import annotations

import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class RateBucket:
    """Token bucket for rate limiting."""

    tokens: float
    last_refill: float = field(default_factory=time.time)


class RateLimiter:
    """Token-bucket rate limiter per API key.

    Limits both request count and estimated token usage.
    """

    def __init__(self):
        settings = get_settings()
        self.max_requests = settings.rate_limit_requests
        self.max_tokens = settings.rate_limit_tokens
        self.window = 60.0  # 1 minute window

        self._request_buckets: dict[str, RateBucket] = defaultdict(
            lambda: RateBucket(tokens=self.max_requests)
        )
        self._token_buckets: dict[str, RateBucket] = defaultdict(
            lambda: RateBucket(tokens=self.max_tokens)
        )

    def check_rate_limit(self, api_key: str) -> tuple[bool, dict]:
        """Check if a request is within rate limits.

        Args:
            api_key: The API key making the request.

        Returns:
            Tuple of (allowed: bool, info: dict).
        """
        now = time.time()

        # Refill request bucket
        req_bucket = self._request_buckets[api_key]
        elapsed = now - req_bucket.last_refill
        req_bucket.tokens = min(
            self.max_requests,
            req_bucket.tokens + (elapsed / self.window) * self.max_requests,
        )
        req_bucket.last_refill = now

        if req_bucket.tokens < 1:
            return False, {
                "error": "Rate limit exceeded",
                "retry_after_seconds": int(self.window * (1 - req_bucket.tokens) / self.max_requests),
                "limit": self.max_requests,
                "remaining": 0,
            }

        # Consume a token
        req_bucket.tokens -= 1

        return True, {
            "limit": self.max_requests,
            "remaining": int(req_bucket.tokens),
            "window": f"{int(self.window)}s",
        }

    def consume_tokens(self, api_key: str, token_count: int) -> bool:
        """Consume estimated tokens from the token bucket.

        Args:
            api_key: API key.
            token_count: Estimated tokens used.

        Returns:
            True if within limits.
        """
        now = time.time()
        bucket = self._token_buckets[api_key]

        elapsed = now - bucket.last_refill
        bucket.tokens = min(
            self.max_tokens,
            bucket.tokens + (elapsed / self.window) * self.max_tokens,
        )
        bucket.last_refill = now

        if bucket.tokens < token_count:
            return False

        bucket.tokens -= token_count
        return True
