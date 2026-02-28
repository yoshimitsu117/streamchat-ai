"""StreamChat AI — API Key Authentication."""

from __future__ import annotations

import logging

from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import get_settings

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


async def verify_api_key(request: Request) -> str:
    """Verify the API key from the Authorization header.

    Args:
        request: FastAPI request.

    Returns:
        The validated API key.

    Raises:
        HTTPException: If key is missing or invalid.
    """
    settings = get_settings()

    # Check Authorization header
    auth = request.headers.get("Authorization", "")

    if auth.startswith("Bearer "):
        api_key = auth[7:].strip()
    else:
        api_key = auth.strip()

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Include 'Authorization: Bearer <key>' header.",
        )

    if api_key not in settings.valid_api_keys:
        logger.warning(f"Invalid API key attempt: {api_key[:8]}...")
        raise HTTPException(
            status_code=403,
            detail="Invalid API key.",
        )

    return api_key
