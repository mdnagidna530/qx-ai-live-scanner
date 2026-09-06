
import os
import secrets
import time


PASSWORD = os.getenv("BOT_PASSWORD", "")

_SESSIONS = {}


def login(password: str) -> str:

    if not PASSWORD:
        raise RuntimeError(
            "BOT_PASSWORD environment variable is not configured"
        )

    if not secrets.compare_digest(
        str(password),
        PASSWORD
    ):
        raise ValueError(
            "Invalid password"
        )

    token = secrets.token_urlsafe(32)

    _SESSIONS[token] = {
        "created": time.time()
    }

    return token


def require(token: str | None) -> None:

    if not token:
        raise PermissionError(
            "Authentication required"
        )

    session = _SESSIONS.get(token)

    if not session:
        raise PermissionError(
            "Invalid session"
        )

    # 12-hour session lifetime
    if time.time() - session["created"] > 43200:

        _SESSIONS.pop(token, None)

        raise PermissionError(
            "Session expired"
        )
