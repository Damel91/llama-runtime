"""Explicit loopback helper routing; never ambient proxy discovery."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

_TOKEN = re.compile(r"[A-Za-z0-9_-]{32,256}\Z")


def normalize_proxy_url(value: str | None) -> str | None:
    """Validate a local HTTP helper URL without echoing its capability token."""
    if value is None:
        return None
    invalid = "proxy_url must identify an explicit loopback HTTP helper"
    try:
        if (
            not isinstance(value, str)
            or not value
            or len(value) > 1024
            or any(
                char.isspace()
                or ord(char) < 32
                or 127 <= ord(char) <= 159
                or 0xD800 <= ord(char) <= 0xDFFF
                or char in "\\?#"
                for char in value
            )
        ):
            raise ValueError
        parsed = urlsplit(value)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.path
            or parsed.query
            or parsed.fragment
            or parsed.port is None
            or not 1 <= parsed.port <= 65535
        ):
            raise ValueError
        credentials = ""
        if parsed.username is not None or parsed.password is not None:
            if (
                parsed.username != "runtime"
                or not isinstance(parsed.password, str)
                or not _TOKEN.fullmatch(parsed.password)
            ):
                raise ValueError
            credentials = f"runtime:{parsed.password}@"
        return f"http://{credentials}127.0.0.1:{parsed.port}"
    except (TypeError, ValueError):
        raise ValueError(invalid) from None


def assert_client_proxy(client: Any, expected: str | None) -> None:
    """A profile requiring a helper cannot silently use a different route."""
    if client is None or expected is None:
        return
    try:
        matches = getattr(client, "proxy_url", None) == expected
    except Exception:
        matches = False
    if not matches:
        raise ValueError(
            "injected client does not match the required proxy route"
        ) from None


def proxy_secrets(value: str | None) -> tuple[str, ...]:
    """Redact the full route and its capability token from opted-in captures."""
    if value is None:
        return ()
    password = urlsplit(value).password
    return (value, password) if password else (value,)
