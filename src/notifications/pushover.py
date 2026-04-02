"""Pushover notifications (https://pushover.net/api).

Env vars:
  PUSHOVER_APP_TOKEN, PUSHOVER_USER_KEY (required to send)
  PUSHOVER_DEVICE (optional)
  PUSHOVER_SOUND (optional)
  PUSHOVER_PRIORITY (optional, default 0)
"""

from __future__ import annotations

import os
import threading
from typing import Any

import httpx

_PUSHOVER_URL = "https://api.pushover.net/1/messages.json"


def send_pushover(
    message: str,
    title: str | None = None,
    *,
    priority: int | None = None,
    extra: dict[str, Any] | None = None,
) -> bool:
    """
    Send a Pushover message. Returns True if request accepted (HTTP 200 and status==1).

    If credentials are missing, returns False without raising.
    """
    token = os.getenv("PUSHOVER_APP_TOKEN", "").strip()
    user = os.getenv("PUSHOVER_USER_KEY", "").strip()
    if not token or not user:
        return False

    data: dict[str, Any] = {
        "token": token,
        "user": user,
        "message": message,
    }
    if title:
        data["title"] = title[:250]

    dev = os.getenv("PUSHOVER_DEVICE", "").strip()
    if dev:
        data["device"] = dev
    sound = os.getenv("PUSHOVER_SOUND", "").strip()
    if sound:
        data["sound"] = sound

    pr = priority
    if pr is None:
        try:
            pr = int(os.getenv("PUSHOVER_PRIORITY", "0"))
        except ValueError:
            pr = 0
    data["priority"] = pr

    if extra:
        for k, v in extra.items():
            if v is not None and k not in data:
                data[k] = v

    try:
        resp = httpx.post(_PUSHOVER_URL, data=data, timeout=15)
        resp.raise_for_status()
        body = resp.json()
        return body.get("status") == 1
    except Exception as exc:  # noqa: BLE001
        print(f"[Pushover] Failed to send: {exc}")
        return False


def send_pushover_async(
    message: str,
    title: str | None = None,
    *,
    priority: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Fire-and-forget Pushover in a daemon thread (non-blocking for the bot)."""

    def _run() -> None:
        send_pushover(message, title, priority=priority, extra=extra)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
