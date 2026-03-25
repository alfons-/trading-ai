"""Email notifications for trade executions (stdlib only: smtplib + email.mime).

SMTP settings: ``SMTP_HOST``, ``SMTP_PORT``, ``SMTP_USER``, ``SMTP_PASS``.
When ``recipients`` is empty, returns immediately (opt-in, silent).
"""

from __future__ import annotations

import os
import smtplib
import threading
from email.mime.text import MIMEText
from typing import Any


def _format_price(price: Any) -> str:
    if price is None or price == "":
        return "?"
    try:
        return f"{float(price):.2f}"
    except (TypeError, ValueError):
        return str(price)


def _build_message(order_info: dict[str, Any], recipients: list[str], sender: str) -> MIMEText:
    side_raw = str(order_info.get("side", "TRADE"))
    side = "BUY" if side_raw.lower() == "buy" else "SELL" if side_raw.lower() == "sell" else side_raw.upper()
    symbol = order_info.get("symbol", "???")
    price = _format_price(order_info.get("price"))

    subject = f"[Trading AI] {side} {symbol} @ {price}"

    lines = [f"{k}: {v}" for k, v in order_info.items()]
    body = "\n".join(lines)

    msg = MIMEText(body, "plain", "utf-8")
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    return msg


def _send(order_info: dict[str, Any], recipients: list[str]) -> None:
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER", "")
    password = os.getenv("SMTP_PASS", "")

    if not user or not password:
        return

    try:
        msg = _build_message(order_info, recipients, sender=user)
        with smtplib.SMTP(host, port, timeout=15) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(user, recipients, msg.as_string())
    except Exception as exc:  # noqa: BLE001 — must not propagate to caller thread
        print(f"[Email] Failed to send: {exc}")


def send_trade_email(order_info: dict[str, Any], recipients: list[str]) -> None:
    """Notify configured addresses about a trade in a daemon thread (non-blocking).

    No-op if ``recipients`` is empty or order ``ret_code`` is non-zero.
    Never raises: SMTP work runs in a thread and errors are swallowed.
    """
    if not recipients:
        return
    if order_info.get("ret_code", 0) != 0:
        return

    try:
        thread = threading.Thread(target=_send, args=(order_info, recipients), daemon=True)
        thread.start()
    except Exception:
        pass
