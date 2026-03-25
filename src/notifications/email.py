"""Email notifications for trade executions.

Reads SMTP configuration from environment variables.  When ``NOTIFY_EMAILS``
is unset or empty the module is a silent no-op so the bot never breaks.
"""

from __future__ import annotations

import os
import smtplib
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any


def _build_message(order_info: dict[str, Any], recipients: list[str], sender: str) -> MIMEMultipart:
    side = str(order_info.get("side", "TRADE")).upper()
    symbol = order_info.get("symbol", "???")
    price = order_info.get("price", "?")

    subject = f"[Tradedan] {side} {symbol} @ {price}"

    lines = [f"{k}: {v}" for k, v in order_info.items()]
    body = "\n".join(lines)

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))
    return msg


def _send(order_info: dict[str, Any], recipients: list[str]) -> None:
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER", "")
    password = os.getenv("SMTP_PASS", "")

    if not user or not password:
        print("[Email] SMTP_USER / SMTP_PASS not configured – skipping notification.")
        return

    msg = _build_message(order_info, recipients, sender=user)

    try:
        with smtplib.SMTP(host, port, timeout=15) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(user, recipients, msg.as_string())
        print(f"[Email] Notification sent to {', '.join(recipients)}")
    except Exception as exc:  # noqa: BLE001
        print(f"[Email] Failed to send notification: {exc}")


def send_trade_email(order_info: dict[str, Any]) -> None:
    """Send an email notification about a trade to all configured recipients.

    Reads ``NOTIFY_EMAILS`` (comma-separated) from the environment.
    Runs in a daemon thread so the trading loop is never blocked.
    """
    raw = os.getenv("NOTIFY_EMAILS", "")
    recipients = [e.strip() for e in raw.split(",") if e.strip()]
    if not recipients:
        return

    thread = threading.Thread(target=_send, args=(order_info, recipients), daemon=True)
    thread.start()
