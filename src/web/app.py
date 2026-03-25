from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .bot_runner import BotRunner

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SIGNALS_FILE = _PROJECT_ROOT / "data" / "signal_logs" / "signals.jsonl"
_PAPER_ORDERS_FILE = _PROJECT_ROOT / "data" / "paper_logs" / "paper_orders.jsonl"
_PAPER_TRADES_FILE = _PROJECT_ROOT / "data" / "paper_logs" / "paper_trades.jsonl"
_STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Tradedan Dashboard")

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

_BOT = BotRunner(_PROJECT_ROOT)


def _read_jsonl_tail(path: Path, limit: int) -> list[dict[str, Any]]:
    """Read the last *limit* records from a JSONL file.

    For small files (< 512 KB) the whole file is read.  For larger files only
    the tail is read to avoid loading huge logs into memory.
    """
    if not path.exists():
        return []

    size = path.stat().st_size
    if size == 0:
        return []

    CHUNK = 512 * 1024  # 512 KB

    if size <= CHUNK:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    else:
        with open(path, "rb") as f:
            f.seek(-CHUNK, os.SEEK_END)
            raw = f.read()
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        # first partial line is unreliable – drop it
        if lines:
            lines = lines[1:]

    records: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return records[-limit:]


@app.get("/")
async def index():
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/api/signals")
async def get_signals(limit: int = Query(200, ge=1, le=5000)):
    return _read_jsonl_tail(_SIGNALS_FILE, limit)


@app.get("/api/paper-orders")
async def get_paper_orders(limit: int = Query(200, ge=1, le=5000)):
    return _read_jsonl_tail(_PAPER_ORDERS_FILE, limit)


@app.get("/api/paper-trades")
async def get_paper_trades(limit: int = Query(200, ge=1, le=5000)):
    return _read_jsonl_tail(_PAPER_TRADES_FILE, limit)


@app.get("/api/bot/status")
async def bot_status():
    return _BOT.status_dict()


@app.post("/api/bot/start")
async def bot_start(
    config: str = Query("configs/execution.yaml"),
    paper: bool = Query(True),
):
    return _BOT.start(config=config, paper=paper).__dict__


@app.post("/api/bot/stop")
async def bot_stop():
    return _BOT.stop().__dict__
