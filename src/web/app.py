from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .bot_runner import MultiBotRunner

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SIGNALS_FILE = _PROJECT_ROOT / "data" / "signal_logs" / "signals.jsonl"
_PAPER_ORDERS_FILE = _PROJECT_ROOT / "data" / "paper_logs" / "paper_orders.jsonl"
_PAPER_TRADES_FILE = _PROJECT_ROOT / "data" / "paper_logs" / "paper_trades.jsonl"
_STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Tradedan Dashboard")

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

_BOTS = MultiBotRunner(_PROJECT_ROOT)


def _parse_symbols(raw: str) -> list[str]:
    items = [p.strip().upper() for p in (raw or "").split(",")]
    out = [s for s in items if s]
    # dedupe preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


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


def _read_jsonl_all(path: Path) -> list[dict[str, Any]]:
    """Read all valid JSONL records from *path*."""
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


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


@app.get("/api/paper-summary")
async def get_paper_summary():
    trades = _read_jsonl_all(_PAPER_TRADES_FILE)
    orders = _read_jsonl_all(_PAPER_ORDERS_FILE)
    signals = _read_jsonl_tail(_SIGNALS_FILE, 1)

    if not trades:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "pnl_total": 0.0,
            "avg_return": 0.0,
            "best_trade_pnl": None,
            "worst_trade_pnl": None,
            "last_trade_timestamp": None,
            "last_signal": signals[-1] if signals else None,
            "open_order_count": len(orders),
            "symbols": [],
        }

    pnl_values = [float(t.get("pnl", 0.0) or 0.0) for t in trades]
    returns = [float(t.get("retorno", 0.0) or 0.0) for t in trades]
    wins = sum(1 for p in pnl_values if p > 0)

    by_symbol: dict[str, dict[str, Any]] = {}
    for t, pnl in zip(trades, pnl_values):
        symbol = str(t.get("symbol", "UNKNOWN"))
        info = by_symbol.setdefault(symbol, {"symbol": symbol, "trade_count": 0, "wins": 0, "pnl_total": 0.0})
        info["trade_count"] += 1
        info["pnl_total"] += pnl
        if pnl > 0:
            info["wins"] += 1

    symbols = []
    for info in by_symbol.values():
        n = int(info["trade_count"])
        symbols.append(
            {
                "symbol": info["symbol"],
                "trade_count": n,
                "win_rate": (float(info["wins"]) / n) if n else 0.0,
                "pnl_total": float(info["pnl_total"]),
            }
        )
    symbols.sort(key=lambda x: x["pnl_total"], reverse=True)

    return {
        "trade_count": len(trades),
        "win_rate": wins / len(trades),
        "pnl_total": sum(pnl_values),
        "avg_return": sum(returns) / len(returns),
        "best_trade_pnl": max(pnl_values),
        "worst_trade_pnl": min(pnl_values),
        "last_trade_timestamp": trades[-1].get("timestamp"),
        "last_signal": signals[-1] if signals else None,
        "open_order_count": len(orders),
        "symbols": symbols,
    }


@app.get("/api/bot/status")
async def bot_status():
    # backwards-compatible: return first bot if any
    st = _BOTS.statuses()
    return st[0] if st else {"running": False, "pid": None, "last_log_lines": []}


@app.post("/api/bot/start")
async def bot_start(
    config: str = Query("configs/execution.yaml"),
    paper: bool = Query(True),
):
    # backwards-compatible: start default BTCUSDT
    return _BOTS.start_one("BTCUSDT", config=config, paper=paper).__dict__


@app.post("/api/bot/stop")
async def bot_stop():
    return _BOTS.stop_one("BTCUSDT").__dict__


@app.get("/api/bots/status")
async def bots_status():
    return _BOTS.statuses()


@app.post("/api/bots/start")
async def bots_start(
    symbols: str = Query("BTCUSDT,ETHUSDT,ADAUSDT"),
    config: str = Query("configs/execution.yaml"),
    paper: bool = Query(True),
):
    return _BOTS.start_many(_parse_symbols(symbols), config=config, paper=paper)


@app.post("/api/bots/stop")
async def bots_stop(
    symbols: str = Query("BTCUSDT,ETHUSDT,ADAUSDT"),
):
    out = []
    for s in _parse_symbols(symbols):
        out.append(_BOTS.stop_one(s).__dict__)
    return out
