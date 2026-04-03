"""
DataAgent: descarga datos OHLCV históricos desde la API pública de Bybit (v5).

No requiere autenticación (endpoint público de klines).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BASE_URL = "https://api.bybit.com"

# Conexión lenta / SSL: handshake puede superar timeouts cortos; reintentar fallos transitorios.
_KLINE_MAX_ATTEMPTS = 3
_KLINE_HTTP_TIMEOUT = httpx.Timeout(connect=35.0, read=60.0, write=30.0, pool=15.0)
_KLINE_RETRY_EXC: tuple[type[BaseException], ...] = (
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.ConnectError,
)

# spot = contado; linear = perpetuo USDT (compatibilidad con CSV antiguos en subcarpeta)
DEFAULT_BYBIT_CATEGORY = "linear"

COLUMNAS_OHLCV = ["fecha", "abierto", "alto", "bajo", "cierre", "volumen"]

_INTERVAL_MAP = {
    "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
    "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720",
    "1D": "D", "1W": "W", "1M": "M",
    "1": "1", "3": "3", "5": "5", "15": "15", "30": "30",
    "60": "60", "120": "120", "240": "240", "360": "360", "720": "720",
    "D": "D", "W": "W", "M": "M",
}


class DataAgent:
    """Obtiene y almacena datos OHLCV de Bybit."""

    def __init__(
        self,
        data_dir: Path | None = None,
        category: str | None = None,
    ):
        self.category = (category or DEFAULT_BYBIT_CATEGORY).strip().lower()
        base = Path(data_dir) if data_dir else _PROJECT_ROOT / "data" / "bybit"
        self.data_dir = base / self.category
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_interval(self, timeframe: str) -> str:
        key = timeframe.strip()
        if key in _INTERVAL_MAP:
            return _INTERVAL_MAP[key]
        raise ValueError(f"Timeframe no soportado: {timeframe!r}")

    def _fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        limit: int = 200,
    ) -> list[list]:
        """Descarga un bloque de klines desde Bybit v5 (público)."""
        params = {
            "category": self.category,
            "symbol": symbol,
            "interval": interval,
            "start": start_ms,
            "end": end_ms,
            "limit": limit,
        }
        url = f"{_BASE_URL}/v5/market/kline"
        for attempt in range(_KLINE_MAX_ATTEMPTS):
            try:
                resp = httpx.get(url, params=params, timeout=_KLINE_HTTP_TIMEOUT)
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
                if data.get("retCode") != 0:
                    raise RuntimeError(f"Bybit API error: {data}")
                return data["result"]["list"]
            except _KLINE_RETRY_EXC:
                if attempt + 1 >= _KLINE_MAX_ATTEMPTS:
                    raise
                time.sleep(0.5 * (2**attempt))
        raise RuntimeError("kline fetch: internal retry loop exhausted")

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        days: int = 365,
        force: bool = False,
    ) -> pd.DataFrame:
        """
        Devuelve DataFrame OHLCV para *symbol* con *days* días de histórico.

        Si el CSV ya existe y *force* es False, carga desde disco.
        """
        interval = self._resolve_interval(timeframe)
        csv_path = self.data_dir / f"{symbol}_{timeframe}.csv"

        if not force and csv_path.exists():
            df = pd.read_csv(csv_path)
            df["fecha"] = pd.to_datetime(df["fecha"])
            return df.sort_values("fecha").reset_index(drop=True)

        now_ms = int(time.time() * 1000)
        start_ms = now_ms - days * 24 * 60 * 60 * 1000
        all_rows: list[list] = []

        # Bybit devuelve las velas del más reciente al más antiguo
        cursor_end = now_ms
        while cursor_end > start_ms:
            batch = self._fetch_klines(symbol, interval, start_ms, cursor_end, limit=200)
            if not batch:
                break
            all_rows.extend(batch)
            oldest_ts = int(batch[-1][0])
            if oldest_ts >= cursor_end:
                break
            cursor_end = oldest_ts - 1
            time.sleep(0.12)

        if not all_rows:
            raise ValueError(f"Sin datos para {symbol} ({timeframe}, {days}d)")

        # Bybit kline: [startTime, open, high, low, close, volume, turnover]
        df = pd.DataFrame(all_rows, columns=[
            "ts", "abierto", "alto", "bajo", "cierre", "volumen", "turnover",
        ])
        df["fecha"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
        for col in ["abierto", "alto", "bajo", "cierre", "volumen"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[COLUMNAS_OHLCV].drop_duplicates(subset="fecha").sort_values("fecha").reset_index(drop=True)

        df.to_csv(csv_path, index=False)
        print(f"[DataAgent] Guardado {csv_path} ({len(df)} filas) [category={self.category}]")
        return df
