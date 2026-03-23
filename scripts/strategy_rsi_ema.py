"""
Estrategia basada solo en RSI y MACD (sin XGBoost ni otros filtros).

Entrada — solo velas de 4 horas (4h):
    - RSI(14) en 4h ha tocado el nivel 30 (sobreventa) en una ventana reciente.
    - EMA(14) del RSI cruza el RSI de abajo hacia arriba:
      en la vela anterior la EMA estaba por debajo del RSI y en la actual
      la EMA está en o por encima del RSI (ema_prev < rsi_prev y ema_now >= rsi_now).
    - MACD(12,26,9) en 4h en zona alcista: histograma > 0 (línea MACD > señal).

Salida — condición en velas de 1 día (1D), duración del trade libre:
    - Cierras cuando el RSI(14) calculado sobre cierres DIARIOS (vela 1D) >= 70.
    - Eso NO implica que el trade dure 24 h: puede durar días o meses hasta que
      el RSI de la vela diaria alcance 70.
    - El backtest recorre la serie 4h; en cada vela 4h se consulta el último RSI
      diario ya conocido (merge_asof hacia atrás, sin mirar el futuro). La salida
      se registra en la fecha/hora de esa vela 4h y el precio es su cierre 4h.

Uso:
    python -m scripts.strategy_rsi_ema
    python -m scripts.strategy_rsi_ema --symbol ETHUSDT --days 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.data_agent import DataAgent

TRADES_SUFFIX = "rsi_macd"


def build_strategy_df(
    symbol: str,
    timeframe: str = "4h",
    days: int = 365,
    rsi_period: int = 14,
    ema_rsi_period: int = 14,
    rsi_oversold: float = 30.0,
    rsi_overbought_daily: float = 70.0,
    touch_lookback_bars: int = 12,
) -> pd.DataFrame:
    """
    Descarga 4h + 1D, calcula RSI/MACD en 4h, RSI en 1D (merge_asof), señales.
    """
    data_agent = DataAgent()
    df = data_agent.get_ohlcv(symbol=symbol, timeframe=timeframe, days=days)
    df = df.sort_values("fecha").reset_index(drop=True)
    df["fecha"] = pd.to_datetime(df["fecha"])

    # ── RSI + EMA(RSI) en 4h ──
    rsi_4h = RSIIndicator(close=df["cierre"], window=rsi_period)
    df["rsi"] = rsi_4h.rsi()
    df["rsi_ema_14"] = df["rsi"].ewm(span=ema_rsi_period, adjust=False).mean()

    # ── MACD en 4h (solo indicadores permitidos junto al RSI) ──
    macd_ind = MACD(close=df["cierre"], window_fast=12, window_slow=26, window_sign=9)
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_hist"] = macd_ind.macd_diff()

    # ── RSI diario para salida (1D) ──
    daily = data_agent.get_ohlcv(symbol=symbol, timeframe="1D", days=days)
    daily = daily.sort_values("fecha").reset_index(drop=True)
    daily["fecha"] = pd.to_datetime(daily["fecha"])
    rsi_1d = RSIIndicator(close=daily["cierre"], window=rsi_period)
    daily["rsi_1d"] = rsi_1d.rsi()
    daily_feat = daily[["fecha", "rsi_1d"]].dropna()

    df = pd.merge_asof(
        df,
        daily_feat,
        on="fecha",
        direction="backward",
        suffixes=("", "_dup"),
    )

    # Señales
    signals = np.zeros(len(df), dtype=int)
    in_position = False

    for i in range(1, len(df)):
        row_now = df.iloc[i]
        row_prev = df.iloc[i - 1]

        rsi_now = row_now["rsi"]
        rsi_prev = row_prev["rsi"]
        ema_now = row_now["rsi_ema_14"]
        ema_prev = row_prev["rsi_ema_14"]
        rsi_d = row_now["rsi_1d"]
        hist = row_now["macd_hist"]

        if (
            np.isnan(rsi_now)
            or np.isnan(ema_now)
            or np.isnan(rsi_prev)
            or np.isnan(ema_prev)
            or np.isnan(hist)
        ):
            signals[i] = 1 if in_position else 0
            continue

        # Tocó 30 en ventana reciente (incluye vela actual)
        lo = i - touch_lookback_bars + 1
        lo = max(lo, 0)
        window_rsi = df["rsi"].iloc[lo : i + 1]
        touched_30 = window_rsi.min() <= rsi_oversold

        # EMA(14) del RSI cruza al alza respecto al RSI
        cross_ema_up_through_rsi = ema_prev < rsi_prev and ema_now >= rsi_now

        macd_bullish = hist > 0

        if not in_position:
            if touched_30 and cross_ema_up_through_rsi and macd_bullish:
                in_position = True
                signals[i] = 1
            else:
                signals[i] = 0
        else:
            # Salida: solo RSI de velas 1D (rsi_1d), nunca el RSI de 4h
            if not np.isnan(rsi_d) and rsi_d >= rsi_overbought_daily:
                in_position = False
                signals[i] = 0
            else:
                signals[i] = 1

    df["senal"] = signals
    df["entrada"] = (df["senal"] == 1) & (df["senal"].shift(1, fill_value=0) == 0)
    df["salida"] = (df["senal"] == 0) & (df["senal"].shift(1, fill_value=0) == 1)

    return df


def build_trades_log(df: pd.DataFrame) -> list[dict]:
    """Construye una lista de trades a partir de columnas entrada/salida."""
    trades: list[dict] = []
    entry_date = None
    entry_price = None

    for _, row in df.iterrows():
        if row["entrada"]:
            entry_date = row["fecha"]
            entry_price = row["cierre"]
        elif row["salida"] and entry_date is not None:
            exit_price = row["cierre"]
            ret = exit_price / entry_price - 1
            trades.append(
                {
                    "entrada_fecha": entry_date,
                    "entrada_precio": entry_price,
                    "salida_fecha": row["fecha"],
                    "salida_precio": exit_price,
                    "retorno": ret,
                }
            )
            entry_date = None
            entry_price = None

    if entry_date is not None:
        last = df.iloc[-1]
        ret = last["cierre"] / entry_price - 1
        trades.append(
            {
                "entrada_fecha": entry_date,
                "entrada_precio": entry_price,
                "salida_fecha": last["fecha"],
                "salida_precio": last["cierre"],
                "retorno": ret,
            }
        )

    return trades


def compute_metrics(df: pd.DataFrame, trades: list[dict]) -> dict:
    """Calcula métricas básicas de la estrategia."""
    df = df.copy()
    df["retorno"] = df["cierre"].pct_change()
    df["retorno_estrategia"] = df["senal"].shift(1) * df["retorno"]
    df = df.dropna(subset=["retorno_estrategia"])

    retorno_total = (1 + df["retorno_estrategia"]).prod() - 1
    retorno_buyhold = (1 + df["retorno"]).prod() - 1

    cum = (1 + df["retorno_estrategia"]).cumprod()
    peak = cum.cummax()
    max_dd = ((cum - peak) / peak).min()

    n_trades = len(trades)
    if n_trades > 0:
        win_rate = sum(1 for t in trades if t["retorno"] > 0) / n_trades
    else:
        win_rate = 0.0

    mean_r = df["retorno_estrategia"].mean()
    std_r = df["retorno_estrategia"].std()
    sharpe = (mean_r / std_r * np.sqrt(252 * 6)) if std_r > 0 else 0.0  # ~6 velas 4h por día

    return {
        "retorno_total": retorno_total,
        "retorno_buyhold": retorno_buyhold,
        "max_drawdown": max_dd,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "sharpe": sharpe,
    }


def save_trades_csv(trades: list[dict], symbol: str) -> Path:
    results_dir = ROOT / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{symbol}_trades_{TRADES_SUFFIX}.csv"

    if not trades:
        print(f"[RSI+MACD] No hay trades que guardar para {symbol}.")
        return csv_path

    df_trades = pd.DataFrame(trades)
    df_trades.index = range(1, len(df_trades) + 1)
    df_trades.index.name = "#"
    df_trades.to_csv(csv_path)
    print(f"[RSI+MACD] Trades guardados en {csv_path}")
    return csv_path


def print_trades(trades: list[dict]) -> None:
    if not trades:
        print("\n[RSI+MACD] Sin operaciones en este periodo.")
        return

    print(f"\n[RSI+MACD] Log de operaciones ({len(trades)} trades):")
    print(f"  {'#':>3}  {'Entrada':>19}  {'Precio':>12}  {'Salida':>19}  {'Precio':>12}  {'Retorno':>9}")
    print(f"  {'---':>3}  {'---':>19}  {'---':>12}  {'---':>19}  {'---':>12}  {'---':>9}")
    for i, t in enumerate(trades, 1):
        e_fecha = str(t["entrada_fecha"])[:19]
        s_fecha = str(t["salida_fecha"])[:19]
        ret_str = f"{t['retorno']:+.2%}"
        print(
            f"  {i:>3}  {e_fecha:>19}  {t['entrada_precio']:>12.2f}"
            f"  {s_fecha:>19}  {t['salida_precio']:>12.2f}  {ret_str:>9}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estrategia RSI + MACD (4h) con salida por RSI diario (1D)."
    )
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Símbolo (ej. BTCUSDT)")
    parser.add_argument("--timeframe", type=str, default="4h", help="TF de entrada (por defecto 4h)")
    parser.add_argument("--days", type=int, default=365, help="Días de histórico")
    parser.add_argument("--touch-lookback", type=int, default=12, help="Velas 4h para detectar toque a RSI 30")
    args = parser.parse_args()

    df = build_strategy_df(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        touch_lookback_bars=args.touch_lookback,
    )
    trades = build_trades_log(df)
    metrics = compute_metrics(df, trades)

    print_trades(trades)
    print("\n[RSI+MACD] Métricas:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    save_trades_csv(trades, symbol=args.symbol)


if __name__ == "__main__":
    main()
