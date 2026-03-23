"""
Estrategia long en 4h con EMA 200, RSI, filtro diario, SL/TP, divergencia y volumen.

Condiciones de entrada (todas):
  1) RSI(14) en 4h en la vela anterior < 30 y en la actual cruza hacia arriba el 30
     (RSI_prev < 30 y RSI_now >= 30).
  2) Tendencia: cierre > EMA(200) en 4h.
  3) Filtro mercado: cierre diario (1D) >= EMA(200) diaria (sin operar si precio < EMA200 1D).
  4) Divergencia alcista RSI (dos mínimos en bajo: precio más bajo, RSI más alto).
  5) Volumen: volumen actual > volumen anterior y > SMA(volumen, N).

Salidas (resto de posición hasta cerrar todo):
  - Stop loss: -5% desde precio de entrada (toca el mínimo de la vela).
  - Take profit total: +10% desde entrada (toca el máximo).
  - Take profit parcial (opcional): +5% cierra una fracción (p. ej. 50%).
  - RSI(14) diario > 70.

Misma vela con SL y TP: prioridad conservadora (por defecto SL primero).

Uso:
    python -m scripts.strategy_long_rules --config configs/strategy_long.yaml
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from ta.momentum import RSIIndicator

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.data_agent import DataAgent

TRADES_SUFFIX = "long_rules"


def _deep_merge_inplace(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge_inplace(base[k], v)
        else:
            base[k] = copy.deepcopy(v)


def load_config(path: str | Path | None) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "symbol": "BTCUSDT",
        "timeframe_base": "4h",
        "history_days": 2000,
        "rsi_period": 14,
        "rsi_entry_cross_level": 30,
        "ema_trend_4h": 200,
        "ema_filter_1d": 200,
        "volume_ma_period": 20,
        "entry": {
            "require_divergence": True,
            "require_volume_increase": True,
        },
        "divergence_lookback": 40,
        "divergence_pivot_width": 2,
        "divergence_max_bars_since_pivot": 20,
        "exit": {
            "stop_loss_pct": 0.05,
            "take_profit_total_pct": 0.10,
            "rsi_overbought_1d": 70,
            "same_bar_priority": "stop_first",
        },
        "partial_take_profit": {
            "enabled": True,
            "profit_pct": 0.05,
            "fraction": 0.5,
        },
    }
    cfg = copy.deepcopy(defaults)
    if path:
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                _deep_merge_inplace(cfg, yaml.safe_load(f) or {})
        else:
            print(f"[strategy_long] Aviso: no existe {p}, usando defaults.")
    return cfg


def _data_agent_from_cfg(cfg: dict) -> DataAgent:
    """DataAgent con category si el constructor lo soporta."""
    import inspect

    params = inspect.signature(DataAgent.__init__).parameters
    if "category" in params:
        cat = str(cfg.get("bybit_category", "linear")).strip().lower()
        return DataAgent(category=cat)
    return DataAgent()


def build_features(
    symbol: str,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    tf = cfg["timeframe_base"]
    days = int(cfg["history_days"])
    rsi_w = int(cfg["rsi_period"])
    ema4 = int(cfg["ema_trend_4h"])
    ema1d = int(cfg["ema_filter_1d"])
    vol_ma = int(cfg["volume_ma_period"])

    agent = _data_agent_from_cfg(cfg)
    df = agent.get_ohlcv(symbol=symbol, timeframe=tf, days=days)
    df = df.sort_values("fecha").reset_index(drop=True)
    df["fecha"] = pd.to_datetime(df["fecha"])

    rsi_ind = RSIIndicator(close=df["cierre"], window=rsi_w)
    df["rsi"] = rsi_ind.rsi()
    df["ema200_4h"] = df["cierre"].ewm(span=ema4, adjust=False).mean()
    df["vol_sma"] = df["volumen"].rolling(vol_ma).mean()

    daily = agent.get_ohlcv(symbol=symbol, timeframe="1D", days=days)
    daily = daily.sort_values("fecha").reset_index(drop=True)
    daily["fecha"] = pd.to_datetime(daily["fecha"])
    daily["ema200_1d"] = daily["cierre"].ewm(span=ema1d, adjust=False).mean()
    rsi_d = RSIIndicator(close=daily["cierre"], window=rsi_w)
    daily["rsi_1d"] = rsi_d.rsi()
    daily_m = daily[["fecha", "cierre", "ema200_1d", "rsi_1d"]].rename(
        columns={"cierre": "cierre_1d"}
    )

    df = pd.merge_asof(df, daily_m, on="fecha", direction="backward")
    return df


def bullish_rsi_divergence(
    low: np.ndarray,
    rsi: np.ndarray,
    i: int,
    width: int,
    max_since: int,
) -> bool:
    """
    Divergencia alcista en mínimos: dos swing lows con precio LL y RSI HL.
    El segundo pivote debe estar cerca de la vela i (señal).
    """
    if i < width * 2 + 5:
        return False
    start = max(width, i - 100)
    lows_idx: list[int] = []
    for j in range(start + width, i - width + 1):
        seg = low[j - width : j + width + 1]
        if seg.size == 0 or np.all(np.isnan(seg)):
            continue
        lj = low[j]
        if np.isnan(lj):
            continue
        if lj <= np.nanmin(seg) + 1e-12:
            lows_idx.append(j)
    if len(lows_idx) < 2:
        return False
    j1, j2 = lows_idx[-2], lows_idx[-1]
    if i - j2 > max_since:
        return False
    p1, p2 = low[j1], low[j2]
    r1, r2 = rsi[j1], rsi[j2]
    if np.isnan(r1) or np.isnan(r2):
        return False
    return p2 < p1 and r2 > r1


def run_backtest(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[dict]]:
    level = float(cfg["rsi_entry_cross_level"])
    ex = cfg["exit"]
    sl_pct = float(ex["stop_loss_pct"])
    tp_full_pct = float(ex["take_profit_total_pct"])
    rsi_ob = float(ex["rsi_overbought_1d"])
    same_bar = ex.get("same_bar_priority", "stop_first")

    ptp = cfg.get("partial_take_profit", {})
    ptp_on = bool(ptp.get("enabled", False))
    ptp_pct = float(ptp.get("profit_pct", 0.05))
    ptp_frac = float(ptp.get("fraction", 0.5))

    div_w = int(cfg["divergence_pivot_width"])
    div_max = int(cfg["divergence_max_bars_since_pivot"])

    low = df["bajo"].to_numpy(float)
    high = df["alto"].to_numpy(float)
    close = df["cierre"].to_numpy(float)
    rsi = df["rsi"].to_numpy(float)
    vol = df["volumen"].to_numpy(float)
    vol_sma = df["vol_sma"].to_numpy(float)
    ema200_4h = df["ema200_4h"].to_numpy(float)
    rsi_1d = df["rsi_1d"].to_numpy(float)
    cierre_1d = df["cierre_1d"].to_numpy(float)
    ema200_1d = df["ema200_1d"].to_numpy(float)

    n = len(df)
    qty = np.zeros(n, dtype=float)
    signals = np.zeros(n, dtype=int)

    trades: list[dict] = []
    entry_price = np.nan
    entry_i = -1
    partial_taken = False
    # PnL acumulado del trade como fracción del capital (1 unidad): sum(fr_cerrada * (px/E - 1))
    realized_pnl = 0.0
    remaining = 0.0
    exit_reasons: list[str] = []

    def finalize_trade(i_bar: int, last_px: float, last_reason: str) -> None:
        nonlocal entry_price, entry_i, partial_taken, realized_pnl, remaining, exit_reasons
        total_ret = realized_pnl + remaining * (last_px / entry_price - 1.0)
        reasons = "+".join(exit_reasons + [last_reason])
        trades.append(
            {
                "entrada_fecha": df.iloc[entry_i]["fecha"],
                "entrada_precio": entry_price,
                "salida_fecha": df.iloc[i_bar]["fecha"],
                "salida_precio": last_px,
                "retorno": total_ret,
                "motivo_salida": reasons,
            }
        )
        entry_price = np.nan
        entry_i = -1
        partial_taken = False
        realized_pnl = 0.0
        remaining = 0.0
        exit_reasons = []

    for i in range(1, n):
        rsi_prev, rsi_now = rsi[i - 1], rsi[i]
        q_prev = qty[i - 1]

        if q_prev > 0 and not np.isnan(entry_price):
            sl_price = entry_price * (1.0 - sl_pct)
            tp_price = entry_price * (1.0 + tp_full_pct)
            ptp_price = entry_price * (1.0 + ptp_pct)

            hit_sl = low[i] <= sl_price
            hit_tp_full = high[i] >= tp_price
            hit_ptp = (
                ptp_on
                and (not partial_taken)
                and remaining >= 1.0 - 1e-9
                and high[i] >= ptp_price
            )
            hit_rsi = (not np.isnan(rsi_1d[i])) and rsi_1d[i] >= rsi_ob

            exit_price_sl = min(close[i], sl_price)
            exit_price_tp = min(close[i], tp_price)
            exit_price_ptp = min(close[i], ptp_price)

            chosen = None
            if same_bar == "stop_first":
                if hit_sl:
                    chosen = ("sl", exit_price_sl)
                elif hit_tp_full:
                    chosen = ("tp_full", exit_price_tp)
                elif hit_ptp:
                    chosen = ("partial_tp", exit_price_ptp)
                elif hit_rsi:
                    chosen = ("rsi_1d", close[i])
            else:
                if hit_tp_full:
                    chosen = ("tp_full", exit_price_tp)
                elif hit_sl:
                    chosen = ("sl", exit_price_sl)
                elif hit_ptp:
                    chosen = ("partial_tp", exit_price_ptp)
                elif hit_rsi:
                    chosen = ("rsi_1d", close[i])

            if chosen is not None:
                reason, px = chosen
                if reason == "partial_tp":
                    realized_pnl += ptp_frac * (px / entry_price - 1.0)
                    remaining = 1.0 - ptp_frac
                    partial_taken = True
                    exit_reasons.append("partial_tp")
                    qty[i] = remaining
                    signals[i] = 1
                    continue

                qty[i] = 0.0
                signals[i] = 0
                finalize_trade(i, px, reason)
                continue

            qty[i] = remaining if partial_taken else q_prev
            signals[i] = 1
            continue

        # Sin posición
        qty[i] = 0.0
        signals[i] = 0

        if np.isnan(rsi_now) or np.isnan(rsi_prev) or np.isnan(ema200_4h[i]):
            continue
        if np.isnan(cierre_1d[i]) or np.isnan(ema200_1d[i]):
            continue

        ent = cfg.get("entry", {})
        req_div = bool(ent.get("require_divergence", True))
        req_vol = bool(ent.get("require_volume_increase", True))

        cross_30 = rsi_prev < level and rsi_now >= level
        trend_4h = close[i] > ema200_4h[i]
        market_ok = cierre_1d[i] >= ema200_1d[i]
        div_ok = (
            bullish_rsi_divergence(low, rsi, i, div_w, div_max) if req_div else True
        )
        vol_ok = True
        if req_vol:
            vol_ok = (
                not np.isnan(vol_sma[i])
                and vol[i] > vol[i - 1]
                and vol[i] > vol_sma[i]
            )

        if cross_30 and trend_4h and market_ok and div_ok and vol_ok:
            qty[i] = 1.0
            signals[i] = 1
            entry_price = close[i]
            entry_i = i
            partial_taken = False
            realized_pnl = 0.0
            remaining = 1.0
            exit_reasons = []

    df = df.copy()
    df["senal"] = signals
    df["qty"] = qty
    df["retorno"] = df["cierre"].pct_change()
    df["retorno_estrategia"] = df["qty"].shift(1).fillna(0) * df["retorno"]

    if entry_i >= 0 and not np.isnan(entry_price) and remaining > 0:
        last = n - 1
        px = close[last]
        finalize_trade(last, px, "fin_datos")

    return df, trades


def compute_metrics(df: pd.DataFrame, trades: list[dict]) -> dict:
    d = df.dropna(subset=["retorno_estrategia"])
    retorno_total = (1 + d["retorno_estrategia"]).prod() - 1
    retorno_buyhold = (1 + d["retorno"]).prod() - 1
    cum = (1 + d["retorno_estrategia"]).cumprod()
    peak = cum.cummax()
    max_dd = ((cum - peak) / peak).min()
    n_trades = len(trades)
    win_rate = sum(1 for t in trades if t["retorno"] > 0) / n_trades if n_trades else 0.0
    mean_r = d["retorno_estrategia"].mean()
    std_r = d["retorno_estrategia"].std()
    sharpe = (mean_r / std_r * np.sqrt(252 * 6)) if std_r > 0 else 0.0
    return {
        "retorno_total": retorno_total,
        "retorno_buyhold": retorno_buyhold,
        "max_drawdown": max_dd,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "sharpe": sharpe,
    }


def save_trades_csv(trades: list[dict], symbol: str) -> Path:
    out = ROOT / "data" / "results"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{symbol}_trades_{TRADES_SUFFIX}.csv"
    if not trades:
        print(f"[strategy_long] Sin trades para {symbol}")
        return path
    pd.DataFrame(trades, index=range(1, len(trades) + 1)).to_csv(path, index_label="#")
    print(f"[strategy_long] Trades → {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Estrategia long 4h (reglas completas)")
    ap.add_argument("--config", type=str, default="configs/strategy_long.yaml")
    ap.add_argument("--symbol", type=str, default=None)
    ap.add_argument("--days", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    symbol = args.symbol or cfg["symbol"]
    if args.days is not None:
        cfg["history_days"] = args.days

    print(f"[strategy_long] {symbol} | {cfg['timeframe_base']} | divergencia + volumen + EMA200")
    df = build_features(symbol, cfg)
    df, trades = run_backtest(df, cfg)
    metrics = compute_metrics(df, trades)

    print(f"\n[strategy_long] Trades: {metrics['n_trades']} | win_rate: {metrics['win_rate']:.2%}")
    print(f"  retorno_total: {metrics['retorno_total']:.4f} | buy_hold: {metrics['retorno_buyhold']:.4f}")
    print(f"  max_dd: {metrics['max_drawdown']:.4f} | sharpe: {metrics['sharpe']:.4f}")

    for i, t in enumerate(trades[:20], 1):
        print(
            f"  {i:>3}  {str(t['entrada_fecha'])[:19]}  {t['entrada_precio']:.2f}  →  "
            f"{str(t['salida_fecha'])[:19]}  {t['salida_precio']:.2f}  {t['retorno']:+.2%}  [{t.get('motivo_salida','')}]"
        )
    if len(trades) > 20:
        print(f"  ... ({len(trades) - 20} más)")

    save_trades_csv(trades, symbol)


if __name__ == "__main__":
    main()
