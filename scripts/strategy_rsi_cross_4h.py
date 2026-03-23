"""
Estrategia mínima solo en 4h: entradas y salidas por cruce de RSI.

Entrada (long, 4h): cruce de RSI de abajo hacia arriba del nivel configurado
    (p. ej. 32): RSI[vela anterior] < entry_cross_level y RSI[vela actual] >= entry_cross_level.
    Opcional: entry_touch_lookback_bars > 0 exige que en esa ventana el RSI haya estado
    <= entry_cross_level (toque previo en zona).

Salida: lo que ocurra primero (stop loss siempre en velas 4h):
    - RSI cruza de arriba hacia abajo exit_cross_level en el timeframe configurado
      (exit_timeframe: "4h" o "1D"), o
    - Stop loss: el bajo de la vela 4h toca entrada × (1 - stop_loss_pct).
      Si stop_loss_pct = 0, no hay stop.

    Con exit_timeframe "1D": el cruce se evalúa en velas diarias; la orden de salida
    se toma al cierre del día donde RSI cruza; la ejecución es en el primer cierre 4h
    posterior a ese instante (no repinta intradía).

Uso:
    python -m scripts.strategy_rsi_cross_4h --config configs/strategy_rsi_cross.yaml
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

from src.agents.data_agent import DEFAULT_BYBIT_CATEGORY, DataAgent

TRADES_SUFFIX_BASE = "rsi_cross_4h"


def _deep_merge_inplace(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge_inplace(base[k], v)
        else:
            base[k] = copy.deepcopy(v)


def load_config(path: str | Path | None) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "symbol": "BTCUSDT",
        "timeframe": "4h",
        "history_days": 2000,
        "bybit_category": DEFAULT_BYBIT_CATEGORY,
        "rsi_period": 14,
        "entry_cross_level": 32.0,
        "exit_cross_level": 50.0,
        "entry_touch_lookback_bars": 0,
        "stop_loss_pct": 0.02,
        "exit_timeframe": "4h",
        # Multiplicador de retorno por barra en la curva de equity (apalancamiento simulado).
        # 1.0 = spot. >1 aumenta retorno_total y drawdowns proporcionalmente (aprox.).
        "position_leverage": 1.0,
        # Solo para mostrar métricas en € (el backtest sigue siendo en retornos).
        "initial_capital_eur": 10000.0,
    }
    cfg = copy.deepcopy(defaults)
    if path:
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            with open(p, encoding="utf-8") as f:
                _deep_merge_inplace(cfg, yaml.safe_load(f) or {})
        else:
            print(f"[rsi_cross_4h] No existe {p}, usando defaults.")
    return cfg


def build_df(symbol: str, cfg: dict[str, Any], timeframe: str | None = None) -> pd.DataFrame:
    agent = DataAgent(category=str(cfg.get("bybit_category", DEFAULT_BYBIT_CATEGORY)))
    tf = timeframe or cfg["timeframe"]
    days = int(cfg["history_days"])
    df = agent.get_ohlcv(symbol=symbol, timeframe=tf, days=days)
    df = df.sort_values("fecha").reset_index(drop=True)
    df["fecha"] = pd.to_datetime(df["fecha"])
    w = int(cfg["rsi_period"])
    df["rsi"] = RSIIndicator(close=df["cierre"], window=w).rsi()
    return df


def _daily_rsi_exit_timestamps(df_1d: pd.DataFrame, exit_lvl: float) -> list[pd.Timestamp]:
    """
    Para cada vela diaria donde RSI cruza de arriba hacia abajo exit_lvl,
    devuelve el instante en que el cruce queda confirmado (cierre de esa vela).

    Bybit usa startTime como `fecha`; el cierre del día D es fecha[D] + 1 día.
    """
    rsi = df_1d["rsi"].to_numpy(float)
    fechas = pd.to_datetime(df_1d["fecha"])
    out: list[pd.Timestamp] = []
    for d in range(1, len(df_1d)):
        rp, rn = rsi[d - 1], rsi[d]
        if np.isnan(rp) or np.isnan(rn):
            continue
        if rp > exit_lvl and rn <= exit_lvl:
            # cruce confirmado al cierre de la vela d
            close_ts = fechas.iloc[d] + pd.Timedelta(days=1)
            out.append(pd.Timestamp(close_ts))
    return sorted(out)


def _first_exit_ts_after(
    entry_ts: pd.Timestamp, exit_ts_list: list[pd.Timestamp]
) -> pd.Timestamp | None:
    for ts in exit_ts_list:
        if ts > entry_ts:
            return ts
    return None


def run_signals(
    df: pd.DataFrame,
    entry_lvl: float,
    exit_lvl: float,
    touch_lookback: int = 0,
    stop_loss_pct: float = 0.0,
    exit_timeframe: str = "4h",
    df_1d: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Señales + trades con stop loss en 4h; salida por RSI en 4h o en 1D."""
    exit_tf = (exit_timeframe or "4h").strip().upper()
    use_1d_exit = exit_tf in ("1D", "D", "1d", "DAY", "DAILY")
    daily_exit_ts: list[pd.Timestamp] = []
    if use_1d_exit:
        if df_1d is None or len(df_1d) < 2:
            raise ValueError("exit_timeframe 1D requiere df_1d con RSI calculado")
        daily_exit_ts = _daily_rsi_exit_timestamps(df_1d, exit_lvl)

    rsi = df["rsi"].to_numpy(float)
    close = df["cierre"].to_numpy(float)
    low = df["bajo"].to_numpy(float)
    n = len(rsi)
    signals = np.zeros(n, dtype=int)
    in_pos = False
    entry_price = np.nan
    entry_i = -1

    touch_lb = max(0, int(touch_lookback))
    if touch_lb == 1:
        touch_lb = 2
    rsi_roll_min: np.ndarray | None = None
    if touch_lb > 0:
        rsi_roll_min = (
            pd.Series(rsi).rolling(window=touch_lb, min_periods=1).min().to_numpy()
        )

    use_sl = stop_loss_pct > 0
    trades: list[dict] = []
    pending_1d_exit_ts: pd.Timestamp | None = None

    def _close_trade(i_bar: int, px: float, reason: str) -> None:
        nonlocal in_pos, entry_price, entry_i, pending_1d_exit_ts
        trades.append(
            {
                "entrada_fecha": df.iloc[entry_i]["fecha"],
                "entrada_precio": entry_price,
                "salida_fecha": df.iloc[i_bar]["fecha"],
                "salida_precio": px,
                "retorno": px / entry_price - 1.0,
                "motivo": reason,
            }
        )
        in_pos = False
        entry_price = np.nan
        entry_i = -1
        pending_1d_exit_ts = None

    for i in range(1, n):
        rp, rn = rsi[i - 1], rsi[i]
        if np.isnan(rp) or np.isnan(rn):
            signals[i] = 1 if in_pos else 0
            continue

        if in_pos:
            # Stop loss: bajo de la vela toca el nivel
            if use_sl:
                sl_price = entry_price * (1.0 - stop_loss_pct)
                if low[i] <= sl_price:
                    exit_px = min(close[i], sl_price)
                    signals[i] = 0
                    _close_trade(i, exit_px, "stop_loss")
                    continue

            if use_1d_exit:
                t_now = pd.Timestamp(df.iloc[i]["fecha"])
                if pending_1d_exit_ts is not None and t_now >= pending_1d_exit_ts:
                    signals[i] = 0
                    _close_trade(i, close[i], "rsi_cross_1d")
                    pending_1d_exit_ts = None
                    continue
                signals[i] = 1
            else:
                # RSI cruza hacia abajo el nivel de salida (4h)
                cross_down = rp > exit_lvl and rn <= exit_lvl
                if cross_down:
                    signals[i] = 0
                    _close_trade(i, close[i], "rsi_cross")
                else:
                    signals[i] = 1
        else:
            cross_up = rp < entry_lvl and rn >= entry_lvl
            touch_ok = True
            if touch_lb > 0 and rsi_roll_min is not None:
                touch_ok = not np.isnan(rsi_roll_min[i]) and rsi_roll_min[i] <= entry_lvl
            if cross_up and touch_ok:
                in_pos = True
                entry_price = close[i]
                entry_i = i
                signals[i] = 1
                if use_1d_exit:
                    et = pd.Timestamp(df.iloc[i]["fecha"])
                    pending_1d_exit_ts = _first_exit_ts_after(et, daily_exit_ts)
            else:
                signals[i] = 0

    # Posición abierta al final del periodo
    if in_pos and entry_i >= 0:
        _close_trade(n - 1, close[n - 1], "fin_datos")

    out = df.copy()
    out["senal"] = signals
    out["entrada"] = (out["senal"] == 1) & (out["senal"].shift(1, fill_value=0) == 0)
    out["salida"] = (out["senal"] == 0) & (out["senal"].shift(1, fill_value=0) == 1)
    out["retorno"] = out["cierre"].pct_change()
    out["retorno_estrategia"] = out["senal"].shift(1) * out["retorno"]
    return out, trades


def compute_metrics(
    sim: pd.DataFrame,
    trades: list[dict],
    position_leverage: float = 1.0,
    initial_capital_eur: float | None = None,
) -> dict:
    """
    position_leverage: factor por el que se multiplica retorno_estrategia en cada barra
    antes de componer (modelo simple tipo margen/futuros). buy_hold no se escala.

    initial_capital_eur: si > 0, añade capital final / ganancias / max DD en €
    (asumiendo todo el capital asignado a la estrategia o al buy & hold según corresponda).
    """
    lev = float(position_leverage)
    if lev <= 0:
        lev = 1.0
    d = sim.dropna(subset=["retorno_estrategia"])
    r_strat = d["retorno_estrategia"] * lev
    rt = (1 + r_strat).prod() - 1
    bh = (1 + d["retorno"]).prod() - 1
    cum = (1 + r_strat).cumprod()
    peak = cum.cummax()
    mdd = ((cum - peak) / peak).min()
    nt = len(trades)
    wr = sum(1 for t in trades if t["retorno"] > 0) / nt if nt else 0.0
    sr = 0.0
    if r_strat.std() > 0:
        sr = r_strat.mean() / r_strat.std() * np.sqrt(252 * 6)
    ratio_vs_bh = (rt / bh) if bh > 0 else float("nan")
    out: dict[str, Any] = {
        "retorno_total": rt,
        "retorno_buyhold": bh,
        "ratio_vs_buyhold": ratio_vs_bh,
        "position_leverage": lev,
        "max_drawdown": mdd,
        "n_trades": nt,
        "win_rate": wr,
        "sharpe": sr,
    }
    cap0 = initial_capital_eur
    if cap0 is not None and float(cap0) > 0:
        c0 = float(cap0)
        out["initial_capital_eur"] = c0
        out["capital_final_estrategia_eur"] = c0 * (1.0 + rt)
        out["ganancia_estrategia_eur"] = out["capital_final_estrategia_eur"] - c0
        out["capital_final_buyhold_eur"] = c0 * (1.0 + bh)
        out["ganancia_buyhold_eur"] = out["capital_final_buyhold_eur"] - c0
        equity_eur = cum * c0
        peak_eur = equity_eur.cummax()
        # Caída máxima en € desde el máximo histórico de la curva (≤ 0)
        out["max_drawdown_eur"] = float((equity_eur - peak_eur).min())
    return out


def save_trades(trades: list[dict], symbol: str, suffix: str) -> Path:
    out = ROOT / "data" / "results"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{symbol}_trades_{suffix}.csv"
    if not trades:
        print(f"[rsi_cross_4h] Sin trades para {symbol}")
        return path
    pd.DataFrame(trades, index=range(1, len(trades) + 1)).to_csv(path, index_label="#")
    print(f"[rsi_cross_4h] Trades → {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="RSI cruces solo 4h")
    ap.add_argument("--config", type=str, default="configs/strategy_rsi_cross.yaml")
    ap.add_argument("--symbol", type=str, default=None)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument(
        "--initial-capital-eur",
        type=float,
        default=None,
        help="Capital inicial en € para métricas (por defecto: initial_capital_eur del YAML, 10000).",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    sym = args.symbol or cfg["symbol"]
    if args.days is not None:
        cfg["history_days"] = args.days
    cap_eur = args.initial_capital_eur
    if cap_eur is None:
        cap_eur = float(cfg.get("initial_capital_eur", 10000.0))

    el = float(cfg["entry_cross_level"])
    xl = float(cfg["exit_cross_level"])
    tlb = int(cfg.get("entry_touch_lookback_bars", 0))
    sl = float(cfg.get("stop_loss_pct", 0.0))
    exit_tf = str(cfg.get("exit_timeframe", "4h"))
    lev = float(cfg.get("position_leverage", 1.0))

    trades_suffix = TRADES_SUFFIX_BASE
    df_1d: pd.DataFrame | None = None
    exit_tf_u = exit_tf.strip().upper()
    if exit_tf_u in ("1D", "D", "DAY", "DAILY"):
        trades_suffix = f"{TRADES_SUFFIX_BASE}_1dexit"
        df_1d = build_df(sym, cfg, timeframe="1D")

    print(f"[rsi_cross_4h] {sym} | entrada {cfg['timeframe']} | RSI({cfg['rsi_period']})")
    print(f"  Entrada: cruce ↑ nivel {el} (RSI_prev < {el} y RSI_now >= {el})")
    if tlb > 0:
        print(f"           + toque en ventana: mínimo RSI últimas {tlb} velas <= {el}")
    if exit_tf_u in ("1D", "D", "DAY", "DAILY"):
        print(f"  Salida: cruce ↓ nivel {xl} en 1D (ejecución en primera vela 4h tras cierre día)")
    else:
        print(f"  Salida: cruce ↓ nivel {xl} en {cfg['timeframe']}")
    if sl > 0:
        print(f"  Stop loss: -{sl:.0%} desde precio de entrada (velas 4h)")
    print(
        f"  Métricas: position_leverage = {lev:g} "
        f"({'sin multiplicador (1×), comparable a spot' if lev == 1.0 else 'apalancamiento simulado en retornos'})"
    )
    if lev != 1.0:
        print(
            f"    → retorno_estrategia × {lev:g} por barra; buy_hold sin escalar"
        )

    df = build_df(sym, cfg)
    sim, trades = run_signals(
        df,
        el,
        xl,
        touch_lookback=tlb,
        stop_loss_pct=sl,
        exit_timeframe=exit_tf,
        df_1d=df_1d,
    )
    m = compute_metrics(
        sim, trades, position_leverage=lev, initial_capital_eur=cap_eur
    )

    print(f"\n  Trades: {m['n_trades']} | win_rate: {m['win_rate']:.2%}")
    print(
        f"  retorno_total: {m['retorno_total']:.4f} | buy_hold: {m['retorno_buyhold']:.4f} "
        f"| ratio (estrategia/buy_hold): {m['ratio_vs_buyhold']:.4f}"
    )
    print(f"  max_dd: {m['max_drawdown']:.4f} | sharpe: {m['sharpe']:.4f}")
    if "initial_capital_eur" in m:
        c0 = m["initial_capital_eur"]
        print(f"\n  --- Métricas con capital inicial {c0:,.2f} € ---")
        print(
            f"  Estrategia:  final {m['capital_final_estrategia_eur']:,.2f} €  "
            f"(ganancia {m['ganancia_estrategia_eur']:+,.2f} €)"
        )
        print(
            f"  Buy & hold: final {m['capital_final_buyhold_eur']:,.2f} €  "
            f"(ganancia {m['ganancia_buyhold_eur']:+,.2f} €)"
        )
        print(
            f"  Max. drawdown (desde máximo de equity, estrategia): "
            f"{m['max_drawdown_eur']:,.2f} €  ({m['max_drawdown']:.2%})"
        )

    for i, t in enumerate(trades[:25], 1):
        motivo = t.get("motivo", "")
        print(
            f"  {i:>3}  {str(t['entrada_fecha'])[:19]}  {t['entrada_precio']:.2f}  →  "
            f"{str(t['salida_fecha'])[:19]}  {t['salida_precio']:.2f}  {t['retorno']:+.2%}  [{motivo}]"
        )
    if len(trades) > 25:
        print(f"  ... ({len(trades) - 25} más)")

    save_trades(trades, sym, trades_suffix)


if __name__ == "__main__":
    main()
