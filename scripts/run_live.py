"""
Script de ejecución live / paper trading.

Conecta las estrategias existentes (RSI cross, XGBoost) con Bybit
para ejecutar órdenes en testnet o producción.

Modos:
  --paper       Simulación local sin conexión a exchange (no necesita API keys)
  (sin --paper) Conecta a Bybit (testnet o mainnet según YAML)

Uso:
  python -m scripts.run_live --config configs/execution.yaml --paper
  python -m scripts.run_live --config configs/execution.yaml
  python -m scripts.run_live --symbol ETHUSDT --paper
"""

from __future__ import annotations

import argparse
import os
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from ta.momentum import RSIIndicator

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.data_agent import DEFAULT_BYBIT_CATEGORY, DataAgent
from src.agents.execution_agent import ExecutionAgent, PaperExecutionAgent
from src.agents.regime_agent import (
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_SIDEWAYS,
    RegimeAgent,
)
from src.notifications.email import send_trade_email

_running = True

_SIGNALS_DIR = ROOT / "data" / "signal_logs"
_SIGNALS_FILE = _SIGNALS_DIR / "signals.jsonl"


def _notify_email_recipients() -> list[str]:
    raw = os.getenv("NOTIFY_EMAILS", "")
    return [e.strip() for e in raw.split(",") if e.strip()]


def log_signal(record: dict[str, Any]) -> None:
    _SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_SIGNALS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _handle_signal(signum, frame):
    global _running
    print("\n[LiveRunner] Señal de parada recibida. Finalizando tras el ciclo actual...")
    _running = False


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def create_execution_agent(cfg: dict, paper: bool = False) -> ExecutionAgent | PaperExecutionAgent:
    """Crea el agente de ejecución según el modo."""
    if paper:
        paper_cfg = cfg.get("paper", {})
        return PaperExecutionAgent(
            initial_balance=paper_cfg.get("initial_balance_usdt", 10_000),
            category=cfg.get("bybit_category", DEFAULT_BYBIT_CATEGORY),
        )

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("BYBIT_API_KEY", "")
    api_secret = os.getenv("BYBIT_API_SECRET", "")

    if not api_key or not api_secret or api_key == "tu_api_key_aqui":
        print("[LiveRunner] ERROR: API keys no configuradas.")
        print("  1. Copia .env.example → .env")
        print("  2. Rellena BYBIT_API_KEY y BYBIT_API_SECRET")
        print("  3. Para testnet: https://testnet.bybit.com → API Management")
        sys.exit(1)

    return ExecutionAgent(
        api_key=api_key,
        api_secret=api_secret,
        testnet=cfg.get("testnet", True),
        category=cfg.get("bybit_category", DEFAULT_BYBIT_CATEGORY),
        tld=cfg.get("bybit_tld", "com"),
    )


def _ml_all_timeframes(ml_cfg: dict[str, Any]) -> list[str]:
    base = ml_cfg.get("timeframe_base") or ml_cfg.get("timeframe", "4h")
    tfs: list[str] = [base]
    for _name, h in (ml_cfg.get("higher_timeframes") or {}).items():
        if h.get("enabled") and h.get("timeframe"):
            tf = str(h["timeframe"])
            if tf not in tfs:
                tfs.append(tf)
    return tfs


def fetch_latest_data(
    symbol: str, cfg: dict, timeframe: str | None = None
) -> pd.DataFrame:
    """Descarga datos recientes y calcula RSI."""
    agent = DataAgent(category=cfg.get("bybit_category", DEFAULT_BYBIT_CATEGORY))
    tf = timeframe or cfg.get("timeframe_base", "4h")
    days = cfg.get("loop", {}).get("history_days", 30)
    df = agent.get_ohlcv(symbol=symbol, timeframe=tf, days=days, force=True)
    df = df.sort_values("fecha").reset_index(drop=True)
    df["fecha"] = pd.to_datetime(df["fecha"])

    rsi_period = cfg.get("rsi_cross", {}).get("rsi_period", 14)
    df["rsi"] = RSIIndicator(close=df["cierre"], window=rsi_period).rsi()
    return df


def fetch_multi_tf_for_ml(
    symbol: str, exec_cfg: dict[str, Any], ml_cfg: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Descarga base + higher TFs como en el orquestador (merge_asof en features)."""
    agent = DataAgent(category=exec_cfg.get("bybit_category", DEFAULT_BYBIT_CATEGORY))
    days = int(exec_cfg.get("loop", {}).get("history_days", 30))
    dfs_by_tf: dict[str, pd.DataFrame] = {}
    for tf in _ml_all_timeframes(ml_cfg):
        d = agent.get_ohlcv(symbol=symbol, timeframe=tf, days=days, force=True)
        d = d.sort_values("fecha").reset_index(drop=True)
        d["fecha"] = pd.to_datetime(d["fecha"])
        dfs_by_tf[tf] = d
    base = ml_cfg.get("timeframe_base") or ml_cfg.get("timeframe", "4h")
    return dfs_by_tf[base], dfs_by_tf


def check_rsi_cross_signal(df: pd.DataFrame, cfg: dict) -> str | None:
    """
    Evalúa si la última vela cerrada genera señal RSI cross.

    Returns: "buy", "sell", o None
    """
    rsi_cfg = cfg.get("rsi_cross", {})
    entry_lvl = float(rsi_cfg.get("entry_cross_level", 32))
    exit_lvl = float(rsi_cfg.get("exit_cross_level", 50))
    touch_lb = int(rsi_cfg.get("entry_touch_lookback_bars", 0))

    if len(df) < 3:
        return None

    rsi = df["rsi"].to_numpy(float)
    last = len(rsi) - 1

    rsi_prev = rsi[last - 1]
    rsi_now = rsi[last]

    if np.isnan(rsi_prev) or np.isnan(rsi_now):
        return None

    cross_up = rsi_prev < entry_lvl and rsi_now >= entry_lvl
    if cross_up:
        if touch_lb > 0:
            window = rsi[max(0, last - touch_lb): last + 1]
            if np.nanmin(window) > entry_lvl:
                return None
        return "buy"

    cross_down = rsi_prev > exit_lvl and rsi_now <= exit_lvl
    if cross_down:
        return "sell"

    return None


def _merge_regime_thresholds(exec_xgb: dict[str, Any], ml_cfg: dict[str, Any], regime: str) -> dict[str, Any]:
    rb = (ml_cfg.get("regime_backtest") or {}).get(regime, {}) or {}
    ex = (exec_xgb.get(regime) or {}) if isinstance(exec_xgb.get(regime), dict) else {}
    out = {**rb, **ex}
    return out


def check_xgboost_signal(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    symbol: str,
    dfs_by_tf: dict[str, pd.DataFrame] | None = None,
) -> str | None:
    """
    Genera señal usando modelo XGBoost entrenado (una o tres cabezas por régimen).

    Returns: "buy", "sell", "open_short", o None
    """
    import joblib

    from src.agents.feature_agent import FeatureAgent

    xgb_cfg = cfg.get("xgboost", {})

    ml_config_path = ROOT / xgb_cfg.get("config_path", "configs/default.yaml")
    with open(ml_config_path, encoding="utf-8") as f:
        ml_cfg = yaml.safe_load(f)

    multi_regime = xgb_cfg.get("multi_regime")
    if multi_regime is None:
        multi_regime = bool(ml_cfg.get("multi_regime", False))

    feat_cfg = ml_cfg["features"]
    ht_cfg = ml_cfg.get("higher_timeframes", {})
    feature_agent = FeatureAgent(
        sma_corta=feat_cfg["sma_corta"],
        sma_larga=feat_cfg["sma_larga"],
        rsi_window=feat_cfg["rsi_window"],
        volatility_window=feat_cfg["volatility_window"],
        return_lags=feat_cfg["return_lags"],
        macd_cfg=feat_cfg.get("macd", {}),
        sr_cfg=feat_cfg.get("support_resistance", {}),
        higher_timeframes_cfg=ht_cfg,
    )

    use_higher = any(h.get("enabled") for h in ht_cfg.values()) if ht_cfg else False
    higher_dfs = dfs_by_tf if use_higher else None
    df_feat = feature_agent.build_features(df, higher_dfs=higher_dfs)
    feature_cols = feature_agent.feature_names

    if multi_regime:
        reg_cfg = ml_cfg.get("regimes", {})
        regime_agent = RegimeAgent(
            adx_trending_min=float(reg_cfg.get("adx_trending_min", 20)),
            trend_column=str(reg_cfg.get("trend_column", "weekly_trend")),
            adx_column=str(reg_cfg.get("adx_column", "weekly_adx")),
        )
        df_feat = regime_agent.assign_regime(df_feat)

    df_clean = df_feat.dropna(subset=feature_cols)

    if df_clean.empty:
        return None

    last_row = df_clean.iloc[[-1]][feature_cols]

    if not multi_regime:
        model_path = ROOT / xgb_cfg.get("model_path", "models/xgb_model.joblib")
        if not model_path.exists():
            print(f"[LiveRunner] Modelo no encontrado: {model_path}")
            print("  Ejecuta primero: python -m scripts.run_experiment")
            return None
        model = joblib.load(model_path)
        prob = float(model.predict_proba(last_row)[0, 1])
        buy_thresh = float(xgb_cfg.get("prob_buy_threshold", 0.55))
        sell_thresh = float(xgb_cfg.get("prob_sell_threshold", 0.45))
        print(f"  XGBoost prob(up)={prob:.4f} | buy>{buy_thresh} sell<{sell_thresh}")
        if prob > buy_thresh:
            return "buy"
        if prob < sell_thresh:
            return "sell"
        return None

    regime = str(df_clean["regime"].iloc[-1])

    rm_cfg = ml_cfg.get("regime_models", {})
    path_tpl = str(rm_cfg.get("path_template", "models/xgb_{regime}_{symbol}.joblib"))
    model_path = ROOT / path_tpl.format(regime=regime, symbol=symbol)
    if not model_path.exists():
        print(f"[LiveRunner] Modelo multi-régimen no encontrado: {model_path} (régimen={regime})")
        print("  Entrena con multi_regime: true en el YAML de ML y run_experiment.")
        return None

    model = joblib.load(model_path)
    prob = float(model.predict_proba(last_row)[0, 1])

    if regime == REGIME_BULL:
        th = _merge_regime_thresholds(xgb_cfg, ml_cfg, "bull")
        buy_t = float(th.get("prob_buy_threshold", xgb_cfg.get("prob_buy_threshold", 0.55)))
        sell_t = float(th.get("prob_sell_threshold", xgb_cfg.get("prob_sell_threshold", 0.45)))
        print(f"  XGB [{regime}] prob={prob:.4f} | buy>{buy_t} sell<{sell_t}")
        if prob > buy_t:
            return "buy"
        if prob < sell_t:
            return "sell"
        return None

    if regime == REGIME_BEAR:
        th = _merge_regime_thresholds(xgb_cfg, ml_cfg, "bear")
        o_t = float(th.get("prob_short_open_threshold", 0.55))
        c_t = float(th.get("prob_short_close_threshold", 0.45))
        print(f"  XGB [{regime}] prob(short-signal)={prob:.4f} | open>{o_t} close<{c_t}")
        if prob > o_t:
            return "open_short"
        if prob < c_t:
            return "sell"
        return None

    th = _merge_regime_thresholds(xgb_cfg, ml_cfg, "sideways")
    buy_t = float(th.get("prob_buy_threshold", 0.55))
    sell_t = float(th.get("prob_sell_threshold", 0.45))
    print(f"  XGB [{regime}] prob={prob:.4f} | buy>{buy_t} sell<{sell_t}")
    if prob > buy_t:
        return "buy"
    if prob < sell_t:
        return "sell"
    return None


def execute_signal(
    signal_type: str,
    symbol: str,
    agent: ExecutionAgent | PaperExecutionAgent,
    cfg: dict,
) -> None:
    """Ejecuta compra, venta, apertura de short o cierre de posición."""
    risk_cfg = cfg.get("risk", {})
    leverage = int(risk_cfg.get("leverage", 1))
    sl_pct = float(risk_cfg.get("stop_loss_pct", 0.03))
    tp_pct = float(risk_cfg.get("take_profit_pct", 0.10))
    max_positions = int(risk_cfg.get("max_open_positions", 1))
    allow_flip = bool(risk_cfg.get("allow_long_and_short_same_symbol", False))

    positions = agent.get_positions(symbol)
    ticker = agent.get_ticker(symbol)
    price = ticker["last"]

    if signal_type == "buy":
        if not allow_flip:
            for pos in list(agent.get_positions(symbol)):
                if pos.get("side") == "Sell":
                    result = agent.close_short(symbol=symbol, qty=pos["size"])
                    send_trade_email(result, _notify_email_recipients())
            positions = agent.get_positions(symbol)

        if len(positions) >= max_positions:
            print(f"  Ya hay {len(positions)} posición(es) abierta(s). Señal ignorada.")
            return

        capital_pct = float(risk_cfg.get("capital_pct_per_trade", 0))
        if capital_pct > 0:
            balance = agent.get_balance()
            capital = balance["available"] * capital_pct
        else:
            capital = float(risk_cfg.get("capital_per_trade_usdt", 100))

        qty = agent.calculate_qty(symbol, capital, leverage=leverage, price=price)

        sl_price = round(price * (1 - sl_pct), 2) if sl_pct > 0 else None
        tp_price = round(price * (1 + tp_pct), 2) if tp_pct > 0 else None

        if leverage > 1:
            agent.set_leverage(symbol, leverage)

        result = agent.open_long(
            symbol=symbol,
            qty=qty,
            stop_loss=sl_price,
            take_profit=tp_price,
        )
        send_trade_email(result, _notify_email_recipients())

    elif signal_type == "open_short":
        if not allow_flip:
            for pos in list(agent.get_positions(symbol)):
                if pos.get("side") == "Buy":
                    result = agent.close_long(symbol=symbol, qty=pos["size"])
                    send_trade_email(result, _notify_email_recipients())
            positions = agent.get_positions(symbol)

        if len(positions) >= max_positions:
            print(f"  Ya hay {len(positions)} posición(es) abierta(s). Señal open_short ignorada.")
            return

        capital_pct = float(risk_cfg.get("capital_pct_per_trade", 0))
        if capital_pct > 0:
            balance = agent.get_balance()
            capital = balance["available"] * capital_pct
        else:
            capital = float(risk_cfg.get("capital_per_trade_usdt", 100))

        qty = agent.calculate_qty(symbol, capital, leverage=leverage, price=price)

        sl_price = round(price * (1 + sl_pct), 2) if sl_pct > 0 else None
        tp_price = round(price * (1 - tp_pct), 2) if tp_pct > 0 else None

        if leverage > 1:
            agent.set_leverage(symbol, leverage)

        result = agent.open_short(
            symbol=symbol,
            qty=qty,
            stop_loss=sl_price,
            take_profit=tp_price,
        )
        send_trade_email(result, _notify_email_recipients())

    elif signal_type == "sell":
        positions = agent.get_positions(symbol)
        if not positions:
            print(f"  Sin posición abierta en {symbol}. Señal de venta ignorada.")
            return

        for pos in positions:
            if pos["side"] == "Buy":
                result = agent.close_long(symbol=symbol, qty=pos["size"])
            elif pos["side"] == "Sell":
                result = agent.close_short(symbol=symbol, qty=pos["size"])
            else:
                continue
            send_trade_email(result, _notify_email_recipients())


def run_loop(cfg: dict, agent: ExecutionAgent | PaperExecutionAgent) -> None:
    """Bucle principal de trading."""
    symbol = cfg["symbol"]
    strategy = cfg.get("strategy", "rsi_cross")
    loop_cfg = cfg.get("loop", {})
    interval = int(loop_cfg.get("check_interval_seconds", 60))
    exit_tf = cfg.get("exit_timeframe", "4h")

    risk_cfg = cfg.get("risk", {})
    max_daily_loss = float(risk_cfg.get("max_daily_loss_usdt", 0))

    initial_balance = agent.get_balance()["wallet_balance"]

    print(f"\n{'='*60}")
    print(f"  Live Trading — {symbol}")
    print(f"  Estrategia: {strategy}")
    print(f"  Intervalo: cada {interval}s")
    env = "TESTNET" if cfg.get("testnet", True) else "MAINNET"
    mode = "PAPER" if isinstance(agent, PaperExecutionAgent) else env
    print(f"  Modo: {mode}")
    print(f"  Balance inicial: {initial_balance:.2f} USDT")
    if max_daily_loss > 0:
        print(f"  Limite perdida diaria: {max_daily_loss:.2f} USDT")
    print(f"{'='*60}\n")

    agent.print_status(symbol)

    last_candle_check: str | None = None

    while _running:
        try:
            if max_daily_loss > 0:
                current_balance = agent.get_balance()["wallet_balance"]
                daily_loss = initial_balance - current_balance
                if daily_loss >= max_daily_loss:
                    print(f"\n[LiveRunner] LIMITE DE PERDIDA DIARIA ALCANZADO")
                    print(f"  Perdida: {daily_loss:.2f} USDT (limite: {max_daily_loss:.2f} USDT)")
                    print(f"  Bot detenido por seguridad. Revisa la estrategia antes de reiniciar.")
                    break

            if strategy == "xgboost":
                ml_cfg_path = ROOT / cfg.get("xgboost", {}).get("config_path", "configs/default.yaml")
                ml_cfg = load_config(ml_cfg_path)
                df, dfs_tf = fetch_multi_tf_for_ml(symbol, cfg, ml_cfg)
                rsi_period = cfg.get("rsi_cross", {}).get("rsi_period", 14)
                df = df.copy()
                df["rsi"] = RSIIndicator(close=df["cierre"], window=rsi_period).rsi()
            else:
                df = fetch_latest_data(symbol, cfg, timeframe=cfg.get("timeframe_base", "4h"))
                dfs_tf = None

            if df.empty:
                print("[LiveRunner] Sin datos. Reintentando...")
                time.sleep(interval)
                continue

            last_fecha = str(df.iloc[-1]["fecha"])

            if last_fecha == last_candle_check:
                time.sleep(interval)
                continue

            last_candle_check = last_fecha
            rsi_val = df.iloc[-1].get("rsi", float("nan"))
            price = df.iloc[-1]["cierre"]
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(f"[{ts}] {symbol} vela={last_fecha[:16]} cierre={price:.2f} RSI={rsi_val:.1f}")

            signal = None
            if strategy == "rsi_cross":
                signal = check_rsi_cross_signal(df, cfg)

                if signal is None and exit_tf.upper() in ("1D", "D"):
                    df_1d = fetch_latest_data(symbol, cfg, timeframe="1D")
                    if len(df_1d) >= 2:
                        rsi_1d = df_1d["rsi"].to_numpy(float)
                        exit_lvl = float(cfg.get("rsi_cross", {}).get("exit_cross_level", 50))
                        if rsi_1d[-2] > exit_lvl and rsi_1d[-1] <= exit_lvl:
                            signal = "sell"
                            print(f"  RSI 1D cruce ↓ {exit_lvl} → señal de salida")

            elif strategy == "xgboost":
                signal = check_xgboost_signal(df, cfg, symbol, dfs_by_tf=dfs_tf)
            else:
                print(f"[LiveRunner] Estrategia desconocida: {strategy}")

            if signal:
                print(f"  *** SEÑAL: {signal.upper()} ***")
                execute_signal(signal, symbol, agent, cfg)
            else:
                print("  Sin señal")

            try:
                log_signal(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "symbol": symbol,
                        "strategy": strategy,
                        "candle_time": last_fecha,
                        "close_price": float(price),
                        "rsi": float(rsi_val) if rsi_val == rsi_val else None,
                        "signal": signal,
                        "mode": "PAPER" if isinstance(agent, PaperExecutionAgent) else env,
                    }
                )
            except Exception:
                pass

            agent.print_status(symbol)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[LiveRunner] Error: {e}")
            time.sleep(interval)
            continue

        time.sleep(interval)

    print("\n[LiveRunner] Finalizando...")
    agent.print_status(symbol)

    log = agent.get_execution_log()
    if not log.empty:
        print(f"\n[LiveRunner] {len(log)} órdenes ejecutadas en esta sesión.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Live / Paper Trading con Bybit")
    ap.add_argument("--config", type=str, default="configs/execution.yaml")
    ap.add_argument("--paper", action="store_true", help="Modo paper trading (simulación local)")
    ap.add_argument("--symbol", type=str, default=None)
    ap.add_argument("--once", action="store_true", help="Ejecutar un solo ciclo (sin loop)")
    ap.add_argument("--status", action="store_true", help="Solo mostrar estado de cuenta")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")

    cfg = load_config(args.config)
    if args.symbol:
        cfg["symbol"] = args.symbol

    agent = create_execution_agent(cfg, paper=args.paper)
    symbol = cfg["symbol"]

    if args.status:
        agent.print_status(symbol)
        return

    if args.once:
        strategy = cfg.get("strategy", "rsi_cross")
        dfs_tf = None
        if strategy == "xgboost":
            ml_cfg_path = ROOT / cfg.get("xgboost", {}).get("config_path", "configs/default.yaml")
            ml_cfg = load_config(ml_cfg_path)
            df, dfs_tf = fetch_multi_tf_for_ml(symbol, cfg, ml_cfg)
            rsi_period = cfg.get("rsi_cross", {}).get("rsi_period", 14)
            df = df.copy()
            df["rsi"] = RSIIndicator(close=df["cierre"], window=rsi_period).rsi()
        else:
            df = fetch_latest_data(symbol, cfg)

        rsi_val = df.iloc[-1].get("rsi", float("nan"))
        price = df.iloc[-1]["cierre"]
        print(f"[once] {symbol} cierre={price:.2f} RSI={rsi_val:.1f}")

        signal = None
        if strategy == "rsi_cross":
            signal = check_rsi_cross_signal(df, cfg)
        elif strategy == "xgboost":
            signal = check_xgboost_signal(df, cfg, symbol, dfs_by_tf=dfs_tf)

        if signal:
            print(f"  *** SEÑAL: {signal.upper()} ***")
            execute_signal(signal, symbol, agent, cfg)
        else:
            print("  Sin señal")

        agent.print_status(symbol)
        return

    run_loop(cfg, agent)


if __name__ == "__main__":
    main()
