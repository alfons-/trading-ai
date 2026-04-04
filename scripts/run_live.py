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

import httpx
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
from src.notifications.pushover import send_pushover_async

_running = True

# Debounce identical Pushover error messages (avoid spam on tight loops).
_pushover_error_last_mon: float = 0.0
_pushover_error_last_msg: str = ""


def _pushover_error_debounced(msg: str, *, debounce_s: float = 30.0) -> bool:
    global _pushover_error_last_mon, _pushover_error_last_msg
    now = time.monotonic()
    if msg == _pushover_error_last_msg and (now - _pushover_error_last_mon) < debounce_s:
        return False
    _pushover_error_last_mon = now
    _pushover_error_last_msg = msg
    return True

_SIGNALS_DIR = ROOT / "data" / "signal_logs"
_SIGNALS_FILE = _SIGNALS_DIR / "signals.jsonl"
_SHARED_LOG_FILE = ROOT / "logs" / "bot_web.log"

_TF_CACHE: dict[tuple[str, str, str], pd.DataFrame] = {}
_TF_LAST_TS: dict[tuple[str, str, str], pd.Timestamp] = {}


def _merge_ohlcv(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        return new.sort_values("fecha").reset_index(drop=True)
    df = pd.concat([old, new], ignore_index=True)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.drop_duplicates(subset="fecha").sort_values("fecha").reset_index(drop=True)
    return df


def _safe_is_rate_limit_error(e: Exception) -> bool:
    msg = str(e)
    return "retCode" in msg and "10006" in msg


def _safe_is_transient_network_error(e: BaseException) -> bool:
    """SSL handshake timeout, cortes de red, etc. (reintentar con backoff, sin Pushover)."""
    transient_types: tuple[type[BaseException], ...] = (
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.PoolTimeout,
        httpx.ConnectError,
    )
    seen: set[int] = set()
    cur: BaseException | None = e
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, transient_types):
            return True
        msg = str(cur).lower()
        if "handshake" in msg and "timeout" in msg:
            return True
        if "_ssl.c" in str(cur) and "timed out" in msg:
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _fetch_ohlcv_smart(
    agent: DataAgent,
    *,
    symbol: str,
    timeframe: str,
    history_days: int,
    refresh_days: int,
    force_full_init: bool = False,
) -> tuple[pd.DataFrame, bool]:
    """
    Devuelve (df, updated).

    - Inicializa cache con histórico completo (desde CSV si existe; si no, descarga 1 vez).
    - En refrescos, solo descarga pocos días para “append” si hay velas nuevas.
    """
    key = (agent.category, symbol, timeframe)

    # 1) Init: intenta cargar CSV (sin pegar a API)
    if key not in _TF_CACHE or force_full_init:
        try:
            base = agent.get_ohlcv(symbol=symbol, timeframe=timeframe, days=history_days, force=False)
        except Exception:
            # Si no hay CSV o está corrupto, descarga una vez el histórico completo.
            base = agent.get_ohlcv(symbol=symbol, timeframe=timeframe, days=history_days, force=True)
        _TF_CACHE[key] = base
        _TF_LAST_TS[key] = pd.to_datetime(base["fecha"].iloc[-1]) if not base.empty else pd.Timestamp.min
        return base, True

    # 2) Refresh: descarga solo pocos días para evitar rate limit
    old = _TF_CACHE[key]
    old_last = _TF_LAST_TS.get(key, pd.Timestamp.min)
    try:
        recent = agent.get_ohlcv(symbol=symbol, timeframe=timeframe, days=refresh_days, force=True)
    except Exception as e:
        # Propaga: el caller aplicará backoff en rate limit
        raise

    if recent.empty:
        return old, False
    new_last = pd.to_datetime(recent["fecha"].iloc[-1])
    if new_last <= old_last:
        return old, False

    merged = _merge_ohlcv(old, recent)
    _TF_CACHE[key] = merged
    _TF_LAST_TS[key] = pd.to_datetime(merged["fecha"].iloc[-1])
    return merged, True


class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _setup_shared_log_capture() -> None:
    if os.getenv("TRADEDAN_DISABLE_SHARED_LOG") == "1":
        return
    _SHARED_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    shared = open(_SHARED_LOG_FILE, "a", encoding="utf-8")
    sys.stdout = _TeeStream(sys.stdout, shared)
    sys.stderr = _TeeStream(sys.stderr, shared)


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
    else:
        # Compatibilidad con rutas absolutas antiguas del repo renombrado
        # p.ej. /Users/.../trading-ai/configs/foo.yaml -> ROOT/configs/foo.yaml
        if not p.exists():
            try:
                rel = p.relative_to(Path("/Users/alfonsomartinezdomenech/trading-ai"))
            except ValueError:
                rel = None
            if rel is not None:
                legacy_candidate = ROOT / rel
                if legacy_candidate.exists():
                    p = legacy_candidate
    p = p.resolve()
    if not p.is_file():
        raise FileNotFoundError(
            f"No existe el archivo de configuración (ruta resuelta): {p}"
        )
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

    # Refresco incremental para no exceder rate limits:
    # - base TF: refresca pocos días cada ciclo (si hay vela nueva se actualiza)
    # - 1D/1W: refrescan muy poco y solo se usan como contexto (merge_asof)
    refresh_days_by_tf = {
        "4h": 7,
        "1h": 3,
        "1D": 30,
        "1W": 180,
    }

    dfs_by_tf: dict[str, pd.DataFrame] = {}
    updated_any = False
    for tf in _ml_all_timeframes(ml_cfg):
        refresh_days = int(refresh_days_by_tf.get(tf, 7))
        d, updated = _fetch_ohlcv_smart(
            agent,
            symbol=symbol,
            timeframe=tf,
            history_days=days,
            refresh_days=refresh_days,
        )
        updated_any = updated_any or updated
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
) -> tuple[str | None, str | None]:
    """
    Genera señal usando modelo XGBoost entrenado (una o tres cabezas por régimen).

    Returns: (signal, regime) donde signal es "buy", "sell", "open_short", o None;
    regime es bull/bear/sideways en multi_régimen, o None si no aplica.
    """
    import joblib

    from src.agents.feature_agent import FeatureAgent

    xgb_cfg = cfg.get("xgboost", {})

    ml_config_path = ROOT / xgb_cfg.get("config_path", "configs/default.yaml")
    ml_cfg = load_config(ml_config_path)

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
        regime_out = None
        if multi_regime and "regime" in df_feat.columns:
            s = df_feat["regime"].dropna()
            if not s.empty:
                regime_out = str(s.iloc[-1])
        return None, regime_out

    last_row = df_clean.iloc[[-1]][feature_cols]

    if not multi_regime:
        model_path = ROOT / xgb_cfg.get("model_path", "models/xgb_model.joblib")
        if not model_path.exists():
            print(f"[LiveRunner] Modelo no encontrado: {model_path}")
            print("  Ejecuta primero: python -m scripts.run_experiment")
            return None, None
        model = joblib.load(model_path)
        prob = float(model.predict_proba(last_row)[0, 1])
        buy_thresh = float(xgb_cfg.get("prob_buy_threshold", 0.55))
        sell_thresh = float(xgb_cfg.get("prob_sell_threshold", 0.45))
        print(f"  XGBoost prob(up)={prob:.4f} | buy>{buy_thresh} sell<{sell_thresh}")
        if prob > buy_thresh:
            return "buy", None
        if prob < sell_thresh:
            return "sell", None
        return None, None

    regime = str(df_clean["regime"].iloc[-1])

    rm_cfg = ml_cfg.get("regime_models", {})
    path_tpl = str(rm_cfg.get("path_template", "models/xgb_{regime}_{symbol}.joblib"))
    model_path = ROOT / path_tpl.format(regime=regime, symbol=symbol)
    if not model_path.exists():
        print(f"[LiveRunner] Modelo multi-régimen no encontrado: {model_path} (régimen={regime})")
        print("  Entrena con multi_regime: true en el YAML de ML y run_experiment.")
        return None, regime

    model = joblib.load(model_path)
    prob = float(model.predict_proba(last_row)[0, 1])

    if regime == REGIME_BULL:
        th = _merge_regime_thresholds(xgb_cfg, ml_cfg, "bull")
        buy_t = float(th.get("prob_buy_threshold", xgb_cfg.get("prob_buy_threshold", 0.55)))
        sell_t = float(th.get("prob_sell_threshold", xgb_cfg.get("prob_sell_threshold", 0.45)))
        print(f"  XGB [{regime}] prob={prob:.4f} | buy>{buy_t} sell<{sell_t}")
        if prob > buy_t:
            return "buy", regime
        if prob < sell_t:
            return "sell", regime
        return None, regime

    if regime == REGIME_BEAR:
        th = _merge_regime_thresholds(xgb_cfg, ml_cfg, "bear")
        o_t = float(th.get("prob_short_open_threshold", 0.55))
        c_t = float(th.get("prob_short_close_threshold", 0.45))
        print(f"  XGB [{regime}] prob(short-signal)={prob:.4f} | open>{o_t} close<{c_t}")
        if prob > o_t:
            return "open_short", regime
        if prob < c_t:
            return "sell", regime
        return None, regime

    th = _merge_regime_thresholds(xgb_cfg, ml_cfg, "sideways")
    buy_t = float(th.get("prob_buy_threshold", 0.55))
    sell_t = float(th.get("prob_sell_threshold", 0.45))
    print(f"  XGB [{regime}] prob={prob:.4f} | buy>{buy_t} sell<{sell_t}")
    if prob > buy_t:
        return "buy", regime
    if prob < sell_t:
        return "sell", regime
    return None, regime


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
    last_known_candle: str | None = None
    rate_limit_backoff_s = 5

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
            last_known_candle = last_fecha
            rate_limit_backoff_s = 5  # reset backoff tras un ciclo “útil”
            rsi_val = df.iloc[-1].get("rsi", float("nan"))
            price = df.iloc[-1]["cierre"]
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(f"[{ts}] {symbol} vela={last_fecha[:16]} cierre={price:.2f} RSI={rsi_val:.1f}")

            signal = None
            regime = None
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
                signal, regime = check_xgboost_signal(df, cfg, symbol, dfs_by_tf=dfs_tf)
            else:
                print(f"[LiveRunner] Estrategia desconocida: {strategy}")

            if signal:
                print(f"  *** SEÑAL: {signal.upper()} ***")
                reg_line = f"regime={regime}\n" if regime else ""
                send_pushover_async(
                    (
                        f"symbol={symbol}\n"
                        f"candle={last_fecha}\n"
                        f"price={float(price):.6g}\n"
                        f"{reg_line}"
                        f"signal={signal}\n"
                        f"strategy={strategy}\n"
                        f"mode={mode}"
                    ),
                    title="Señal detectada",
                )
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
                        "regime": regime,
                        "mode": "PAPER" if isinstance(agent, PaperExecutionAgent) else env,
                    }
                )
            except Exception:
                pass

            agent.print_status(symbol)

        except KeyboardInterrupt:
            break
        except Exception as e:
            if _safe_is_rate_limit_error(e):
                print(f"[LiveRunner] Rate limit Bybit detectado. Backoff {rate_limit_backoff_s}s...")
                time.sleep(rate_limit_backoff_s)
                rate_limit_backoff_s = min(rate_limit_backoff_s * 2, 120)
                continue
            if _safe_is_transient_network_error(e):
                print(
                    f"[LiveRunner] Red/API temporal ({e!s}). "
                    f"Backoff {rate_limit_backoff_s}s..."
                )
                time.sleep(rate_limit_backoff_s)
                rate_limit_backoff_s = min(rate_limit_backoff_s * 2, 120)
                continue
            extra = ""
            fn = getattr(e, "filename", None)
            if fn is not None:
                extra = f" | archivo: {fn!r}"
            print(f"[LiveRunner] Error: {e!s}{extra}")
            err_body = (
                f"{e!s}{extra}\n"
                f"symbol={symbol}\n"
                f"last_candle={last_known_candle or 'n/a'}"
            )
            if _pushover_error_debounced(err_body):
                send_pushover_async(err_body, title="LiveRunner error")
            time.sleep(interval)
            continue

        time.sleep(interval)

    print("\n[LiveRunner] Finalizando...")
    agent.print_status(symbol)

    log = agent.get_execution_log()
    if not log.empty:
        print(f"\n[LiveRunner] {len(log)} órdenes ejecutadas en esta sesión.")


def main() -> None:
    _setup_shared_log_capture()
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
        regime = None
        if strategy == "rsi_cross":
            signal = check_rsi_cross_signal(df, cfg)
        elif strategy == "xgboost":
            signal, regime = check_xgboost_signal(df, cfg, symbol, dfs_by_tf=dfs_tf)

        if signal:
            print(f"  *** SEÑAL: {signal.upper()} ***")
            last_fecha_once = str(df.iloc[-1]["fecha"])
            env_once = "TESTNET" if cfg.get("testnet", True) else "MAINNET"
            mode_once = "PAPER" if isinstance(agent, PaperExecutionAgent) else env_once
            reg_line = f"regime={regime}\n" if regime else ""
            send_pushover_async(
                (
                    f"symbol={symbol}\n"
                    f"candle={last_fecha_once}\n"
                    f"price={float(price):.6g}\n"
                    f"{reg_line}"
                    f"signal={signal}\n"
                    f"strategy={strategy}\n"
                    f"mode={mode_once}"
                ),
                title="Señal detectada",
            )
            execute_signal(signal, symbol, agent, cfg)
        else:
            print("  Sin señal")

        agent.print_status(symbol)
        return

    run_loop(cfg, agent)


if __name__ == "__main__":
    main()
