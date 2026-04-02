"""
Launcher: run 3 concurrent run_live processes (one per symbol).

Uso:
  python -m scripts.run_live_multi --config configs/execution.yaml --paper
  python -m scripts.run_live_multi --config configs/execution.yaml --paper --symbols BTCUSDT,ETHUSDT,ADAUSDT
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _python() -> str:
    return sys.executable or "python"


def _parse_symbols(raw: str) -> list[str]:
    out: list[str] = []
    for part in raw.split(","):
        s = part.strip().upper()
        if s:
            out.append(s)
    # dedupe preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


def main() -> None:
    ap = argparse.ArgumentParser(description="Run multiple paper/live bots (one per symbol)")
    ap.add_argument("--config", type=str, default="configs/execution.yaml")
    ap.add_argument("--paper", action="store_true", help="Modo paper trading (simulación local)")
    ap.add_argument(
        "--symbols",
        type=str,
        default="BTCUSDT,ETHUSDT,ADAUSDT",
        help="Lista separada por comas: BTCUSDT,ETHUSDT,ADAUSDT",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg_path = cfg_path.resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"No existe config: {cfg_path}")

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        raise ValueError("Debes indicar al menos 1 símbolo en --symbols")

    logs_dir = ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    procs: list[subprocess.Popen[str]] = []

    def stop_all() -> None:
        for p in procs:
            if p.poll() is None:
                try:
                    p.terminate()
                except OSError:
                    pass

    def _handle(_signum, _frame):
        print("\n[run_live_multi] Señal recibida. Parando bots…")
        stop_all()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    env = os.environ.copy()
    # Evita que cada hijo haga tee a logs/bot_web.log (si está habilitado).
    env["TRADEDAN_DISABLE_SHARED_LOG"] = "1"

    print(f"[run_live_multi] Config: {cfg_path}")
    print(f"[run_live_multi] Symbols: {', '.join(symbols)}")
    print(f"[run_live_multi] Mode: {'PAPER' if args.paper else 'LIVE'}")

    for sym in symbols:
        log_file = logs_dir / f"bot_{sym}.log"
        out = open(log_file, "a", encoding="utf-8")
        cmd = [
            _python(),
            "-m",
            "scripts.run_live",
            "--config",
            str(cfg_path),
            "--symbol",
            sym,
        ]
        if args.paper:
            cmd.append("--paper")

        print(f"[run_live_multi] Start {sym} → {log_file}")
        p = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        procs.append(p)

    # Espera: si cualquier bot termina, paramos todos
    try:
        while True:
            for p in procs:
                code = p.poll()
                if code is not None:
                    print(f"[run_live_multi] Un bot terminó (exit_code={code}). Parando el resto…")
                    stop_all()
                    return
            signal.pause()
    finally:
        stop_all()


if __name__ == "__main__":
    main()

