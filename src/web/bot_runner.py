from __future__ import annotations

import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil


@dataclass
class BotStatus:
    symbol: str
    running: bool
    pid: int | None
    started_at: str | None
    command: list[str] | None
    config: str | None
    paper: bool | None
    last_log_lines: list[str]


class MultiBotRunner:
    """
    Minimal process manager for multiple scripts.run_live processes.

    Runs one subprocess per symbol, logs to logs/bot_<SYMBOL>.log, and provides
    start/stop/status for a local dashboard.
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self._procs: dict[str, subprocess.Popen[str]] = {}
        self._started_at: dict[str, datetime] = {}
        self._cmd: dict[str, list[str]] = {}
        self._config: dict[str, str] = {}
        self._paper: dict[str, bool] = {}

        self.log_dir = self.project_root / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _python(self) -> str:
        return sys.executable or "python"

    def _log_file(self, symbol: str) -> Path:
        return self.log_dir / f"bot_{symbol.upper()}.log"

    def _is_running(self, symbol: str) -> bool:
        proc = self._procs.get(symbol)
        if proc is None or proc.pid is None:
            return False
        try:
            p = psutil.Process(proc.pid)
            return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
        except psutil.Error:
            return False

    def _tail_log(self, symbol: str, n: int = 20) -> list[str]:
        try:
            lf = self._log_file(symbol)
            if not lf.exists():
                return []
            with open(lf, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            return [ln.rstrip("\n") for ln in lines[-n:]]
        except OSError:
            return []

    def start_one(
        self,
        symbol: str,
        *,
        config: str = "configs/execution.yaml",
        paper: bool = True,
    ) -> BotStatus:
        symbol = symbol.upper()
        if self._is_running(symbol):
            return self.status_one(symbol)

        cfg_path = str((self.project_root / config).resolve())
        args = [
            self._python(),
            "-m",
            "scripts.run_live",
            "--config",
            cfg_path,
            "--symbol",
            symbol,
        ]
        if paper:
            args.append("--paper")

        out = open(self._log_file(symbol), "w", encoding="utf-8")
        env = os.environ.copy()
        env["TRADEDAN_DISABLE_SHARED_LOG"] = "1"
        proc = subprocess.Popen(
            args,
            cwd=str(self.project_root),
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        self._procs[symbol] = proc
        self._started_at[symbol] = datetime.now(timezone.utc)
        self._cmd[symbol] = args
        self._config[symbol] = config
        self._paper[symbol] = paper
        return self.status_one(symbol)

    def stop_one(self, symbol: str) -> BotStatus:
        symbol = symbol.upper()
        if not self._is_running(symbol):
            return self.status_one(symbol)

        proc = self._procs.get(symbol)
        if proc is None or proc.pid is None:
            return self.status_one(symbol)
        try:
            os.kill(proc.pid, signal.SIGTERM)
        except OSError:
            pass
        return self.status_one(symbol)

    def stop_all(self) -> list[BotStatus]:
        out: list[BotStatus] = []
        for symbol in list(self._procs.keys()):
            out.append(self.stop_one(symbol))
        return out

    def status_one(self, symbol: str) -> BotStatus:
        symbol = symbol.upper()
        running = self._is_running(symbol)
        proc = self._procs.get(symbol)
        pid = proc.pid if (proc and proc.pid) else None
        started_at = self._started_at.get(symbol).isoformat() if self._started_at.get(symbol) else None
        return BotStatus(
            symbol=symbol,
            running=running,
            pid=pid,
            started_at=started_at,
            command=self._cmd.get(symbol),
            config=self._config.get(symbol),
            paper=self._paper.get(symbol),
            last_log_lines=self._tail_log(symbol),
        )

    def statuses(self) -> list[dict[str, Any]]:
        # include even if stopped (known symbols) to keep UI stable
        symbols = sorted(set(self._procs.keys()) | set(self._config.keys()) | set(self._started_at.keys()))
        return [self.status_one(s).__dict__ for s in symbols]

    def start_many(
        self,
        symbols: list[str],
        *,
        config: str = "configs/execution.yaml",
        paper: bool = True,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for s in symbols:
            out.append(self.start_one(s, config=config, paper=paper).__dict__)
        return out


