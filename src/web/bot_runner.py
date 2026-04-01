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
    running: bool
    pid: int | None
    started_at: str | None
    command: list[str] | None
    config: str | None
    paper: bool | None
    last_log_lines: list[str]


class BotRunner:
    """
    Minimal process manager for scripts.run_live.

    Runs a single bot subprocess, logs to logs/bot_web.log, and provides
    start/stop/status for a local dashboard.
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self._proc: subprocess.Popen[str] | None = None
        self._started_at: datetime | None = None
        self._cmd: list[str] | None = None
        self._config: str | None = None
        self._paper: bool | None = None

        self.log_dir = self.project_root / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "bot_web.log"

    def _python(self) -> str:
        return sys.executable or "python"

    def _is_running(self) -> bool:
        if self._proc is None or self._proc.pid is None:
            return False
        try:
            p = psutil.Process(self._proc.pid)
            return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
        except psutil.Error:
            return False

    def _tail_log(self, n: int = 20) -> list[str]:
        try:
            if not self.log_file.exists():
                return []
            with open(self.log_file, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            return [ln.rstrip("\n") for ln in lines[-n:]]
        except OSError:
            return []

    def start(self, *, config: str = "configs/execution.yaml", paper: bool = True) -> BotStatus:
        if self._is_running():
            return self.status()

        cfg_path = str((self.project_root / config).resolve())
        args = [self._python(), "-m", "scripts.run_live", "--config", cfg_path]
        if paper:
            args.append("--paper")

        out = open(self.log_file, "w", encoding="utf-8")
        self._proc = subprocess.Popen(
            args,
            cwd=str(self.project_root),
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
        self._started_at = datetime.now(timezone.utc)
        self._cmd = args
        self._config = config
        self._paper = paper
        return self.status()

    def stop(self) -> BotStatus:
        if not self._is_running():
            return self.status()

        assert self._proc is not None and self._proc.pid is not None
        try:
            os.kill(self._proc.pid, signal.SIGTERM)
        except OSError:
            pass
        return self.status()

    def status(self) -> BotStatus:
        running = self._is_running()
        pid = self._proc.pid if (self._proc and self._proc.pid) else None
        started_at = self._started_at.isoformat() if self._started_at else None
        return BotStatus(
            running=running,
            pid=pid,
            started_at=started_at,
            command=self._cmd,
            config=self._config,
            paper=self._paper,
            last_log_lines=self._tail_log(),
        )

    def status_dict(self) -> dict[str, Any]:
        s = self.status()
        return {
            "running": s.running,
            "pid": s.pid,
            "started_at": s.started_at,
            "command": s.command,
            "config": s.config,
            "paper": s.paper,
            "last_log_lines": s.last_log_lines,
        }

