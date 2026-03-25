"""
BacktestAgent: simula una estrategia basada en las probabilidades de XGBoost.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.regime_agent import REGIME_BEAR, REGIME_BULL, REGIME_SIDEWAYS

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class BacktestAgent:
    """Backtest sencillo que convierte probabilidades en señales de trading."""

    def __init__(
        self,
        prob_buy_threshold: float = 0.55,
        prob_sell_threshold: float = 0.45,
        min_hold_bars: int = 0,
    ):
        self.prob_buy = prob_buy_threshold
        self.prob_sell = prob_sell_threshold
        self.min_hold_bars = min_hold_bars

    def run(
        self,
        df: pd.DataFrame,
        probas: np.ndarray,
        symbol: str = "UNKNOWN",
    ) -> dict:
        """
        Ejecuta el backtest.

        Args:
            df: DataFrame con al menos columnas 'fecha' y 'cierre' (alineado con probas).
            probas: array de probabilidad de subida (prob_class_1) para cada fila.
            symbol: símbolo del activo (para nombrar el CSV de trades).

        Returns:
            dict con métricas, DataFrame de simulación y lista de trades.
        """
        sim = df[["fecha", "cierre"]].copy()
        sim["prob_up"] = probas

        # Señales con min_hold_bars: no evaluar salida hasta cumplir el mínimo
        signals = np.zeros(len(sim), dtype=int)
        in_position = False
        bars_held = 0

        for i in range(len(sim)):
            prob = sim.iloc[i]["prob_up"]
            if not in_position:
                if prob > self.prob_buy:
                    in_position = True
                    bars_held = 0
                    signals[i] = 1
                else:
                    signals[i] = 0
            else:
                bars_held += 1
                if bars_held >= self.min_hold_bars and prob < self.prob_sell:
                    in_position = False
                    signals[i] = 0
                else:
                    signals[i] = 1

        sim["senal"] = signals

        # Marcar entradas y salidas
        sim["entrada"] = (sim["senal"] == 1) & (sim["senal"].shift(1, fill_value=0) == 0)
        sim["salida"] = (sim["senal"] == 0) & (sim["senal"].shift(1, fill_value=0) == 1)

        sim["retorno"] = sim["cierre"].pct_change()
        sim["retorno_estrategia"] = sim["senal"].shift(1) * sim["retorno"]
        sim = sim.dropna(subset=["retorno_estrategia"])

        # Generar log de operaciones (pares entrada → salida)
        trades = self._build_trades_log(sim)

        # Métricas
        retorno_total = (1 + sim["retorno_estrategia"]).prod() - 1
        retorno_buyhold = (1 + sim["retorno"]).prod() - 1

        # Drawdown
        cum = (1 + sim["retorno_estrategia"]).cumprod()
        peak = cum.cummax()
        drawdown = ((cum - peak) / peak).min()

        n_trades = len(trades)

        if n_trades > 0:
            win_rate = sum(1 for t in trades if t["retorno"] > 0) / n_trades
        else:
            win_rate = 0.0

        # Sharpe sencillo (anualizado asumiendo ~252 barras/año para daily, orientativo)
        mean_r = sim["retorno_estrategia"].mean()
        std_r = sim["retorno_estrategia"].std()
        sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0

        metrics = {
            "retorno_total": retorno_total,
            "retorno_buyhold": retorno_buyhold,
            "max_drawdown": drawdown,
            "n_trades": n_trades,
            "win_rate": win_rate,
            "sharpe": sharpe,
        }

        print("[BacktestAgent] Resultados:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        self._print_trades(trades)
        self._save_trades_csv(trades, symbol)

        return {"metrics": metrics, "sim": sim, "trades": trades}

    def run_regime(
        self,
        df: pd.DataFrame,
        probas: np.ndarray,
        *,
        regime_col: str = "regime",
        thresholds: dict | None = None,
        min_hold_bars: int = 0,
        symbol: str = "UNKNOWN",
    ) -> dict:
        """
        Backtest long / short / flat según régimen y probabilidades alineadas.

        *thresholds* espera claves bull/bear/sideways con umbrales como en
        configs/default.yaml → regime_backtest.
        """
        th = thresholds or {}
        tb = th.get("bull", {})
        bear = th.get("bear", {})
        ts = th.get("sideways", {})
        prob_buy_bull = float(tb.get("prob_buy_threshold", self.prob_buy))
        prob_sell_bull = float(tb.get("prob_sell_threshold", self.prob_sell))
        p_open_s = float(bear.get("prob_short_open_threshold", 0.55))
        p_close_s = float(bear.get("prob_short_close_threshold", 0.45))
        prob_buy_sw = float(ts.get("prob_buy_threshold", self.prob_buy))
        prob_sell_sw = float(ts.get("prob_sell_threshold", self.prob_sell))

        sim = df[["fecha", "cierre", regime_col]].copy()
        sim.rename(columns={regime_col: "regime"}, inplace=True)
        sim["prob"] = probas

        n = len(sim)
        position = np.zeros(n, dtype=np.int8)
        bars_in_pos = 0
        prev_regime: str | None = None
        pos = 0

        for i in range(n):
            reg = sim.iloc[i]["regime"]
            prob = float(sim.iloc[i]["prob"])

            if prev_regime is not None and reg != prev_regime:
                if pos == 1 and reg != REGIME_BULL:
                    pos = 0
                    bars_in_pos = 0
                if pos == -1 and reg != REGIME_BEAR:
                    pos = 0
                    bars_in_pos = 0
            prev_regime = str(reg)

            if reg == REGIME_BULL:
                if pos <= 0:
                    if prob > prob_buy_bull:
                        pos = 1
                        bars_in_pos = 0
                else:
                    bars_in_pos += 1
                    if bars_in_pos >= min_hold_bars and prob < prob_sell_bull:
                        pos = 0
                        bars_in_pos = 0

            elif reg == REGIME_BEAR:
                if pos >= 0:
                    if prob > p_open_s:
                        pos = -1
                        bars_in_pos = 0
                else:
                    bars_in_pos += 1
                    if bars_in_pos >= min_hold_bars and prob < p_close_s:
                        pos = 0
                        bars_in_pos = 0

            elif reg == REGIME_SIDEWAYS:
                if pos <= 0:
                    if prob > prob_buy_sw:
                        pos = 1
                        bars_in_pos = 0
                else:
                    bars_in_pos += 1
                    if bars_in_pos >= min_hold_bars and prob < prob_sell_sw:
                        pos = 0
                        bars_in_pos = 0

            position[i] = pos

        sim["position"] = position
        sim["retorno"] = sim["cierre"].pct_change()
        sim["retorno_estrategia"] = sim["position"].shift(1).fillna(0) * sim["retorno"]
        sim = sim.dropna(subset=["retorno_estrategia"])

        trades = self._build_trades_log_from_position(sim)
        retorno_total = (1 + sim["retorno_estrategia"]).prod() - 1
        retorno_buyhold = (1 + sim["retorno"]).prod() - 1
        cum = (1 + sim["retorno_estrategia"]).cumprod()
        peak = cum.cummax()
        drawdown = ((cum - peak) / peak).min()
        n_trades = len(trades)
        win_rate = (
            sum(1 for t in trades if t["retorno"] > 0) / n_trades if n_trades else 0.0
        )
        mean_r = sim["retorno_estrategia"].mean()
        std_r = sim["retorno_estrategia"].std()
        sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0

        metrics = {
            "retorno_total": retorno_total,
            "retorno_buyhold": retorno_buyhold,
            "max_drawdown": drawdown,
            "n_trades": n_trades,
            "win_rate": win_rate,
            "sharpe": sharpe,
        }

        metrics_by_regime: dict[str, dict[str, float]] = {}
        for reg in (REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS):
            mask = sim["regime"] == reg
            if not mask.any():
                metrics_by_regime[reg] = {"bars": 0.0, "retorno_acum": 0.0}
                continue
            sub = sim.loc[mask, "retorno_estrategia"]
            metrics_by_regime[reg] = {
                "bars": float(mask.sum()),
                "retorno_acum": float((1 + sub).prod() - 1),
            }

        print("[BacktestAgent] Resultados (multi-régimen):")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        print("  por régimen (retorno acum. en barras del test):")
        for reg, m in metrics_by_regime.items():
            print(f"    {reg}: bars={int(m['bars'])}, ret_acum={m['retorno_acum']:.4f}")

        self._print_trades(trades)
        self._save_trades_csv(trades, f"{symbol}_regime")

        return {
            "metrics": metrics,
            "metrics_by_regime": metrics_by_regime,
            "sim": sim,
            "trades": trades,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _build_trades_log_from_position(sim: pd.DataFrame) -> list[dict]:
        """Trades desde serie position: 1 long, -1 short, 0 flat."""
        trades: list[dict] = []
        entry_date = None
        entry_price = None
        side: str | None = None
        prev_pos = 0

        for _, row in sim.iterrows():
            pos = int(row["position"])
            if prev_pos == 0 and pos == 1:
                entry_date = row["fecha"]
                entry_price = row["cierre"]
                side = "long"
            elif prev_pos == 0 and pos == -1:
                entry_date = row["fecha"]
                entry_price = row["cierre"]
                side = "short"
            elif prev_pos == 1 and pos == 0 and entry_date is not None:
                exit_price = row["cierre"]
                ret = exit_price / entry_price - 1
                trades.append({
                    "entrada_fecha": entry_date,
                    "entrada_precio": entry_price,
                    "salida_fecha": row["fecha"],
                    "salida_precio": exit_price,
                    "retorno": ret,
                    "lado": "long",
                })
                entry_date = None
                entry_price = None
                side = None
            elif prev_pos == -1 and pos == 0 and entry_date is not None:
                exit_price = row["cierre"]
                ret = (entry_price - exit_price) / entry_price
                trades.append({
                    "entrada_fecha": entry_date,
                    "entrada_precio": entry_price,
                    "salida_fecha": row["fecha"],
                    "salida_precio": exit_price,
                    "retorno": ret,
                    "lado": "short",
                })
                entry_date = None
                entry_price = None
                side = None
            prev_pos = pos

        if entry_date is not None and entry_price is not None and side:
            last = sim.iloc[-1]
            exit_price = last["cierre"]
            if side == "long":
                ret = exit_price / entry_price - 1
            else:
                ret = (entry_price - exit_price) / entry_price
            trades.append({
                "entrada_fecha": entry_date,
                "entrada_precio": entry_price,
                "salida_fecha": last["fecha"],
                "salida_precio": exit_price,
                "retorno": ret,
                "lado": side,
            })

        return trades

    # ------------------------------------------------------------------
    @staticmethod
    def _build_trades_log(sim: pd.DataFrame) -> list[dict]:
        """Construye una lista de trades (pares entrada/salida) a partir de la simulación."""
        trades: list[dict] = []
        entry_date = None
        entry_price = None

        for _, row in sim.iterrows():
            if row["entrada"]:
                entry_date = row["fecha"]
                entry_price = row["cierre"]
            elif row["salida"] and entry_date is not None:
                exit_price = row["cierre"]
                ret = exit_price / entry_price - 1
                trades.append({
                    "entrada_fecha": entry_date,
                    "entrada_precio": entry_price,
                    "salida_fecha": row["fecha"],
                    "salida_precio": exit_price,
                    "retorno": ret,
                })
                entry_date = None
                entry_price = None

        # Si hay posición abierta al final del periodo
        if entry_date is not None:
            last = sim.iloc[-1]
            ret = last["cierre"] / entry_price - 1
            trades.append({
                "entrada_fecha": entry_date,
                "entrada_precio": entry_price,
                "salida_fecha": last["fecha"],
                "salida_precio": last["cierre"],
                "retorno": ret,
            })

        return trades

    @staticmethod
    def _print_trades(trades: list[dict]) -> None:
        """Imprime la tabla de entradas y salidas."""
        if not trades:
            print("\n[BacktestAgent] Sin operaciones en este periodo.")
            return

        print(f"\n[BacktestAgent] Log de operaciones ({len(trades)} trades):")
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

    @staticmethod
    def _save_trades_csv(trades: list[dict], symbol: str) -> None:
        """Guarda el log de operaciones como CSV."""
        results_dir = _PROJECT_ROOT / "data" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = results_dir / f"{symbol}_trades.csv"

        if not trades:
            print(f"[BacktestAgent] No hay trades que guardar para {symbol}.")
            return

        df_trades = pd.DataFrame(trades)
        df_trades.index = range(1, len(df_trades) + 1)
        df_trades.index.name = "#"
        df_trades.to_csv(csv_path)
        print(f"[BacktestAgent] Trades guardados en {csv_path}")
