"""
BacktestAgent: simula una estrategia basada en las probabilidades de XGBoost.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

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
