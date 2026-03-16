"""
BacktestAgent: simula una estrategia basada en las probabilidades de XGBoost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class BacktestAgent:
    """Backtest sencillo que convierte probabilidades en señales de trading."""

    def __init__(
        self,
        prob_buy_threshold: float = 0.55,
        prob_sell_threshold: float = 0.45,
    ):
        self.prob_buy = prob_buy_threshold
        self.prob_sell = prob_sell_threshold

    def run(
        self,
        df: pd.DataFrame,
        probas: np.ndarray,
    ) -> dict:
        """
        Ejecuta el backtest.

        Args:
            df: DataFrame con al menos columnas 'fecha' y 'cierre' (alineado con probas).
            probas: array de probabilidad de subida (prob_class_1) para cada fila.

        Returns:
            dict con métricas y un DataFrame con la simulación.
        """
        sim = df[["fecha", "cierre"]].copy()
        sim["prob_up"] = probas

        # Señal: 1 = largo, 0 = fuera
        sim["senal"] = 0
        sim.loc[sim["prob_up"] > self.prob_buy, "senal"] = 1
        sim.loc[sim["prob_up"] < self.prob_sell, "senal"] = 0

        sim["retorno"] = sim["cierre"].pct_change()
        sim["retorno_estrategia"] = sim["senal"].shift(1) * sim["retorno"]
        sim = sim.dropna(subset=["retorno_estrategia"])

        # Métricas
        retorno_total = (1 + sim["retorno_estrategia"]).prod() - 1
        retorno_buyhold = (1 + sim["retorno"]).prod() - 1

        # Drawdown
        cum = (1 + sim["retorno_estrategia"]).cumprod()
        peak = cum.cummax()
        drawdown = ((cum - peak) / peak).min()

        n_trades = (sim["senal"].diff().abs() > 0).sum()

        # Win rate sobre días en posición
        dias_en_posicion = sim[sim["senal"].shift(1) == 1]
        if len(dias_en_posicion) > 0:
            win_rate = (dias_en_posicion["retorno_estrategia"] > 0).mean()
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
            "n_trades": int(n_trades),
            "win_rate": win_rate,
            "sharpe": sharpe,
        }

        print("[BacktestAgent] Resultados:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        return {"metrics": metrics, "sim": sim}
