"""
LabelAgent: construye la variable objetivo (target) para XGBoost.
"""

from __future__ import annotations

import pandas as pd


class LabelAgent:
    """Define y alinea el target de clasificación binaria sobre un DataFrame OHLCV."""

    def __init__(self, horizon: int = 5, threshold: float = 0.005):
        """
        Args:
            horizon: número de barras hacia delante para calcular el retorno futuro.
            threshold: umbral de retorno para clasificar como subida (1) o no (0).
        """
        self.horizon = horizon
        self.threshold = threshold

    def build_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade columnas 'retorno_futuro' y 'target' al DataFrame.

        - retorno_futuro = cierre[t+horizon] / cierre[t] - 1
        - target = 1 si retorno_futuro > threshold, 0 si no.

        Las últimas *horizon* filas quedan con target = NaN (no se puede calcular).
        """
        df = df.copy()
        df["retorno_futuro"] = df["cierre"].shift(-self.horizon) / df["cierre"] - 1
        df["target"] = (df["retorno_futuro"] > self.threshold).astype(float)
        df.loc[df["retorno_futuro"].isna(), "target"] = float("nan")
        return df
