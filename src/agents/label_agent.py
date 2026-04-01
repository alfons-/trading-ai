"""
LabelAgent: construye la variable objetivo (target) para XGBoost.
"""

from __future__ import annotations

import pandas as pd

from src.agents.regime_agent import REGIME_BEAR, REGIME_BULL, REGIME_SIDEWAYS


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

    def build_regime_aware_target(
        self,
        df: pd.DataFrame,
        *,
        regime_column: str = "regime",
        range_threshold: float = 0.003,
        sideways_target_mode: str = "range",
    ) -> pd.DataFrame:
        """
        Añade 'retorno_futuro' y 'target' según el régimen de cada fila.

        - Bull: target=1 si retorno_futuro > threshold (favorable long).
        - Bear: target=1 si retorno_futuro < -threshold (favorable short).
        - Sideways (range): target=1 si abs(retorno_futuro) < range_threshold.
        - Sideways (bounce_up): target=1 si retorno_futuro > range_threshold.
        """
        if regime_column not in df.columns:
            raise KeyError(f"Falta columna '{regime_column}' (RegimeAgent.assign_regime).")

        df = df.copy()
        rf = df["cierre"].shift(-self.horizon) / df["cierre"] - 1
        df["retorno_futuro"] = rf

        reg = df[regime_column]
        tgt = pd.Series(float("nan"), index=df.index, dtype=float)

        m_bull = reg == REGIME_BULL
        m_bear = reg == REGIME_BEAR
        m_side = reg == REGIME_SIDEWAYS

        tgt.loc[m_bull] = (rf.loc[m_bull] > self.threshold).astype(float)
        tgt.loc[m_bear] = (rf.loc[m_bear] < -self.threshold).astype(float)
        abs_rf = rf.abs()
        side_mode = str(sideways_target_mode).strip().lower()
        if side_mode == "bounce_up":
            tgt.loc[m_side] = (rf.loc[m_side] > range_threshold).astype(float)
        else:
            tgt.loc[m_side] = (abs_rf.loc[m_side] < range_threshold).astype(float)

        tgt.loc[rf.isna()] = float("nan")
        df["target"] = tgt
        return df
