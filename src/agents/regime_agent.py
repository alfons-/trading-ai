"""
RegimeAgent: asigna por fila un régimen de mercado (bull / bear / sideways).

Usa solo columnas ya mergeadas del TF superior (p. ej. weekly) sin lookahead.
"""

from __future__ import annotations

import pandas as pd

REGIME_BULL = "bull"
REGIME_BEAR = "bear"
REGIME_SIDEWAYS = "sideways"

ALL_REGIMES = (REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS)


class RegimeAgent:
    """Clasifica régimen con tendencia semanal + ADX."""

    def __init__(
        self,
        adx_trending_min: float = 20.0,
        trend_column: str = "weekly_trend",
        adx_column: str = "weekly_adx",
    ):
        self.adx_trending_min = float(adx_trending_min)
        self.trend_column = trend_column
        self.adx_column = adx_column

    def assign_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade columna categórica 'regime': bull | bear | sideways.

        - Bull: tendencia alcista y ADX >= umbral (mercado trending arriba).
        - Bear: tendencia bajista y ADX >= umbral.
        - Sideways: ADX débil o sin tendencia clara.
        """
        df = df.copy()
        missing = [c for c in (self.trend_column, self.adx_column) if c not in df.columns]
        if missing:
            raise KeyError(
                f"RegimeAgent: faltan columnas {missing}. "
                "Activa higher_timeframes.weekly en FeatureAgent."
            )

        trend = df[self.trend_column]
        adx = df[self.adx_column]
        strong = adx.notna() & (adx >= self.adx_trending_min)

        regime = pd.Series(REGIME_SIDEWAYS, index=df.index, dtype=object)
        regime.loc[(trend == 1) & strong] = REGIME_BULL
        regime.loc[(trend == 0) & strong] = REGIME_BEAR
        regime.loc[trend.isna() | adx.isna()] = REGIME_SIDEWAYS

        df["regime"] = regime
        return df
