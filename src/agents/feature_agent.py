"""
FeatureAgent: construye features técnicas sobre un DataFrame OHLCV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.ma import add_sma
from src.indicators.momentum import add_rsi


class FeatureAgent:
    """Genera columnas de features a partir de un DataFrame de precios OHLCV."""

    def __init__(
        self,
        sma_corta: int = 10,
        sma_larga: int = 50,
        rsi_window: int = 14,
        volatility_window: int = 20,
        return_lags: list[int] | None = None,
    ):
        self.sma_corta = sma_corta
        self.sma_larga = sma_larga
        self.rsi_window = rsi_window
        self.volatility_window = volatility_window
        self.return_lags = return_lags or [1, 2, 3, 5]

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade columnas de features al DataFrame y devuelve la lista de nombres
        de features generadas.

        Modifica df in-place y lo devuelve.
        """
        df = df.copy()

        # Medias móviles
        add_sma(df, column="cierre", window=self.sma_corta, nombre_columna="sma_corta")
        add_sma(df, column="cierre", window=self.sma_larga, nombre_columna="sma_larga")

        # RSI
        add_rsi(df, column="cierre", window=self.rsi_window, nombre_columna="rsi")

        # Retorno logarítmico
        df["retorno"] = np.log(df["cierre"] / df["cierre"].shift(1))

        # Volatilidad (desviación estándar de retornos en ventana)
        df["volatilidad"] = df["retorno"].rolling(self.volatility_window).std()

        # Volumen normalizado (respecto a media móvil de volumen)
        df["vol_norm"] = df["volumen"] / df["volumen"].rolling(self.sma_larga).mean()

        # Ratio SMA corta / SMA larga
        df["sma_ratio"] = df["sma_corta"] / df["sma_larga"]

        # Lags de retorno
        for lag in self.return_lags:
            df[f"ret_lag_{lag}"] = df["retorno"].shift(lag)

        return df

    @property
    def feature_names(self) -> list[str]:
        """Nombres de las columnas de features generadas."""
        base = [
            "sma_corta", "sma_larga", "rsi", "retorno",
            "volatilidad", "vol_norm", "sma_ratio",
        ]
        lags = [f"ret_lag_{lag}" for lag in self.return_lags]
        return base + lags
