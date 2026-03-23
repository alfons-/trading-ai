"""
FeatureAgent: construye features técnicas sobre un DataFrame OHLCV.

Soporta multi-timeframe: si se pasan higher_dfs (1D, 1W), genera
features de contexto (SMA diaria, tendencia semanal) mediante merge_asof
para respetar el orden temporal y evitar leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ta.trend import ADXIndicator, MACD

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
        macd_cfg: dict | None = None,
        sr_cfg: dict | None = None,
        higher_timeframes_cfg: dict | None = None,
    ):
        self.sma_corta = sma_corta
        self.sma_larga = sma_larga
        self.rsi_window = rsi_window
        self.volatility_window = volatility_window
        self.return_lags = return_lags or [1, 2, 3, 5]
        self.macd_cfg = macd_cfg or {}
        self.sr_cfg = sr_cfg or {}
        self.ht_cfg = higher_timeframes_cfg or {}
        self._ht_feature_names: list[str] = []
        self._sr_feature_names: list[str] = []

    def build_features(
        self,
        df: pd.DataFrame,
        higher_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """
        Añade columnas de features al DataFrame base y lo devuelve.

        Args:
            df: DataFrame OHLCV del timeframe base (ej. 4h).
            higher_dfs: dict {timeframe: DataFrame} con TFs superiores (1D, 1W).
                        Si se pasan junto con higher_timeframes_cfg, se generan
                        features de contexto vía merge_asof.
        """
        df = df.copy()

        # ── Features del TF base ──
        add_sma(df, column="cierre", window=self.sma_corta, nombre_columna="sma_corta")
        add_sma(df, column="cierre", window=self.sma_larga, nombre_columna="sma_larga")
        add_rsi(df, column="cierre", window=self.rsi_window, nombre_columna="rsi")

        df["retorno"] = np.log(df["cierre"] / df["cierre"].shift(1))
        df["volatilidad"] = df["retorno"].rolling(self.volatility_window).std()
        df["vol_norm"] = df["volumen"] / df["volumen"].rolling(self.sma_larga).mean()
        df["sma_ratio"] = df["sma_corta"] / df["sma_larga"]

        for lag in self.return_lags:
            df[f"ret_lag_{lag}"] = df["retorno"].shift(lag)

        # ── MACD ──
        if self.macd_cfg.get("enabled"):
            macd_ind = MACD(
                close=df["cierre"],
                window_fast=self.macd_cfg.get("window_fast", 12),
                window_slow=self.macd_cfg.get("window_slow", 26),
                window_sign=self.macd_cfg.get("window_sign", 9),
            )
            df["macd"] = macd_ind.macd()
            df["macd_signal"] = macd_ind.macd_signal()
            df["macd_diff"] = macd_ind.macd_diff()

        # ── Soportes y Resistencias (rolling max/min en 4h) ──
        self._sr_feature_names = []
        if self.sr_cfg.get("enabled"):
            df = self._build_support_resistance(df)

        # ── Features de higher timeframes (1D, 1W) ──
        self._ht_feature_names = []
        if higher_dfs and self.ht_cfg:
            df = self._merge_higher_tf_features(df, higher_dfs)

        return df

    def _build_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade features de soporte/resistencia basadas en rolling max/min."""
        lookback = self.sr_cfg.get("lookback", 30)

        df["resistencia"] = df["alto"].rolling(lookback).max()
        df["soporte"] = df["bajo"].rolling(lookback).min()
        df["dist_resistencia"] = (df["resistencia"] - df["cierre"]) / df["cierre"]
        df["dist_soporte"] = (df["cierre"] - df["soporte"]) / df["cierre"]
        rango = df["resistencia"] - df["soporte"]
        df["posicion_rango"] = np.where(rango > 0, (df["cierre"] - df["soporte"]) / rango, 0.5)

        self._sr_feature_names = ["dist_resistencia", "dist_soporte", "posicion_rango"]
        return df

    def _merge_higher_tf_features(
        self,
        df: pd.DataFrame,
        higher_dfs: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Para cada higher TF configurado, calcula indicadores en ese TF
        y los une al DF base con merge_asof (solo usa velas ya cerradas).
        """
        for name, cfg in self.ht_cfg.items():
            if not cfg.get("enabled"):
                continue
            tf = cfg.get("timeframe")
            if not tf or tf not in higher_dfs:
                continue

            df_ht = higher_dfs[tf].copy()
            df_ht["fecha"] = pd.to_datetime(df_ht["fecha"])
            df_ht = df_ht.sort_values("fecha").reset_index(drop=True)
            prefix = name  # "daily" o "weekly"

            if name == "weekly":
                window = cfg.get("trend_sma_window", 20)
                rsi_w = cfg.get("rsi_window", 14)
                sma_col = f"{prefix}_sma_{window}"
                trend_col = f"{prefix}_trend"
                rsi_col = f"{prefix}_rsi"
                df_ht[sma_col] = df_ht["cierre"].rolling(window).mean()
                df_ht[trend_col] = (df_ht["cierre"] > df_ht[sma_col]).astype(int)
                add_rsi(df_ht, column="cierre", window=rsi_w, nombre_columna=rsi_col)

                feat_cols = [sma_col, trend_col, rsi_col]

                # Trend strength: EMA fast / EMA slow - 1 (continuo)
                ema_fast = cfg.get("trend_ema_fast", 10)
                ema_slow = cfg.get("trend_ema_slow", 30)
                ts_col = f"{prefix}_trend_strength"
                df_ht[ts_col] = df_ht["cierre"].ewm(span=ema_fast).mean() / df_ht["cierre"].ewm(span=ema_slow).mean() - 1
                feat_cols.append(ts_col)

                # ADX (fuerza de tendencia)
                adx_w = cfg.get("adx_window", 14)
                adx_ind = ADXIndicator(high=df_ht["alto"], low=df_ht["bajo"], close=df_ht["cierre"], window=adx_w)
                df_ht[f"{prefix}_adx"] = adx_ind.adx()
                df_ht[f"{prefix}_adx_pos"] = adx_ind.adx_pos()
                df_ht[f"{prefix}_adx_neg"] = adx_ind.adx_neg()
                feat_cols.extend([f"{prefix}_adx", f"{prefix}_adx_pos", f"{prefix}_adx_neg"])

                # Higher Highs / Higher Lows (estructura de tendencia)
                hh_lookback = cfg.get("hh_lookback", 4)
                df_ht["_hh"] = (df_ht["alto"] > df_ht["alto"].shift(1)).astype(int)
                df_ht["_hl"] = (df_ht["bajo"] > df_ht["bajo"].shift(1)).astype(int)
                df_ht[f"{prefix}_hh_count"] = df_ht["_hh"].rolling(hh_lookback).sum()
                df_ht[f"{prefix}_hl_count"] = df_ht["_hl"].rolling(hh_lookback).sum()
                feat_cols.extend([f"{prefix}_hh_count", f"{prefix}_hl_count"])

                merge_cols = ["fecha"] + feat_cols + [f"{prefix}_cierre"]
                df_ht = df_ht.rename(columns={"cierre": f"{prefix}_cierre"})
                df_ht = df_ht[merge_cols].dropna()
                self._ht_feature_names.extend(feat_cols + [f"{prefix}_cierre"])

            elif name == "daily":
                window = cfg.get("sma_window", 20)
                rsi_w = cfg.get("rsi_window", 14)
                sma_col = f"{prefix}_sma_{window}"
                rsi_col = f"{prefix}_rsi"
                df_ht[sma_col] = df_ht["cierre"].rolling(window).mean()
                add_rsi(df_ht, column="cierre", window=rsi_w, nombre_columna=rsi_col)

                feat_cols = [sma_col, rsi_col]

                # Pivot Points (soporte/resistencia diario)
                if cfg.get("pivot_points", False):
                    df_ht["_pivot"] = (df_ht["alto"] + df_ht["bajo"] + df_ht["cierre"]) / 3
                    df_ht[f"{prefix}_pivot"] = df_ht["_pivot"]
                    df_ht[f"{prefix}_r1"] = 2 * df_ht["_pivot"] - df_ht["bajo"]
                    df_ht[f"{prefix}_s1"] = 2 * df_ht["_pivot"] - df_ht["alto"]
                    feat_cols.extend([f"{prefix}_pivot", f"{prefix}_r1", f"{prefix}_s1"])

                merge_cols = ["fecha"] + feat_cols + [f"{prefix}_cierre"]
                df_ht = df_ht.rename(columns={"cierre": f"{prefix}_cierre"})
                df_ht = df_ht[merge_cols].dropna()
                self._ht_feature_names.extend(feat_cols + [f"{prefix}_cierre"])

            else:
                continue

            df["fecha"] = pd.to_datetime(df["fecha"])
            df = df.sort_values("fecha").reset_index(drop=True)
            df = pd.merge_asof(
                df,
                df_ht,
                on="fecha",
                direction="backward",
            )

        return df

    @property
    def feature_names(self) -> list[str]:
        """Nombres de las columnas de features generadas."""
        base = [
            "sma_corta", "sma_larga", "rsi", "retorno",
            "volatilidad", "vol_norm", "sma_ratio",
        ]
        lags = [f"ret_lag_{lag}" for lag in self.return_lags]
        macd_cols = ["macd", "macd_signal", "macd_diff"] if self.macd_cfg.get("enabled") else []
        return base + lags + macd_cols + self._sr_feature_names + self._ht_feature_names
