"""
OrchestratorAgent: coordina el pipeline completo de investigación.

Flujo: DataAgent → FeatureAgent → LabelAgent → ModelAgent → BacktestAgent
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.agents.data_agent import DataAgent
from src.agents.feature_agent import FeatureAgent
from src.agents.label_agent import LabelAgent
from src.agents.model_agent import ModelAgent
from src.agents.backtest_agent import BacktestAgent

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class OrchestratorAgent:
    """Ejecuta un experimento completo a partir de un archivo de configuración YAML."""

    def __init__(self, config_path: str | Path | None = None):
        if config_path is None:
            config_path = _PROJECT_ROOT / "configs" / "default.yaml"
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

    def run(self, symbol: str | None = None) -> dict:
        """
        Ejecuta el pipeline para un símbolo.

        Si *symbol* es None, usa el primer símbolo de la config.
        Devuelve un dict con métricas del modelo y del backtest.
        """
        if symbol is None:
            symbol = self.cfg["symbols"][0]

        print(f"\n{'='*60}")
        print(f"  Experimento: {symbol} | {self.cfg['timeframe']}")
        print(f"{'='*60}\n")

        # 1. Datos
        print("── 1. Descargando datos ──")
        data_agent = DataAgent()
        df = data_agent.get_ohlcv(
            symbol=symbol,
            timeframe=self.cfg["timeframe"],
            days=self.cfg["history_days"],
        )
        print(f"   Filas descargadas: {len(df)}")

        # 2. Features
        print("\n── 2. Construyendo features ──")
        feat_cfg = self.cfg["features"]
        feature_agent = FeatureAgent(
            sma_corta=feat_cfg["sma_corta"],
            sma_larga=feat_cfg["sma_larga"],
            rsi_window=feat_cfg["rsi_window"],
            volatility_window=feat_cfg["volatility_window"],
            return_lags=feat_cfg["return_lags"],
        )
        df = feature_agent.build_features(df)
        print(f"   Features: {feature_agent.feature_names}")

        # 3. Target
        print("\n── 3. Construyendo target ──")
        tgt_cfg = self.cfg["target"]
        label_agent = LabelAgent(
            horizon=tgt_cfg["horizon"],
            threshold=tgt_cfg["threshold"],
        )
        df = label_agent.build_target(df)

        # Limpiar NaN (por rolling windows y shift del target)
        feature_cols = feature_agent.feature_names
        df_clean = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
        print(f"   Filas válidas tras limpieza: {len(df_clean)}")
        print(f"   Distribución target: {df_clean['target'].value_counts().to_dict()}")

        # 4. Split temporal
        print("\n── 4. Split temporal ──")
        split_cfg = self.cfg["split"]
        n = len(df_clean)
        n_train = int(n * split_cfg["train_ratio"])
        n_val = int(n * split_cfg["val_ratio"])

        df_train = df_clean.iloc[:n_train]
        df_val = df_clean.iloc[n_train:n_train + n_val]
        df_test = df_clean.iloc[n_train + n_val:]

        X_train = df_train[feature_cols]
        y_train = df_train["target"]
        X_val = df_val[feature_cols]
        y_val = df_val["target"]
        X_test = df_test[feature_cols]
        y_test = df_test["target"]

        print(f"   Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

        # 5. Entrenar XGBoost
        print("\n── 5. Entrenando XGBoost ──")
        xgb_cfg = self.cfg["xgboost"]
        model_agent = ModelAgent(
            n_estimators=xgb_cfg["n_estimators"],
            max_depth=xgb_cfg["max_depth"],
            learning_rate=xgb_cfg["learning_rate"],
            subsample=xgb_cfg["subsample"],
            colsample_bytree=xgb_cfg["colsample_bytree"],
            eval_metric=xgb_cfg["eval_metric"],
            early_stopping_rounds=xgb_cfg["early_stopping_rounds"],
        )
        train_metrics = model_agent.train(X_train, y_train, X_val, y_val)

        # 6. Evaluar en test
        print("\n── 6. Evaluación en test ──")
        test_metrics = model_agent.evaluate(X_test, y_test)

        # 7. Backtest
        print("\n── 7. Backtest ──")
        bt_cfg = self.cfg["backtest"]
        backtest_agent = BacktestAgent(
            prob_buy_threshold=bt_cfg["prob_buy_threshold"],
            prob_sell_threshold=bt_cfg["prob_sell_threshold"],
        )
        test_probas = model_agent.predict_proba(X_test)[:, 1]
        bt_result = backtest_agent.run(df_test, test_probas)

        # 8. Guardar modelo
        model_agent.save()

        print(f"\n{'='*60}")
        print("  Experimento finalizado")
        print(f"{'='*60}\n")

        return {
            "symbol": symbol,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "backtest": bt_result["metrics"],
        }

    def run_all(self) -> list[dict]:
        """Ejecuta el pipeline para todos los símbolos de la config."""
        results = []
        for symbol in self.cfg["symbols"]:
            result = self.run(symbol)
            results.append(result)
        return results
