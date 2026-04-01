"""
OrchestratorAgent: coordina el pipeline completo de investigación.

Flujo: DataAgent → FeatureAgent → LabelAgent → ModelAgent → BacktestAgent
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.agents.data_agent import DEFAULT_BYBIT_CATEGORY, DataAgent
from src.agents.feature_agent import FeatureAgent
from src.agents.label_agent import LabelAgent
from src.agents.model_agent import ModelAgent
from src.agents.backtest_agent import BacktestAgent
from src.agents.regime_agent import ALL_REGIMES, RegimeAgent

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_scale_pos_weight(y: pd.Series, cfg_val) -> float:
    """
    Si cfg_val es la cadena 'auto', usa neg/pos en el train (cap 25).
    Si no, interpreta como float (default 1.0).
    """
    if isinstance(cfg_val, str) and cfg_val.strip().lower() == "auto":
        yv = y.astype(float)
        neg = int((yv == 0).sum())
        pos = int((yv == 1).sum())
        if pos < 1:
            return 1.0
        w = float(neg) / float(pos)
        return float(min(max(w, 1.0), 25.0))
    try:
        return float(cfg_val) if cfg_val is not None else 1.0
    except (TypeError, ValueError):
        return 1.0


class OrchestratorAgent:
    """Ejecuta un experimento completo a partir de un archivo de configuración YAML."""

    def __init__(self, config_path: str | Path | None = None):
        if config_path is None:
            config_path = _PROJECT_ROOT / "configs" / "default.yaml"
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = _PROJECT_ROOT / config_path
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

    def _get_timeframe_base(self) -> str:
        return self.cfg.get("timeframe_base") or self.cfg.get("timeframe", "4h")

    def _get_all_timeframes(self) -> list[str]:
        """Devuelve lista de todos los TFs a descargar (base + higher)."""
        tfs = [self._get_timeframe_base()]
        ht = self.cfg.get("higher_timeframes", {})
        for _name, ht_cfg in ht.items():
            if ht_cfg.get("enabled") and "timeframe" in ht_cfg:
                tf = ht_cfg["timeframe"]
                if tf not in tfs:
                    tfs.append(tf)
        return tfs

    def run(self, symbol: str | None = None) -> dict:
        """
        Ejecuta el pipeline para un símbolo.

        Si *symbol* es None, usa el primer símbolo de la config.
        Devuelve un dict con métricas del modelo y del backtest.
        """
        if symbol is None:
            symbol = self.cfg["symbols"][0]

        tf_base = self._get_timeframe_base()

        print(f"\n{'='*60}")
        print(f"  Experimento: {symbol} | base={tf_base}")
        print(f"{'='*60}\n")

        # 1. Datos (base + higher timeframes)
        print("── 1. Descargando datos ──")
        bybit_cat = str(self.cfg.get("bybit_category", DEFAULT_BYBIT_CATEGORY)).strip().lower()
        print(f"   Bybit category: {bybit_cat} (spot = contado / velas normales)")
        data_agent = DataAgent(category=bybit_cat)
        days = self.cfg["history_days"]
        all_tfs = self._get_all_timeframes()

        force_dl = bool(self.cfg.get("force_refresh_data", False))
        dfs_by_tf: dict[str, pd.DataFrame] = {}
        for tf in all_tfs:
            tf_df = data_agent.get_ohlcv(
                symbol=symbol, timeframe=tf, days=days, force=force_dl
            )
            dfs_by_tf[tf] = tf_df
            print(f"   {tf}: {len(tf_df)} filas → {symbol}_{tf}.csv")

        df = dfs_by_tf[tf_base]
        print(f"   Base ({tf_base}): {len(df)} filas")

        # 2. Features (base + higher timeframes)
        print("\n── 2. Construyendo features ──")
        feat_cfg = self.cfg["features"]
        ht_cfg = self.cfg.get("higher_timeframes", {})
        feature_agent = FeatureAgent(
            sma_corta=feat_cfg["sma_corta"],
            sma_larga=feat_cfg["sma_larga"],
            rsi_window=feat_cfg["rsi_window"],
            volatility_window=feat_cfg["volatility_window"],
            return_lags=feat_cfg["return_lags"],
            macd_cfg=feat_cfg.get("macd", {}),
            sr_cfg=feat_cfg.get("support_resistance", {}),
            higher_timeframes_cfg=ht_cfg,
        )
        df = feature_agent.build_features(df, higher_dfs=dfs_by_tf)
        feature_cols = feature_agent.feature_names
        print(f"   Features: {feature_cols}")

        if self.cfg.get("multi_regime"):
            return self._run_multi_regime(symbol, df, feature_cols)

        # 3. Target
        print("\n── 3. Construyendo target ──")
        tgt_cfg = self.cfg["target"]
        label_agent = LabelAgent(
            horizon=tgt_cfg["horizon"],
            threshold=tgt_cfg["threshold"],
        )
        df = label_agent.build_target(df)

        # Limpiar NaN (por rolling windows y shift del target)
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
        spw = _resolve_scale_pos_weight(y_train, xgb_cfg.get("scale_pos_weight", 1.0))
        print(f"   scale_pos_weight={spw:.4f} (target train: {y_train.value_counts().to_dict()})")
        model_agent = ModelAgent(
            n_estimators=xgb_cfg["n_estimators"],
            max_depth=xgb_cfg["max_depth"],
            learning_rate=xgb_cfg["learning_rate"],
            subsample=xgb_cfg["subsample"],
            colsample_bytree=xgb_cfg["colsample_bytree"],
            scale_pos_weight=spw,
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
            min_hold_bars=bt_cfg.get("min_hold_bars", 0),
        )
        test_probas = model_agent.predict_proba(X_test)[:, 1]
        bt_result = backtest_agent.run(df_test, test_probas, symbol=symbol)

        # 8. Guardar modelo
        model_agent.save()

        print(f"\n{'='*60}")
        print("  Experimento finalizado")
        print(f"{'='*60}\n")

        return {
            "symbol": symbol,
            "multi_regime": False,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "backtest": bt_result["metrics"],
        }

    def _run_multi_regime(self, symbol: str, df: pd.DataFrame, feature_cols: list[str]) -> dict:
        """Pipeline tres modelos (bull/bear/sideways) + backtest alineado."""
        print("\n── 3. Régimen + target por régimen ──")
        reg_cfg = self.cfg.get("regimes", {})
        regime_agent = RegimeAgent(
            adx_trending_min=float(reg_cfg.get("adx_trending_min", 20)),
            trend_column=str(reg_cfg.get("trend_column", "weekly_trend")),
            adx_column=str(reg_cfg.get("adx_column", "weekly_adx")),
        )
        df = regime_agent.assign_regime(df)

        tgt_cfg = self.cfg["target"]
        rt_cfg = self.cfg.get("regime_targets", {})
        range_th = float(rt_cfg.get("range_threshold", 0.003))
        sideways_target_mode = str(rt_cfg.get("sideways_target_mode", "range"))
        label_agent = LabelAgent(
            horizon=tgt_cfg["horizon"],
            threshold=tgt_cfg["threshold"],
        )
        df = label_agent.build_regime_aware_target(
            df,
            range_threshold=range_th,
            sideways_target_mode=sideways_target_mode,
        )

        df_clean = df.dropna(subset=feature_cols + ["target", "regime"]).reset_index(drop=True)
        print(f"   Filas válidas tras limpieza: {len(df_clean)}")
        print(f"   Regímenes (train+val+test): {df_clean['regime'].value_counts().to_dict()}")

        print("\n── 4. Split temporal (índices globales; máscara por régimen) ──")
        split_cfg = self.cfg["split"]
        n = len(df_clean)
        n_train = int(n * split_cfg["train_ratio"])
        n_val = int(n * split_cfg["val_ratio"])

        df_train = df_clean.iloc[:n_train]
        df_val = df_clean.iloc[n_train:n_train + n_val]
        df_test = df_clean.iloc[n_train + n_val:]
        print(f"   Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

        print("   Target por régimen (train):")
        for reg in ALL_REGIMES:
            sub = df_train[df_train["regime"] == reg]["target"]
            if len(sub) == 0:
                print(f"      {reg}: (sin filas)")
            else:
                print(f"      {reg}: {sub.value_counts().to_dict()}")

        rm_cfg = self.cfg.get("regime_models", {})
        path_tpl = str(rm_cfg.get("path_template", "models/xgb_{regime}_{symbol}.joblib"))
        min_samples = int(rm_cfg.get("min_samples", 500))
        xgb_cfg = self.cfg["xgboost"]

        regime_results: dict = {}
        models: dict = {}

        print("\n── 5–6. Entrenar / evaluar un XGBoost por régimen ──")
        for regime in ALL_REGIMES:
            tr = df_train[df_train["regime"] == regime]
            va = df_val[df_val["regime"] == regime]
            te = df_test[df_test["regime"] == regime]
            if len(tr) < min_samples:
                print(
                    f"   [{regime}] Omitido: train={len(tr)} < min_samples={min_samples}"
                )
                regime_results[regime] = {
                    "skipped": True,
                    "n_train": len(tr),
                    "n_val": len(va),
                    "n_test": len(te),
                }
                models[regime] = None
                continue

            X_tr, y_tr = tr[feature_cols], tr["target"]
            X_va, y_va = va[feature_cols], va["target"]
            X_te, y_te = te[feature_cols], te["target"]

            spw = _resolve_scale_pos_weight(y_tr, xgb_cfg.get("scale_pos_weight", 1.0))
            print(f"   [{regime}] scale_pos_weight={spw:.4f} | target train: {y_tr.value_counts().to_dict()}")

            model_agent = ModelAgent(
                n_estimators=xgb_cfg["n_estimators"],
                max_depth=xgb_cfg["max_depth"],
                learning_rate=xgb_cfg["learning_rate"],
                subsample=xgb_cfg["subsample"],
                colsample_bytree=xgb_cfg["colsample_bytree"],
                scale_pos_weight=spw,
                eval_metric=xgb_cfg["eval_metric"],
                early_stopping_rounds=xgb_cfg["early_stopping_rounds"],
            )
            train_m = model_agent.train(X_tr, y_tr, X_va, y_va)
            test_m = model_agent.evaluate(X_te, y_te) if len(te) > 0 else {}

            rel_path = path_tpl.format(regime=regime, symbol=symbol)
            out_path = _PROJECT_ROOT / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            model_agent.save(out_path)

            regime_results[regime] = {
                "skipped": False,
                "n_train": len(tr),
                "n_val": len(va),
                "n_test": len(te),
                "train_metrics": train_m,
                "test_metrics": test_m,
                "model_path": str(out_path),
            }
            models[regime] = model_agent

        print("\n── 7. Backtest multi-régimen (test) ──")
        probas = np.full(len(df_test), 0.5, dtype=np.float64)
        for regime, m in models.items():
            if m is None:
                continue
            mask = df_test["regime"].to_numpy() == regime
            if not mask.any():
                continue
            probas[mask] = m.predict_proba(df_test.loc[mask, feature_cols])[:, 1]

        rb_cfg = self.cfg.get("regime_backtest", {})
        bt_inner = self.cfg["backtest"]
        backtest_agent = BacktestAgent(
            prob_buy_threshold=bt_inner["prob_buy_threshold"],
            prob_sell_threshold=bt_inner["prob_sell_threshold"],
            min_hold_bars=rb_cfg.get("min_hold_bars", bt_inner.get("min_hold_bars", 0)),
        )
        th = {
            "bull": rb_cfg.get("bull", bt_inner),
            "bear": rb_cfg.get("bear", {}),
            "sideways": rb_cfg.get("sideways", bt_inner),
        }
        bt_result = backtest_agent.run_regime(
            df_test,
            probas,
            thresholds=th,
            min_hold_bars=rb_cfg.get("min_hold_bars", bt_inner.get("min_hold_bars", 0)),
            symbol=symbol,
        )

        print(f"\n{'='*60}")
        print("  Experimento multi-régimen finalizado")
        print(f"{'='*60}\n")

        return {
            "symbol": symbol,
            "multi_regime": True,
            "regimes": regime_results,
            "backtest": bt_result["metrics"],
            "backtest_by_regime": bt_result.get("metrics_by_regime", {}),
        }

    def run_all(self) -> list[dict]:
        """Ejecuta el pipeline para todos los símbolos de la config."""
        results = []
        for symbol in self.cfg["symbols"]:
            result = self.run(symbol)
            results.append(result)
        return results
