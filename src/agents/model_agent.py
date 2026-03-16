"""
ModelAgent: entrena, evalúa, guarda y carga un modelo XGBoost.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class ModelAgent:
    """Encapsula un XGBClassifier para señales de trading."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        eval_metric: str = "logloss",
        early_stopping_rounds: int = 20,
    ):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
            use_label_encoder=False,
            verbosity=0,
        )
        self._is_trained = False

    def train(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
    ) -> dict:
        """
        Entrena el modelo. Si se pasan datos de validación, usa early stopping.

        Returns:
            dict con métricas de entrenamiento.
        """
        fit_params: dict = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False

        self.model.fit(X_train, y_train, **fit_params)
        self._is_trained = True

        train_pred = self.model.predict(X_train)
        metrics = {"train_accuracy": accuracy_score(y_train, train_pred)}

        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_proba = self.model.predict_proba(X_val)[:, 1]
            metrics["val_accuracy"] = accuracy_score(y_val, val_pred)
            try:
                metrics["val_auc"] = roc_auc_score(y_val, val_proba)
            except ValueError:
                metrics["val_auc"] = float("nan")

        print(f"[ModelAgent] Entrenamiento completado: {metrics}")
        return metrics

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Devuelve etiquetas predichas (0 o 1)."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Devuelve probabilidades [prob_0, prob_1] por fila."""
        return self.model.predict_proba(X)

    def evaluate(
        self,
        X_test: pd.DataFrame | np.ndarray,
        y_test: pd.Series | np.ndarray,
    ) -> dict:
        """Evalúa el modelo en un conjunto de test y muestra reporte."""
        pred = self.predict(X_test)
        proba = self.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, pred)
        try:
            auc = roc_auc_score(y_test, proba)
        except ValueError:
            auc = float("nan")

        report = classification_report(y_test, pred, zero_division=0)
        print(f"[ModelAgent] Test — accuracy: {acc:.4f}, AUC: {auc:.4f}")
        print(report)
        return {"test_accuracy": acc, "test_auc": auc, "report": report}

    def save(self, path: str | Path | None = None) -> Path:
        """Guarda el modelo con joblib."""
        if path is None:
            models_dir = _PROJECT_ROOT / "models"
            models_dir.mkdir(exist_ok=True)
            path = models_dir / "xgb_model.joblib"
        path = Path(path)
        joblib.dump(self.model, path)
        print(f"[ModelAgent] Modelo guardado en {path}")
        return path

    def load(self, path: str | Path) -> None:
        """Carga un modelo previamente guardado."""
        self.model = joblib.load(path)
        self._is_trained = True
        print(f"[ModelAgent] Modelo cargado desde {path}")
