"""
training/train_xgboost.py
──────────────────────────
XGBoost model training pipeline.

  - Walk-forward cross-validation (time-series correct, no lookahead bias)
  - MLflow: tracks params, metrics, artifacts, registers model
  - SHAP: feature importance for model transparency
  - Joblib: fast model serialization / deserialization
  - RobustScaler: handles futures price outliers better than StandardScaler
  - All params in config.yaml — zero hardcoded hyperparameters
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler

from agrisignal.utils import ParquetStore, get_logger, load_config


log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# Walk-Forward Cross-Validator
# ─────────────────────────────────────────────────────────────────

class WalkForwardCV:
    """
    Expanding-window walk-forward cross-validation for time series.

    Each fold uses ALL past data for training (expanding window),
    with a mandatory gap between train end and validation start
    to prevent lookahead leakage on the target variable.

    Example with n_splits=4, gap=5, fold_size=50:
      Fold 1: train[0:50]     → gap → val[55:105]
      Fold 2: train[0:100]    → gap → val[105:155]
      Fold 3: train[0:150]    → gap → val[155:205]
      Fold 4: train[0:200]    → gap → val[205:255]
    """

    def __init__(self, n_splits: int = 5, gap: int = 5):
        self.n_splits = n_splits
        self.gap = gap

    def split(self, X: pd.DataFrame):
        n = len(X)
        fold_size = n // (self.n_splits + 1)

        for i in range(1, self.n_splits + 1):
            train_end = fold_size * i
            val_start = train_end + self.gap
            val_end = min(val_start + fold_size, n)
            if val_start >= n:
                break
            yield list(range(train_end)), list(range(val_start, val_end))


# ─────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae":    float(mean_absolute_error(y_true, y_pred)),
        "rmse":   float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":     float(r2_score(y_true, y_pred)),
        "da":     float(np.mean(np.sign(y_true) == np.sign(y_pred))),  # Directional accuracy
        "corr":   float(np.corrcoef(y_true, y_pred)[0, 1]),
    }


# ─────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────

class XGBoostTrainer:

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.cfg = load_config(config_path)
        self.m_cfg = self.cfg["model"]
        self.models_path = Path(self.cfg["storage"]["models"])
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.scaler = RobustScaler()

    def _prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Chronological train/test split.
        Scaler is fit only on training data to prevent data leakage.
        """
        df = df.sort_values("date").dropna(subset=["target"] + feature_cols)
        test_n = int(len(df) * self.m_cfg["test_size"])
        train, test = df.iloc[:-test_n], df.iloc[-test_n:]

        X_tr = pd.DataFrame(
            self.scaler.fit_transform(train[feature_cols]),
            columns=feature_cols,
        )
        X_te = pd.DataFrame(
            self.scaler.transform(test[feature_cols]),
            columns=feature_cols,
        )
        return X_tr, train["target"], X_te, test["target"]

    def train(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> tuple[xgb.XGBRegressor, dict]:
        """
        Full training run with walk-forward CV and MLflow tracking.
        """
        X_train, y_train, X_test, y_test = self._prepare_data(df, feature_cols)

        mlflow_available = False

        try:
            mlflow.set_tracking_uri(self.cfg["mlflow"]["tracking_uri"])
            mlflow.set_experiment(self.cfg["mlflow"]["experiment_name"])
            mlflow_available = True
            log.info("MLflow tracking enabled")
        except Exception as e:
            log.warning(f"MLflow unavailable, metrics will not be logged: {e}")

        if mlflow_available:
            try:
                with mlflow.start_run(run_name="xgboost_corn"):
                    mlflow.log_params({
                        **self.m_cfg["params"],
                        "n_features": len(feature_cols),
                        "train_samples": len(X_train),
                        "test_samples": len(X_test),
                        "n_cv_splits": self.m_cfg["n_cv_splits"],
                        "gap_days": self.m_cfg["gap_days"],
                    })
            except Exception as e:
                log.warning(f"Failed to log to MLflow: {e}")

            # ── Walk-forward CV ────────────────────────────────────
            wfcv = WalkForwardCV(
                n_splits=self.m_cfg["n_cv_splits"],
                gap=self.m_cfg["gap_days"],
            )
            cv_results = []

            for fold_i, (tr_idx, val_idx) in enumerate(wfcv.split(X_train)):
                fold_model = xgb.XGBRegressor(
                    **{k: v for k, v in self.m_cfg["params"].items() if k != "eval_metric"},
                    verbosity=0,
                )
                fold_model.fit(
                    X_train.iloc[tr_idx],
                    y_train.iloc[tr_idx],
                    eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
                    verbose=False,
                )
                preds = fold_model.predict(X_train.iloc[val_idx])
                m = compute_metrics(y_train.iloc[val_idx].values, preds)
                cv_results.append(m)
                log.debug(
                    f"  CV fold {fold_i+1}: "
                    f"MAE={m['mae']:.4f} | DA={m['da']:.3f} | R²={m['r2']:.4f}"
                )

            cv_df = pd.DataFrame(cv_results)
            cv_means = cv_df.mean().to_dict()
            cv_stds = cv_df.std().to_dict()
            for k, v in cv_means.items():
                mlflow.log_metric(f"cv_{k}_mean", v)
                mlflow.log_metric(f"cv_{k}_std", cv_stds[k])

            log.info(
                f"Walk-forward CV ({self.m_cfg['n_cv_splits']} folds): "
                f"MAE={cv_means['mae']:.4f}±{cv_stds['mae']:.4f} | "
                f"DA={cv_means['da']:.3f}±{cv_stds['da']:.3f}"
            )

            # ── Final model on full training set ──────────────────
            params = {k: v for k, v in self.m_cfg["params"].items() if k != "eval_metric"}
            final_model = xgb.XGBRegressor(**params, verbosity=0)
            final_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )

            # ── Test set evaluation ────────────────────────────────
            test_preds = final_model.predict(X_test)
            test_metrics = compute_metrics(y_test.values, test_preds)
            test_metrics = {k : round(v, 2) for k, v in test_metrics.items()}
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

            log.info(
                f"Test holdout: "
                f"MAE={test_metrics['mae']:.4f} | "
                f"RMSE={test_metrics['rmse']:.4f} | "
                f"R²={test_metrics['r2']:.4f} | "
                f"DA={test_metrics['da']:.3f}"
            )

            # ── SHAP feature importance ────────────────────────────
            shap_importance = self._compute_shap(final_model, X_test, feature_cols)
            imp_path = self.models_path / "shap_importance.csv"
            shap_importance.to_csv(imp_path, index=False)
            mlflow.log_artifact(str(imp_path))
            log.info(
                f"Top 5 features: "
                + " | ".join(
                    f"{r['feature']} ({r['mean_abs_shap']:.4f})"
                    for _, r in shap_importance.head(5).iterrows()
                )
            )

            # ── Persist artifacts ──────────────────────────────────
            model_path = self.models_path / "xgboost_corn.json"
            scaler_path = self.models_path / "scaler_corn.pkl"
            meta_path = self.models_path / "model_metadata.json"

            final_model.save_model(str(model_path))
            joblib.dump(self.scaler, scaler_path)

            metadata = {
                "feature_cols": feature_cols,
                "n_features": len(feature_cols),
                "test_metrics": test_metrics,
                "cv_metrics_mean": cv_means,
                "target_horizon_days": self.cfg["features"]["target_horizon_days"],
            }
            meta_path.write_text(json.dumps(metadata, indent=2))

            if mlflow_available:
                mlflow.xgboost.log_model(final_model, "model")
                mlflow.log_artifact(str(scaler_path))
                mlflow.log_artifact(str(meta_path))

        return final_model, test_metrics

    def _compute_shap(
        self,
        model: xgb.XGBRegressor,
        X: pd.DataFrame,
        feature_cols: list[str],
    ) -> pd.DataFrame:
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X)
            return pd.DataFrame({
                "feature": feature_cols,
                "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
            }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        except Exception as exc:
            log.warning(f"SHAP computation failed: {exc}")
            return pd.DataFrame({"feature": feature_cols, "mean_abs_shap": 0})

    def load(self) -> tuple[xgb.XGBRegressor, RobustScaler, dict]:
        """Load saved model, scaler, and metadata."""
        model = xgb.XGBRegressor()
        model.load_model(str(self.models_path / "xgboost_corn.json"))
        scaler = joblib.load(self.models_path / "scaler_corn.pkl")
        metadata = json.loads((self.models_path / "model_metadata.json").read_text())
        return model, scaler, metadata
