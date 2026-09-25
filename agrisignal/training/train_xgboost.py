"""
training/train_xgboost.py
──────────────────────────
XGBoost model training pipeline.

  - Walk-forward cross-validation (time-series correct, no lookahead bias)
  - MLflow: one run per training with params, metrics, the training dataset
    (content digest) and artifacts; each model is registered as a new version
  - SHAP: feature importance for model transparency
  - Joblib: fast model serialization / deserialization
  - RobustScaler: handles futures price outliers better than StandardScaler
  - All params in config.yaml — zero hardcoded hyperparameters
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.data
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler

from agrisignal.utils import get_logger, load_config

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
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "da": float(np.mean(np.sign(y_true) == np.sign(y_pred))),  # Directional accuracy
        "corr": float(np.corrcoef(y_true, y_pred)[0, 1]),
    }


# ─────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────


class XGBoostTrainer:

    def __init__(self, config_path: str | None = None):
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
        Full training run with walk-forward CV, then MLflow tracking.

        Training never depends on MLflow: if tracking fails, the model is
        still trained and saved, and the failure is logged.
        """
        X_train, y_train, X_test, y_test = self._prepare_data(df, feature_cols)
        params = {k: v for k, v in self.m_cfg["params"].items() if k != "eval_metric"}

        # ── Walk-forward CV ────────────────────────────────────────
        wfcv = WalkForwardCV(
            n_splits=self.m_cfg["n_cv_splits"],
            gap=self.m_cfg["gap_days"],
        )
        cv_results = []

        for fold_i, (tr_idx, val_idx) in enumerate(wfcv.split(X_train)):
            fold_model = xgb.XGBRegressor(**params, verbosity=0)
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

        log.info(
            f"Walk-forward CV ({self.m_cfg['n_cv_splits']} folds): "
            f"MAE={cv_means['mae']:.4f}±{cv_stds['mae']:.4f} | "
            f"DA={cv_means['da']:.3f}±{cv_stds['da']:.3f}"
        )

        # ── Final model on full training set ──────────────────────
        final_model = xgb.XGBRegressor(**params, verbosity=0)
        final_model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # ── Test set evaluation ────────────────────────────────────
        test_preds = final_model.predict(X_test)
        test_metrics = compute_metrics(y_test.values, test_preds)
        test_metrics = {k: round(v, 2) for k, v in test_metrics.items()}

        log.info(
            f"Test holdout: "
            f"MAE={test_metrics['mae']:.4f} | "
            f"RMSE={test_metrics['rmse']:.4f} | "
            f"R²={test_metrics['r2']:.4f} | "
            f"DA={test_metrics['da']:.3f}"
        )

        # ── SHAP feature importance ────────────────────────────────
        shap_importance = self._compute_shap(final_model, X_test, feature_cols)
        imp_path = self.models_path / "shap_importance.csv"
        shap_importance.to_csv(imp_path, index=False)
        log.info(
            "Top 5 features: "
            + " | ".join(
                f"{r['feature']} ({r['mean_abs_shap']:.4f})"
                for _, r in shap_importance.head(5).iterrows()
            )
        )

        # ── Persist artifacts ──────────────────────────────────────
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

        # ── MLflow tracking + model registry ───────────────────────
        run_params = {
            **self.m_cfg["params"],
            "n_features": len(feature_cols),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "n_cv_splits": self.m_cfg["n_cv_splits"],
            "gap_days": self.m_cfg["gap_days"],
        }
        run_metrics = {
            **{f"cv_{k}_mean": v for k, v in cv_means.items()},
            **{f"cv_{k}_std": v for k, v in cv_stds.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        # The served metadata records which MLflow run, model version and
        # dataset produced this model, so the API can report what it serves.
        metadata["mlflow"] = self._track(
            df,
            run_params,
            run_metrics,
            final_model,
            infer_signature(X_test, test_preds),
            metadata,
            [imp_path, scaler_path],
        )
        meta_path.write_text(json.dumps(metadata, indent=2))

        return final_model, test_metrics

    def _track(
        self,
        df: pd.DataFrame,
        params: dict,
        metrics: dict[str, float],
        model: xgb.XGBRegressor,
        signature,
        metadata: dict,
        artifacts: list[Path],
    ) -> dict:
        """
        Log one MLflow run (params, metrics, training dataset, artifacts) and
        register the model as a new version aliased "champion", since the
        latest trained model is the one the API serves.

        Returns the run ID, model version and dataset digest, or {} if MLflow
        is unavailable.
        """
        ml_cfg = self.cfg["mlflow"]
        model_name = ml_cfg["registered_model_name"]
        try:
            # MLFLOW_TRACKING_URI (set on Cloud Run) overrides the config file
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ml_cfg["tracking_uri"]))
            mlflow.set_experiment(ml_cfg["experiment_name"])

            # Dataset versioning: the digest is a content hash of the gold matrix
            gold_path = Path(self.cfg["storage"]["gold"]) / "gold_features.parquet"
            dataset = mlflow.data.from_pandas(
                df,
                source=str(gold_path.resolve()),
                name="gold_features",
                targets="target",
            )

            with mlflow.start_run(run_name="xgboost_corn") as run:
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.log_input(dataset, context="training")
                mlflow.set_tags(
                    {
                        "dataset.digest": dataset.digest,
                        "dataset.rows": len(df),
                        "dataset.start_date": str(df["date"].min().date()),
                        "dataset.end_date": str(df["date"].max().date()),
                    }
                )
                git_sha = os.getenv("AGRISIGNAL_GIT_SHA")
                if git_sha:
                    mlflow.set_tag("mlflow.source.git.commit", git_sha)

                for path in artifacts:
                    mlflow.log_artifact(str(path))
                mlflow.log_dict(metadata, "model_metadata.json")

                model_info = mlflow.xgboost.log_model(
                    model,
                    name="model",
                    signature=signature,
                    registered_model_name=model_name,
                )

            version = model_info.registered_model_version
            client = mlflow.MlflowClient()
            client.set_registered_model_alias(model_name, "champion", version)
            client.set_model_version_tag(model_name, version, "dataset_digest", dataset.digest)
        except Exception as e:
            log.warning(f"MLflow tracking failed, model saved without it: {e}")
            return {}

        log.info(
            f"MLflow run {run.info.run_id} | {model_name} v{version} (champion) | "
            f"dataset digest {dataset.digest}"
        )
        return {
            "run_id": run.info.run_id,
            "registered_model": model_name,
            "model_version": int(version),
            "dataset_digest": dataset.digest,
        }

    def _compute_shap(
        self,
        model: xgb.XGBRegressor,
        X: pd.DataFrame,
        feature_cols: list[str],
    ) -> pd.DataFrame:
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X)
            return (
                pd.DataFrame(
                    {
                        "feature": feature_cols,
                        "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
                    }
                )
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True)
            )
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
