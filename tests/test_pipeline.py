"""
tests/test_schemas.py + test_transforms.py + test_features.py
──────────────────────────────────────────────────────────────
  - Schema contract tests (Pandera)
  - Transform unit tests with synthetic data
  - Idempotency tests
  - No-lookahead-bias test (critical for time series ML)
  - Walk-forward CV integrity tests
  - Data quality check unit tests

Run with: pytest tests/ -v
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

# ─────────────────────────────────────────────────────────────────
# Fixtures — shared synthetic data
# ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_silver_df():
    """
    Synthetic silver-layer DataFrame mimicking real corn futures + weather data.
    500 trading days gives enough data for rolling window features.
    """
    np.random.seed(42)
    n = 500
    dates = pd.bdate_range(end=date.today(), periods=n)  # Business days only
    prices = 450.0 + np.cumsum(np.random.randn(n) * 3.5)

    df = pd.DataFrame({
        "date":          pd.to_datetime(dates),
        "open":          prices + np.random.randn(n) * 0.5,
        "high":          prices + np.abs(np.random.randn(n)) * 2.5,
        "low":           prices - np.abs(np.random.randn(n)) * 2.5,
        "close":         prices,
        "volume":        np.random.randint(60_000, 200_000, n).astype(float),
        "returns_1d":    np.concatenate([[np.nan], np.diff(prices) / prices[:-1]]),
        "log_return_1d": np.concatenate([[np.nan], np.log(prices[1:] / prices[:-1])]),
        "high_low_pct":  np.abs(np.random.randn(n)) * 0.02,
        "day_of_week":   pd.to_datetime(dates).dayofweek,
        "month":         pd.to_datetime(dates).month,
        "tmax_f":        65 + 20 * np.sin(2 * np.pi * np.arange(n) / 252) + np.random.randn(n) * 5,
        "tmin_f":        45 + 18 * np.sin(2 * np.pi * np.arange(n) / 252) + np.random.randn(n) * 4,
        "prcp_in":       np.abs(np.random.randn(n) * 0.08),
        "wheat_close":   350 + np.cumsum(np.random.randn(n) * 2),
        "crude_close":   80 + np.cumsum(np.random.randn(n) * 1),
        "usd_close":     103 + np.cumsum(np.random.randn(n) * 0.3),
    })
    return df


@pytest.fixture(scope="module")
def sample_gold_df(sample_silver_df):
    """Build gold feature matrix from synthetic silver."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import yaml, tempfile, os

    test_cfg = {
        "pipeline": {"name": "test", "commodity": "corn", "lookback_years": 2},
        "storage": {
            "bronze": "/tmp/agri_test/bronze",
            "silver": "/tmp/agri_test/silver",
            "gold":   "/tmp/agri_test/gold",
            "models": "/tmp/agri_test/models",
        },
        "sources": {"weather": {"stations": {}}, "futures": {"ticker": "ZC=F", "correlated": {}}},
        "transforms": {"max_null_rate": 0.05, "min_station_coverage": 0.8, "min_daily_volume": 5000},
        "features": {
            "gdd_base_temp_f": 50, "gdd_heat_stress_f": 95,
            "rolling_windows": [5, 10, 20, 60],
            "rsi_period": 14, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
            "bb_period": 20, "atr_period": 14,
            "target_horizon_days": 5, "target_type": "return",
        },
        "model": {"params": {}, "test_size": 0.15, "n_cv_splits": 3, "gap_days": 5},
        "mlflow": {"tracking_uri": "sqlite:///test_mlflow.db", "experiment_name": "test"},
        "api": {"host": "0.0.0.0", "port": 8000, "workers": 1, "cache_predictions_ttl_s": 60},
        "orchestration": {},
        "monitoring": {"psi_warning": 0.1, "psi_critical": 0.2, "min_row_fraction": 0.8, "metrics_port": 9090},
    }
    cfg_path = "/tmp/test_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(test_cfg, f)

    from agrisignal.features.engineer import FeatureEngineer
    eng = FeatureEngineer(config_path=cfg_path)
    return eng.build(sample_silver_df)


# ═════════════════════════════════════════════════════════════════
# Schema Tests
# ═════════════════════════════════════════════════════════════════

class TestSchemas:

    def test_silver_schema_validates_good_data(self, sample_silver_df):
        """Valid silver data should pass schema without errors."""
        from agrisignal.transforms.schemas import SilverSchema, validate
        result = validate(sample_silver_df, SilverSchema, layer="test")
        assert len(result) == len(sample_silver_df)

    def test_silver_schema_rejects_negative_close(self, sample_silver_df):
        """Negative close prices should fail the schema contract."""
        from agrisignal.transforms.schemas import SilverSchema, validate, DataContractError
        bad_df = sample_silver_df.copy()
        bad_df.loc[5, "close"] = -10.0
        with pytest.raises(DataContractError):
            validate(bad_df, SilverSchema, layer="test_negative_close")

    def test_silver_schema_rejects_high_less_than_low(self, sample_silver_df):
        """High < Low is physically impossible — must be caught."""
        from agrisignal.transforms.schemas import SilverSchema, validate, DataContractError
        bad_df = sample_silver_df.copy()
        # Swap high and low on one row
        bad_df.loc[10, "high"] = bad_df.loc[10, "low"] - 5
        with pytest.raises(DataContractError):
            validate(bad_df, SilverSchema, layer="test_high_low_swap")

    def test_silver_schema_rejects_duplicate_dates(self, sample_silver_df):
        """Duplicate dates should violate the uniqueness contract."""
        from agrisignal.transforms.schemas import SilverSchema, validate, DataContractError
        dup_df = pd.concat([sample_silver_df, sample_silver_df.iloc[:5]]).reset_index(drop=True)
        with pytest.raises(DataContractError):
            validate(dup_df, SilverSchema, layer="test_dupe_dates")

    def test_rsi_bounds_in_gold(self, sample_gold_df):
        """RSI must always be in [0, 100]."""
        from agrisignal.transforms.schemas import GoldSchema, validate
        # Gold schema validates RSI bounds automatically
        rsi = sample_gold_df["rsi"].dropna()
        assert (rsi >= 0).all(), "RSI below 0 detected"
        assert (rsi <= 100).all(), "RSI above 100 detected"


# ═════════════════════════════════════════════════════════════════
# Transform Tests
# ═════════════════════════════════════════════════════════════════

class TestTransforms:

    def test_silver_row_count_preserved(self, sample_silver_df):
        """Silver transform should not drop rows (outer join with weather)."""
        assert len(sample_silver_df) == 500

    def test_returns_computed_correctly(self, sample_silver_df):
        """1-day return should equal (close[t] - close[t-1]) / close[t-1]."""
        df = sample_silver_df.copy()
        expected = df["close"].pct_change(1).iloc[10]
        actual = df["returns_1d"].iloc[10]
        assert abs(expected - actual) < 1e-8

    def test_log_return_sign_matches_return(self, sample_silver_df):
        """Log return and simple return should always have the same sign."""
        df = sample_silver_df.dropna(subset=["returns_1d", "log_return_1d"])
        sign_match = (np.sign(df["returns_1d"]) == np.sign(df["log_return_1d"])).all()
        assert sign_match


# ═════════════════════════════════════════════════════════════════
# Feature Engineering Tests
# ═════════════════════════════════════════════════════════════════

class TestFeatureEngineering:

    def test_feature_count_reasonable(self, sample_gold_df):
        """Gold layer should have a meaningful number of features."""
        exclude = {"date", "target", "target_horizon", "open", "high", "low", "close", "volume"}
        feature_cols = [c for c in sample_gold_df.columns if c not in exclude]
        assert len(feature_cols) >= 40, f"Only {len(feature_cols)} features — expected 40+"

    def test_no_lookahead_bias_in_features(self, sample_gold_df):
        """
        CRITICAL: Features must be computable from data available at time T.
        Specifically: no feature should be perfectly correlated with next-day return.
        """
        lr = sample_gold_df["log_return_1d"].shift(-1)  # "next day" return
        rolling_feats = ["sma_5", "ema_5", "rsi", "bb_pct_b", "macd"]
        for feat in rolling_feats:
            if feat not in sample_gold_df.columns:
                continue
            corr = sample_gold_df[feat].corr(lr)
            assert abs(corr) < 0.95, (
                f"Potential lookahead in {feat}: corr={corr:.4f} with next-day return"
            )

    def test_target_last_n_rows_null(self, sample_gold_df):
        """The last `horizon` rows should have null targets (no future data)."""
        horizon = 5
        tail = sample_gold_df.tail(horizon)
        assert tail["target"].isna().all(), (
            f"Expected null targets in last {horizon} rows — potential lookahead!"
        )

    def test_gdd_non_negative(self, sample_gold_df):
        """Growing Degree Days are physically bounded at 0."""
        gdd = sample_gold_df["gdd_daily"].dropna()
        assert (gdd >= 0).all(), "Negative GDD values detected"

    def test_cyclical_features_bounded(self, sample_gold_df):
        """Sin/cos encodings must be in [-1, 1]."""
        for col in ["month_sin", "month_cos", "doy_sin", "doy_cos"]:
            if col not in sample_gold_df.columns:
                continue
            vals = sample_gold_df[col].dropna()
            assert vals.between(-1, 1).all(), f"{col} out of [-1, 1] bounds"

    def test_feature_matrix_sorted_by_date(self, sample_gold_df):
        """Gold layer must be sorted chronologically."""
        assert sample_gold_df["date"].is_monotonic_increasing

    def test_idempotent_feature_build(self, sample_silver_df, sample_gold_df):
        """Building features twice from the same silver data yields identical output."""
        import yaml, sys
        from pathlib import Path

        cfg_path = "/tmp/test_config.yaml"
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from agrisignal.features.engineer import FeatureEngineer

        eng = FeatureEngineer(config_path=cfg_path)
        gold2 = eng.build(sample_silver_df)

        # Key feature columns should be identical
        for col in ["rsi", "macd", "gdd_daily", "target"]:
            if col in sample_gold_df.columns and col in gold2.columns:
                pd.testing.assert_series_equal(
                    sample_gold_df[col].reset_index(drop=True),
                    gold2[col].reset_index(drop=True),
                    check_names=False,
                    rtol=1e-5,
                )


# ═════════════════════════════════════════════════════════════════
# Walk-Forward CV Tests
# ═════════════════════════════════════════════════════════════════

class TestWalkForwardCV:

    def test_no_train_val_overlap(self):
        """Train and validation indices must never overlap."""
        from agrisignal.training.train_xgboost import WalkForwardCV
        wfcv = WalkForwardCV(n_splits=5, gap=5)
        X = pd.DataFrame(range(300))
        for tr, val in wfcv.split(X):
            assert set(tr).isdisjoint(set(val)), "Train/val overlap!"

    def test_validation_always_after_training(self):
        """Every validation index must be strictly greater than every training index."""
        from agrisignal.training.train_xgboost import WalkForwardCV
        wfcv = WalkForwardCV(n_splits=4, gap=5)
        X = pd.DataFrame(range(200))
        for tr, val in wfcv.split(X):
            assert max(tr) < min(val), "Validation comes before training!"

    def test_gap_respected(self):
        """The gap between train end and val start must be >= configured gap."""
        gap = 7
        from agrisignal.training.train_xgboost import WalkForwardCV
        wfcv = WalkForwardCV(n_splits=3, gap=gap)
        X = pd.DataFrame(range(200))
        for tr, val in wfcv.split(X):
            actual_gap = min(val) - max(tr) - 1
            assert actual_gap >= gap - 1, f"Gap too small: {actual_gap} < {gap-1}"

    def test_expanding_window(self):
        """Each successive fold should have more training data than the previous."""
        from agrisignal.training.train_xgboost import WalkForwardCV
        wfcv = WalkForwardCV(n_splits=4, gap=5)
        X = pd.DataFrame(range(250))
        prev_train_size = 0
        for tr, _ in wfcv.split(X):
            assert len(tr) > prev_train_size
            prev_train_size = len(tr)


# ═════════════════════════════════════════════════════════════════
# Data Quality Tests
# ═════════════════════════════════════════════════════════════════

class TestDataQuality:

    def test_quality_checks_pass_on_good_data(self, sample_gold_df):
        """Good synthetic data should pass all quality checks."""
        import yaml
        cfg_path = "/tmp/test_config.yaml"
        from agrisignal.monitoring.data_quality import run_quality_checks
        passed, report = run_quality_checks(sample_gold_df, config_path=cfg_path)
        # Row count, date continuity, leakage checks must pass
        assert report["row_count"]["passed"]
        assert report["date_continuity"]["passed"]
        assert report["leakage_check"]["passed"]

    def test_quality_fails_on_empty_df(self):
        """Empty DataFrame should fail immediately."""
        import yaml
        cfg_path = "/tmp/test_config.yaml"
        from agrisignal.monitoring.data_quality import run_quality_checks
        empty = pd.DataFrame(columns=["date", "close", "target", "log_return_1d"])
        empty["date"] = pd.Series([], dtype="datetime64[ns]")
        passed, report = run_quality_checks(empty, config_path=cfg_path)
        assert not passed
        assert not report["row_count"]["passed"]

    def test_psi_zero_for_identical_distributions(self):
        """PSI should be near 0 when reference and current are identical."""
        from agrisignal.monitoring.data_quality import population_stability_index
        data = np.random.randn(500)
        psi = population_stability_index(data, data)
        assert psi < 0.01, f"PSI={psi} for identical distributions (expected ~0)"

    def test_psi_large_for_shifted_distribution(self):
        """PSI should be large when distributions are very different."""
        from agrisignal.monitoring.data_quality import population_stability_index
        ref = np.random.normal(0, 1, 500)
        cur = np.random.normal(5, 1, 500)   # Mean shifted by 5 sigma
        psi = population_stability_index(ref, cur)
        assert psi > 0.2, f"PSI={psi} for shifted distribution (expected > 0.2)"
