"""
monitoring/data_quality.py
──────────────────────────
Data quality checks run as a pipeline gate before model training.

Checks are designed to catch:
  - Source data outages (row count drops)
  - Schema drift (unexpected nulls / type changes)
  - Feature distribution shift (PSI-based drift detection)
  - Target leakage (correlation sanity check)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agrisignal.utils import get_logger, load_config

log = get_logger(__name__)


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    PSI: measures how much a distribution has shifted.
      PSI < 0.10 → no significant shift
      PSI 0.10–0.20 → moderate shift (monitor)
      PSI > 0.20 → significant shift (consider retraining)
    """
    bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0

    ref_pcts, _ = np.histogram(reference, bins=bins)
    cur_pcts, _ = np.histogram(current, bins=bins)

    ref_pcts = ref_pcts / len(reference)
    cur_pcts = cur_pcts / len(current)

    ref_pcts = np.where(ref_pcts == 0, 1e-6, ref_pcts)
    cur_pcts = np.where(cur_pcts == 0, 1e-6, cur_pcts)

    return float(np.sum((cur_pcts - ref_pcts) * np.log(cur_pcts / ref_pcts)))


def run_quality_checks(
    df: pd.DataFrame,
    config_path: str = "configs/config.yaml",
) -> tuple[bool, dict]:
    """
    Run all data quality checks on the Gold feature matrix.

    Returns:
        (all_passed: bool, report: dict of check results)
    """
    cfg = load_config(config_path)
    report = {}
    all_passed = True

    # ── Check 1: Minimum row count ────────────────────────────────
    trading_days_expected = cfg["pipeline"]["lookback_years"] * 252
    min_rows = int(trading_days_expected * cfg["monitoring"]["min_row_fraction"])
    row_check = len(df) >= min_rows
    report["row_count"] = {
        "passed": row_check,
        "value": len(df),
        "threshold": min_rows,
        "message": f"{len(df):,} rows (min: {min_rows:,})",
    }
    if not row_check:
        all_passed = False

    # ── Check 2: Target null rate ─────────────────────────────────
    target_null_pct = df["target"].isna().mean()
    horizon = cfg["features"]["target_horizon_days"]
    # Expected: only the last `horizon` rows are NaN
    expected_null_pct = horizon / len(df)
    target_ok = target_null_pct <= expected_null_pct * 3  # 3x tolerance
    report["target_nulls"] = {
        "passed": target_ok,
        "value": round(target_null_pct * 100, 2),
        "threshold": round(expected_null_pct * 3 * 100, 2),
        "message": f"Target null rate: {target_null_pct:.2%}",
    }
    if not target_ok:
        all_passed = False

    # ── Check 3: No future leakage (sanity) ──────────────────────
    # Close price should NOT be highly correlated with next day's target
    # (If it is, there's likely a shift error)
    try:
        corr_close_target = df[["close", "target"]].dropna().corr().iloc[0, 1]
        no_leakage = abs(corr_close_target) < 0.9
        report["leakage_check"] = {
            "passed": no_leakage,
            "value": round(corr_close_target, 4),
            "threshold": 0.9,
            "message": f"Close~Target corr: {corr_close_target:.4f} (should be < 0.9)",
        }
        if not no_leakage:
            all_passed = False
    except Exception as e:
        report["leakage_check"] = {"passed": False, "message": str(e)}
        all_passed = False

    # ── Check 4: Date continuity (no large gaps) ──────────────────
    df_sorted = df.sort_values("date")
    date_diffs = df_sorted["date"].diff().dt.days.dropna()
    gap_date = date_diffs.max()
    max_gap = int(gap_date)
    # Allow up to 5 consecutive non-trading days (holidays, weekends)
    gap_ok = max_gap <= 5
    report["date_continuity"] = {
        "passed": gap_ok,
        "value": max_gap,
        "threshold": 5,
        "message": f"Max date gap: {str(gap_date)} days (threshold: 5)",
    }
    if not gap_ok:
        # Warning only — don't fail pipeline for this
        log.warning(f"Max gap above threshold: {max_gap}")

    # ── Check 5: Feature null rate ────────────────────────────────
    feature_cols = [
        c
        for c in df.columns
        if c not in ("date", "target", "target_horizon", "open", "high", "low", "close", "volume")
        and df[c].dtype in (float, np.float64, int, np.int64)
    ]
    high_null_features = {
        col: round(df[col].isna().mean() * 100, 1)
        for col in feature_cols
        if df[col].isna().mean() > 0.10
    }
    features_ok = len(high_null_features) == 0
    report["feature_nulls"] = {
        "passed": features_ok,
        "value": len(high_null_features),
        "message": f"Features with >10% nulls: {high_null_features or 'none'}",
    }
    if not features_ok:
        # Warning only — don't fail pipeline for this
        log.warning(f"High-null features: {high_null_features}")

    # ── Check 6: PSI drift on key features ───────────────────────
    psi_threshold = cfg["monitoring"]["psi_critical"]
    split_idx = len(df) // 2
    drift_issues = {}
    for col in ["rsi", "rvol_20d", "gdd_cumulative", "close"]:
        if col not in df.columns:
            continue
        ref = df[col].iloc[:split_idx].dropna().values
        cur = df[col].iloc[split_idx:].dropna().values
        if len(ref) < 20 or len(cur) < 20:
            continue
        psi = population_stability_index(ref, cur)
        if psi > psi_threshold:
            drift_issues[col] = round(psi, 4)

    drift_ok = len(drift_issues) == 0
    report["distribution_drift"] = {
        "passed": drift_ok,
        "value": drift_issues,
        "message": f"PSI drift detected: {drift_issues or 'none'}",
    }
    if not drift_ok:
        log.warning(f"Feature drift: {drift_issues}")

    n_passed = sum(1 for v in report.values() if v.get("passed"))
    log.info(f"Data quality: {n_passed}/{len(report)} checks passed | overall={all_passed}")
    return all_passed, report
