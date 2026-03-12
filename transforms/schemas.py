"""
transforms/schemas.py  —  DATA CONTRACTS
──────────────────────────────────────────
Pandera schema definitions for every layer boundary in the pipeline.

  - Catch data quality issues at the source, not after a model is corrupted
  - Make implicit assumptions explicit and testable
  - Every layer boundary has a contract — violations raise DataContractError
  - Schema checks run in < 1ms for typical daily datasets

Layer contracts:
  BronzeWeatherSchema   → Raw NOAA output
  BronzeFuturesSchema   → Raw yfinance output
  SilverSchema          → Cleaned, joined daily record
  GoldSchema            → Feature matrix (model-ready)
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera import Column, DataFrameSchema, Check
import pandas as pd


# ─────────────────────────────────────────────────────────────────
# Bronze layer schemas — minimal constraints, raw types
# ─────────────────────────────────────────────────────────────────

BronzeWeatherSchema = DataFrameSchema(
    columns={
        "date":      Column(object,   nullable=False),
        "datatype":  Column(str,      Check.isin(["TMAX", "TMIN", "PRCP", "SNOW"]),
                            nullable=False),
        "value":     Column(float,    nullable=True),   # NOAA occasionally has gaps
        "station_id": Column(str,     nullable=False),
        "region":    Column(str,      nullable=False),
    },
    checks=[
        Check(lambda df: len(df) > 0, error="Bronze weather DataFrame is empty"),
    ],
    coerce=True,    # Cast types where safe
    strict=False,   # Allow extra columns (audit metadata)
    name="BronzeWeather",
)

BronzeFuturesSchema = DataFrameSchema(
    columns={
        "Date":   Column(object,  nullable=False),
        "Open":   Column(float,   Check.greater_than(0), nullable=False),
        "High":   Column(float,   Check.greater_than(0), nullable=False),
        "Low":    Column(float,   Check.greater_than(0), nullable=False),
        "Close":  Column(float,   Check.greater_than(0), nullable=False),
        "Volume": Column(float,   Check.greater_than_or_equal_to(0), nullable=False),
        "ticker": Column(str,     nullable=False),
        "label":  Column(str,     nullable=False),
    },
    checks=[
        Check(
            lambda df: (df["High"] >= df["Low"]).all(),
            error="High must be >= Low for all rows",
        ),
        Check(
            lambda df: (df["High"] >= df["Open"]).all(),
            error="High must be >= Open",
        ),
    ],
    coerce=True,
    strict=False,
    name="BronzeFutures",
)


# ─────────────────────────────────────────────────────────────────
# Silver schema — cleaned, strongly typed, no nulls in key fields
# ─────────────────────────────────────────────────────────────────

SilverSchema = DataFrameSchema(
    columns={
        # Identity
        "date":           Column(pa.DateTime, nullable=False),
        "day_of_week":    Column(int, Check.isin(range(7)), nullable=False),

        # Futures OHLCV (required, no nulls)
        "open":           Column(float, Check.greater_than(0), nullable=False),
        "high":           Column(float, Check.greater_than(0), nullable=False),
        "low":            Column(float, Check.greater_than(0), nullable=False),
        "close":          Column(float, Check.greater_than(0), nullable=False),
        "volume":         Column(float, Check.greater_than_or_equal_to(0), nullable=False),
        "returns_1d":     Column(float, nullable=True),  # NaN for first row
        "log_return_1d":  Column(float, nullable=True),

        # Correlated instruments (nullable — may not trade same days)
        "wheat_close":    Column(float, nullable=True),
        "crude_close":    Column(float, nullable=True),
        "usd_close":      Column(float, nullable=True),

        # Weather aggregates (Corn Belt average, nullable for non-trading-day coverage)
        "tmax_f":         Column(float, nullable=True),
        "tmin_f":         Column(float, nullable=True),
        "prcp_in":        Column(float, Check.greater_than_or_equal_to(0), nullable=True),
    },
    checks=[
        Check(
            lambda df: df["date"].is_monotonic_increasing,
            error="Silver: dates must be sorted ascending",
        ),
        Check(
            lambda df: df["date"].nunique() == len(df),
            error="Silver: duplicate dates detected",
        ),
        Check(
            lambda df: (df["high"] >= df["low"]).all(),
            error="Silver: high < low detected",
        ),
        Check(
            lambda df: df["close"].notna().mean() >= 0.95,
            error="Silver: more than 5% of close prices are null",
        ),
    ],
    coerce=True,
    strict=False,
    name="Silver",
)


# ─────────────────────────────────────────────────────────────────
# Gold schema — feature matrix, model-ready
# ─────────────────────────────────────────────────────────────────

GoldSchema = DataFrameSchema(
    columns={
        "date":         Column(pa.DateTime, nullable=False),
        "close":        Column(float, Check.greater_than(0), nullable=False),

        # A sample of required feature columns (others validated by name pattern)
        "rsi":          Column(float, Check.in_range(0, 100), nullable=True),
        "bb_pct_b":     Column(float, nullable=True),
        "gdd_cumulative": Column(float, Check.greater_than_or_equal_to(0), nullable=True),

        # Target (nullable — last N rows will be NaN)
        "target":       Column(float, nullable=True),
    },
    checks=[
        Check(
            lambda df: df["date"].is_monotonic_increasing,
            error="Gold: dates must be sorted ascending",
        ),
        Check(
            # At least 80% of rows must have a valid target
            lambda df: df["target"].notna().mean() >= 0.80,
            error="Gold: target has too many nulls (< 80% valid)",
        ),
    ],
    coerce=True,
    strict=False,
    name="Gold",
)


# ─────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────

class DataContractError(Exception):
    """Raised when a DataFrame fails a schema contract."""
    pass


def validate(
    df: pd.DataFrame,
    schema: DataFrameSchema,
    layer: str = "",
) -> pd.DataFrame:
    """
    Validate a DataFrame against a Pandera schema.
    Raises DataContractError with a helpful message on failure.

    Usage:
        df = validate(raw_df, BronzeWeatherSchema, layer="bronze/weather")
    """
    from agrisignal.utils import get_logger
    log = get_logger(__name__)

    try:
        validated = schema.validate(df, lazy=True)
        log.info(f"Schema OK [{schema.name}] layer={layer} rows={len(df):,}")
        return validated
    except pa.errors.SchemaErrors as exc:
        failure_summary = exc.failure_cases.to_string(index=False)
        msg = (
            f"Data contract violation in [{schema.name}] layer={layer}\n"
            f"{len(exc.failure_cases)} failures:\n{failure_summary}"
        )
        log.error(msg)
        raise DataContractError(msg) from exc
