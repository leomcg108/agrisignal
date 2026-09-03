"""
transforms/silver.py  —  SILVER LAYER
──────────────────────────────────────
Transforms raw bronze data into a clean, typed, deduplicated,
and joined silver dataset.

Responsibilities:
  1. Parse and standardise types from bronze
  2. Aggregate Corn Belt weather to single daily series
  3. Join futures + weather + correlated instruments on date
  4. Enforce SilverSchema contract
  5. Log data quality metrics

What does NOT belong here:
  - Feature engineering (that's Gold)
  - Model logic
  - Any business assumptions beyond cleaning
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agrisignal.transforms.schemas import DataContractError, SilverSchema, validate
from agrisignal.utils import ParquetStore, get_logger, load_config

log = get_logger(__name__)


class SilverTransform:
    """
    Produces the silver dataset: one row per trading day, all sources joined.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.cfg = load_config(config_path)
        self.t_cfg = self.cfg["transforms"]
        self.bronze_store = ParquetStore(self.cfg["storage"]["bronze"])
        self.silver_store = ParquetStore(self.cfg["storage"]["silver"])

    # ── Weather cleaning ──────────────────────────────────────────

    def _clean_weather(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot NOAA long-format data to wide, convert units, aggregate stations.

        NOAA delivers:
          date | datatype | value (in tenths of °C or tenths of mm)

        We produce (Corn Belt daily average):
          date | tmax_f | tmin_f | prcp_in | snow_mm
        """
        df = raw.copy()

        # Normalise date
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()

        # Pivot datatypes into columns
        pivoted = df.pivot_table(
            index=["date", "region"],
            columns="datatype",
            values="value",
            aggfunc="first",
        ).reset_index()
        pivoted.columns.name = None

        # Unit conversions (NOAA uses tenths of degrees C / tenths of mm)
        if "TMAX" in pivoted.columns:
            pivoted["tmax_f"] = (pivoted["TMAX"] / 10) * 9 / 5 + 32
            pivoted["tmin_f"] = (pivoted["TMIN"] / 10) * 9 / 5 + 32
        if "PRCP" in pivoted.columns:
            pivoted["prcp_in"] = pivoted["PRCP"] / 10 / 25.4  # tenths-mm → inches
        if "SNOW" in pivoted.columns:
            pivoted["snow_mm"] = pivoted["SNOW"] / 10

        # Average across Corn Belt stations for each day
        weather_cols = ["tmax_f", "tmin_f", "prcp_in", "snow_mm"]
        present_cols = [c for c in weather_cols if c in pivoted.columns]
        agg = pivoted.groupby("date")[present_cols].mean().reset_index()

        # Data quality check: warn if coverage is sparse
        null_rates = agg[present_cols].isna().mean()
        for col, rate in null_rates.items():
            if rate > self.t_cfg["max_null_rate"]:
                log.warning(
                    f"Weather column {col}: {rate:.1%} nulls (threshold {self.t_cfg['max_null_rate']:.0%})"
                )

        log.info(f"Weather silver: {len(agg):,} daily records")
        return agg

    # ── Futures cleaning ──────────────────────────────────────────

    def _clean_futures(self, raw: pd.DataFrame, label: str = "corn") -> pd.DataFrame:
        """
        Clean yfinance OHLCV into typed, sorted trading day series.
        """
        df = raw[raw["label"] == label].copy() if "label" in raw.columns else raw.copy()

        # Normalise column names from yfinance
        df.columns = [c.lower().strip() for c in df.columns]
        date_col = next((c for c in df.columns if c in ("date", "datetime")), None)
        if not date_col:
            raise DataContractError("Futures data missing Date column")

        df["date"] = pd.to_datetime(df[date_col]).dt.normalize()

        # Drop duplicates and sort
        df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

        # Drop clearly bad rows
        min_vol = self.t_cfg["min_daily_volume"]
        before = len(df)
        df = df[df["volume"] >= min_vol]
        if len(df) < before:
            log.warning(f"Dropped {before - len(df)} rows with volume < {min_vol}")

        # Returns
        df["returns_1d"] = df["close"].pct_change(1)
        df["log_return_1d"] = np.log(df["close"] / df["close"].shift(1)).replace(
            [np.inf, -np.inf], np.nan
        )
        df["high_low_pct"] = (df["high"] - df["low"]) / df["close"]

        return df[
            [
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "returns_1d",
                "log_return_1d",
                "high_low_pct",
            ]
        ]

    def _clean_correlated(
        self,
        all_futures: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Join correlated instrument close prices onto a date spine."""
        correlated = {}
        for label in ["wheat", "crude", "usd"]:
            if label not in all_futures:
                continue
            raw = all_futures[label]
            raw.columns = [c.lower().strip() for c in raw.columns]
            date_col = next((c for c in raw.columns if c in ("date", "datetime")), None)
            if not date_col:
                continue
            sub = raw[[date_col, "close"]].copy()
            sub["date"] = pd.to_datetime(sub[date_col]).dt.normalize()
            sub = sub.drop_duplicates("date").rename(columns={"close": f"{label}_close"})
            correlated[label] = sub[["date", f"{label}_close"]]

        if not correlated:
            return pd.DataFrame(columns=["date"])

        # Merge all on date
        result = next(iter(correlated.values()))
        for sub in list(correlated.values())[1:]:
            result = result.merge(sub, on="date", how="outer")
        return result

    # ── Join ──────────────────────────────────────────────────────

    def build(
        self,
        bronze_weather: pd.DataFrame,
        bronze_futures: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Build the silver dataset by joining cleaned sources.

        Join strategy:
          - Corn futures is the spine (trading days only)
          - Weather joined on nearest date (forward-filled up to 3 days)
          - Correlated instruments: left join, nulls allowed for holidays
        """
        log.info("Building silver layer...")

        corn = self._clean_futures(bronze_futures.get("corn", pd.DataFrame()), "corn")
        weather = self._clean_weather(bronze_weather)
        correlated = self._clean_correlated(bronze_futures)

        # Merge weather (nearest date, fill gaps up to 3 days)
        silver = pd.merge_asof(
            corn.sort_values("date"),
            weather.sort_values("date"),
            on="date",
            direction="nearest",
            tolerance=pd.Timedelta("3 days"),
        )

        # Join correlated instruments
        if not correlated.empty and "date" in correlated.columns:
            silver = silver.merge(correlated, on="date", how="left")

        # Calendar fields
        silver["day_of_week"] = silver["date"].dt.dayofweek
        silver["month"] = silver["date"].dt.month

        # Validate schema contract
        silver = validate(silver, SilverSchema, layer="silver")

        # Persist
        path = self.silver_store.write(silver, "silver")
        log.info(f"Silver complete: {len(silver):,} rows → {path}")

        # Log quality summary
        self._log_quality_summary(silver)
        return silver

    def _log_quality_summary(self, df: pd.DataFrame) -> None:
        """Log a concise data quality summary after silver build."""
        null_pcts = (df.isna().mean() * 100).round(1)
        high_null = null_pcts[null_pcts > 5].to_dict()
        log.info(
            f"Silver quality summary: "
            f"rows={len(df):,} | "
            f"date_range={df['date'].min().date()} → {df['date'].max().date()} | "
            f"cols_with_nulls > 5% = {high_null or 'none'}"
        )

    def read(self) -> pd.DataFrame:
        return self.silver_store.read("silver")
