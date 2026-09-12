"""
features/engineer.py  —  GOLD LAYER
──────────────────────────────────────
Transforms the silver dataset into a model-ready feature matrix.

Feature groups:
  1. Price/return features    — momentum, mean-reversion signals
  2. Technical indicators     — RSI, MACD, Bollinger Bands, ATR, OBV
  3. Volatility features      — realized vol, vol regime
  4. Weather/agronomic        — GDD, heat stress, precipitation anomalies
  5. Seasonal/calendar        — cyclical encoding, crop calendar flags
  6. Cross-asset              — wheat spread, crude oil, USD index
  7. Target construction      — N-day forward return (no lookahead)

All features are deterministic, reversible, and logged.
Output is validated against GoldSchema before writing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agrisignal.transforms.schemas import GoldSchema, validate
from agrisignal.utils import ParquetStore, get_logger, load_config

log = get_logger(__name__)


class FeatureEngineer:
    """
    Builds the gold feature matrix from a validated silver DataFrame.
    """

    def __init__(self, config_path: str | None = None):
        self.cfg = load_config(config_path)
        self.f_cfg = self.cfg["features"]
        self.silver_store = ParquetStore(self.cfg["storage"]["silver"])
        self.gold_store = ParquetStore(self.cfg["storage"]["gold"])
        self._feature_names: list[str] = []

    # ─────────────────────────────────────────────────────────────
    # 1. Price & return features
    # ─────────────────────────────────────────────────────────────

    def _price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        windows = self.f_cfg["rolling_windows"]

        for w in windows:
            df[f"return_{w}d"] = close.pct_change(w)
            df[f"sma_{w}"] = close.rolling(w).mean()
            df[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()
            df[f"price_to_sma_{w}"] = (
                close / df[f"sma_{w}"] - 1
            )  # Price vs. SMA (Mean Reversion Signal)

        df["high_low_range"] = (df["high"] - df["low"]) / close
        df["overnight_gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
        df["volume_ratio_20"] = df["volume"] / df["volume"].rolling(20).mean()
        return df

    # ─────────────────────────────────────────────────────────────
    # 2. Technical indicators
    # ─────────────────────────────────────────────────────────────

    def _technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        vol = df["volume"]

        # RSI
        rsi_p = self.f_cfg["rsi_period"]
        delta = close.diff()
        up = delta.clip(lower=0).ewm(com=rsi_p - 1, adjust=False).mean()
        dn = (-delta.clip(upper=0)).ewm(com=rsi_p - 1, adjust=False).mean()
        df["rsi"] = 100 - (100 / (1 + up / dn.replace(0, np.nan)))

        # MACD
        fast, slow, sig = (
            self.f_cfg["macd_fast"],
            self.f_cfg["macd_slow"],
            self.f_cfg["macd_signal"],
        )
        ema_f = close.ewm(span=fast, adjust=False).mean()
        ema_s = close.ewm(span=slow, adjust=False).mean()
        df["macd"] = ema_f - ema_s
        df["macd_signal_line"] = df["macd"].ewm(span=sig, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal_line"]

        # Bollinger Bands
        bb_p = self.f_cfg["bb_period"]
        bb_mid = close.rolling(bb_p).mean()
        bb_std = close.rolling(bb_p).std()
        df["bb_upper"] = bb_mid + 2 * bb_std
        df["bb_lower"] = bb_mid - 2 * bb_std
        df["bb_pct_b"] = (close - df["bb_lower"]) / (
            (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
        )
        df["bb_bandwidth"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid

        # ATR (Average True Range)
        atr_p = self.f_cfg["atr_period"]
        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr"] = tr.rolling(atr_p).mean()
        df["atr_pct"] = df["atr"] / close

        # OBV (On Balance Volume)
        obv = (np.sign(close.diff()) * vol).fillna(0).cumsum()
        df["obv"] = obv
        df["obv_sma20"] = obv.rolling(20).mean()
        df["obv_ratio"] = obv / df["obv_sma20"].replace(0, np.nan)

        return df

    # ─────────────────────────────────────────────────────────────
    # 3. Volatility features
    # ─────────────────────────────────────────────────────────────

    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        lr = df["log_return_1d"]
        for w in [5, 10, 20, 60]:
            df[f"rvol_{w}d"] = lr.rolling(w).std() * np.sqrt(252)  # Annualized

        # Volatility regime: is current vol above its 60d average?
        df["vol_regime"] = (df["rvol_20d"] > df["rvol_20d"].rolling(60).mean()).astype(
            int
        )

        # Parkinson range-based volatility estimator
        df["parkinson_vol"] = (
            np.log(df["high"] / df["low"]).rolling(20).std() / (2 * np.sqrt(np.log(2)))
        ) * np.sqrt(252)

        return df

    # ─────────────────────────────────────────────────────────────
    # 4. Agronomic / weather features
    # ─────────────────────────────────────────────────────────────

    def _weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "tmax_f" not in df.columns:
            log.warning("No weather data in silver — skipping agronomic features")
            return df

        base = self.f_cfg["gdd_base_temp_f"]
        stress = self.f_cfg["gdd_heat_stress_f"]

        # Growing Degree Days
        avg_temp = (df["tmax_f"] + df["tmin_f"]) / 2
        df["gdd_daily"] = (avg_temp - base).clip(lower=0)

        # Cumulative GDD since April 1 of each year (corn development proxy)
        df["gdd_cumulative"] = df.groupby(df["date"].dt.year)["gdd_daily"].transform(
            lambda s: s.cumsum()
        )

        # Heat stress accumulation
        df["heat_stress_flag"] = (df["tmax_f"] > stress).astype(int)
        for w in [7, 14, 30]:
            df[f"heat_stress_{w}d"] = (
                df["heat_stress_flag"].rolling(w, min_periods=1).sum()
            )

        # Precipitation anomaly (z-score vs trailing 365d)
        if "prcp_in" in df.columns:
            rolling_mean = df["prcp_in"].rolling(365, min_periods=30).mean()
            rolling_std = df["prcp_in"].rolling(365, min_periods=30).std()
            df["prcp_zscore"] = (df["prcp_in"] - rolling_mean) / rolling_std.replace(
                0, np.nan
            )
            for w in [10, 20, 30]:
                df[f"prcp_sum_{w}d"] = df["prcp_in"].rolling(w, min_periods=1).sum()

        # Temperature trend
        df["tmax_7d_avg"] = df["tmax_f"].rolling(7).mean()
        df["tmax_anomaly_30d"] = (
            df["tmax_f"] - df["tmax_f"].rolling(365, min_periods=30).mean()
        )

        return df

    # ─────────────────────────────────────────────────────────────
    # 5. Seasonal / calendar features
    # ─────────────────────────────────────────────────────────────

    def _seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        dt = df["date"]

        # Cyclical encoding — avoids Dec/Jan discontinuity
        df["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
        df["doy_sin"] = np.sin(2 * np.pi * dt.dt.dayofyear / 365)
        df["doy_cos"] = np.cos(2 * np.pi * dt.dt.dayofyear / 365)
        df["week_sin"] = np.sin(2 * np.pi * dt.dt.isocalendar().week.astype(int) / 52)
        df["week_cos"] = np.cos(2 * np.pi * dt.dt.isocalendar().week.astype(int) / 52)

        # Corn crop calendar flags (key price-moving periods)
        month = dt.dt.month
        df["is_planting"] = month.isin([4, 5]).astype(int)  # Apr–May
        df["is_pollination"] = month.isin([7]).astype(int)  # July (critical)
        df["is_harvest"] = month.isin([9, 10, 11]).astype(int)  # Sep–Nov
        df["is_growing_season"] = month.between(4, 10).astype(int)

        # WASDE proximity (World Ag Supply/Demand released ~10th each month)
        df["days_to_wasde"] = (10 - dt.dt.day).abs()
        df["wasde_week"] = (df["days_to_wasde"] <= 3).astype(int)

        # Export inspection day (USDA releases weekly export data on Monday)
        df["is_monday"] = (dt.dt.dayofweek == 0).astype(int)

        return df

    # ─────────────────────────────────────────────────────────────
    # 6. Cross-asset spread features
    # ─────────────────────────────────────────────────────────────

    def _cross_asset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "wheat_close" in df.columns:
            df["corn_wheat_ratio"] = df["close"] / df["wheat_close"].replace(0, np.nan)
            df["corn_wheat_spread"] = df["close"] - df["wheat_close"]

        if "crude_close" in df.columns:
            df["corn_crude_corr_20d"] = (
                df["log_return_1d"]
                .rolling(20)
                .corr(np.log(df["crude_close"] / df["crude_close"].shift(1)))
            )

        if "usd_close" in df.columns:
            df["usd_return_5d"] = df["usd_close"].pct_change(5)
            df["corn_usd_corr_20d"] = (
                df["log_return_1d"]
                .rolling(20)
                .corr(np.log(df["usd_close"] / df["usd_close"].shift(1)))
            )

        return df

    # ─────────────────────────────────────────────────────────────
    # 7. Target construction
    # ─────────────────────────────────────────────────────────────

    def _build_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        N-day forward return as the prediction target.

        The target uses shift(-horizon) which means the last
        `horizon` rows will always be NaN. These are dropped before training
        but kept here to allow the schema to validate proportions.
        """
        h = self.f_cfg["target_horizon_days"]
        future_close = df["close"].shift(-h)
        df["target"] = (future_close / df["close"] - 1) * 100  # as percentage
        df["target_horizon"] = h
        return df

    # ─────────────────────────────────────────────────────────────
    # Orchestration
    # ─────────────────────────────────────────────────────────────

    def build(self, silver_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the full gold feature matrix from the silver dataset.
        Applies all feature groups in sequence, validates schema, and writes.
        """
        log.info("Building gold feature matrix...")
        df = silver_df.copy().sort_values("date").reset_index(drop=True)

        df = self._price_features(df)
        df = self._technical_features(df)
        df = self._volatility_features(df)
        df = self._weather_features(df)
        df = self._seasonal_features(df)
        df = self._cross_asset_features(df)
        df = self._build_target(df)

        # Validate schema contract
        df = validate(df, GoldSchema, layer="gold")

        self._feature_names = self.feature_columns(df)

        path = self.gold_store.write(df, "gold_features")
        log.info(
            f"Gold complete: {len(df):,} rows | "
            f"{len(self._feature_names)} features | {path}"
        )

        return df

    @staticmethod
    def feature_columns(gold_df: pd.DataFrame) -> list[str]:
        """
        Model input columns of a gold feature matrix: numeric columns,
        excluding metadata and target cols. Does not rebuild or write.
        """
        _exclude = {
            "date",
            "target",
            "target_horizon",
            "open",
            "high",
            "low",
            "close",
            "volume",
        }
        return [
            c
            for c in gold_df.columns
            if c not in _exclude
            and gold_df[c].dtype
            in (np.float64, np.float32, np.int64, np.int32, float, int)
        ]

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    def read(self) -> pd.DataFrame:
        return self.gold_store.read("gold_features")
