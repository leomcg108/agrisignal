"""
ingestion/futures.py  —  BRONZE LAYER
──────────────────────────────────────
Downloads corn futures OHLCV data (ZC=F) and correlated instruments
from Yahoo Finance (free), writing raw bronze partitions.

Yahoo Finance alternatives (paid):
  - Quandl/Nasdaq Data Link (CHRIS/CME_C1) — professional grade
  - CME DataMine — official exchange data
  - Interactive Brokers TWS API — real-time + historical
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from agrisignal.utils import ParquetStore, get_logger, load_config

log = get_logger(__name__)


class FuturesIngester:
    """
    Downloads corn futures OHLCV data.

    Bronze output schema (bronze/futures partition):
      Date    : object  (YYYY-MM-DD)
      Open    : float64
      High    : float64
      Low     : float64
      Close   : float64
      Volume  : float64
      ticker  : str
      label   : str     (corn | wheat | crude | usd)
      _ingest_date : str
      _ingest_ts   : str
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.cfg = load_config(config_path)
        self.src_cfg = self.cfg["sources"]["futures"]
        self.store = ParquetStore(self.cfg["storage"]["bronze"] + "/futures")

    def _download_ticker(
        self,
        ticker: str,
        label: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Download OHLCV for one ticker. Returns empty DataFrame on failure."""
        log.info(f"Downloading {label} ({ticker}) {start} → {end}")
        try:
            df = yf.download(
                ticker,
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),  # end is exclusive in yfinance
                progress=False,
                auto_adjust=True,
            )
        except Exception as exc:
            log.error(f"yfinance download failed for {ticker}: {exc}")
            return pd.DataFrame()

        if df.empty:
            log.warning(f"Empty response for {ticker}")
            return pd.DataFrame()

        # Flatten MultiIndex columns if present (yfinance quirk)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        df = df.reset_index()
        df.columns = [str(c) for c in df.columns]
        df["ticker"] = ticker
        df["label"] = label
        return df

    def ingest(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        force_refresh: bool = False,
    ) -> list[Path]:
        """
        Download and store all configured futures instruments.
        Partitioned by label (corn, wheat, crude, usd).
        """
        log.info("Starting futures ingestion")
        lookback = self.cfg["pipeline"]["lookback_years"]
        today = date.today()

        if end_date is None:
            end_date = today
        if start_date is None:
            start_date = date(today.year - lookback, 1, 1)

        log.info(f"Using lookback years={lookback}")

        written: list[Path] = []
        tickers = {
            "corn": self.src_cfg["ticker"],
            **self.src_cfg.get("correlated", {}),
        }

        for label, ticker in tickers.items():
            # Idempotency check (skip if we already have recent data)
            existing = self.store.list_bronze_partitions(label)
            if existing and not force_refresh:
                latest = max(existing)
                days_stale = (today - date.fromisoformat(latest)).days
                if days_stale <= 1:
                    log.info(f"Skipping {label} (partition {latest} is current)")
                    continue
                log.info(f"Refreshing {label} (latest partition: {latest}, {days_stale}d stale)")

            df = self._download_ticker(ticker, label, start_date, end_date)
            if df.empty:
                continue

            # Validate minimum required columns are present
            required = {"Date", "Open", "High", "Low", "Close", "Volume"}
            missing = required - set(df.columns)
            if missing:
                log.error(f"Missing columns for {ticker}: {missing}")
                continue

            path = self.store.write_bronze_partition(
                df,
                source=label,
                ingest_date=today.isoformat(),
            )
            written.append(path)

        log.info(f"Futures ingestion complete: {len(written)} instruments")
        return written

    def read_corn(self) -> pd.DataFrame:
        """Read all bronze corn futures partitions."""
        return self.store.read_bronze("corn")

    def read_all(self) -> dict[str, pd.DataFrame]:
        """Read all futures instruments from bronze."""
        result = {}
        for label in ["corn", "wheat", "crude", "usd"]:
            try:
                result[label] = self.store.read_bronze(label)
            except FileNotFoundError:
                log.debug(f"No bronze data for {label}")
        return result
