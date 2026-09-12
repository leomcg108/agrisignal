"""
ingestion/weather.py  —  BRONZE LAYER
──────────────────────────────────────
Pulls raw daily weather observations from NOAA's Climate Data Online (CDO) API
for 5 US Corn Belt stations and writes partitioned bronze Parquet files.

  - Idempotent: safe to re-run any date range; re-run overwrites the partition
  - Partitioned storage: bronze/<station>/ingest_date=YYYY-MM-DD/data.parquet
  - Raw fidelity: zero transformation; data written exactly as received
  - Audit metadata: _ingest_ts, _source_url appended to every row
  - Rate-limit aware: respects NOAA's 5 req/sec limit
  - Incremental: only fetches partitions not already present

Free token: https://www.ncdc.noaa.gov/cdo-web/token
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from agrisignal.utils import ParquetStore, get_logger, load_config, retry

log = get_logger(__name__)


class NOAAWeatherIngester:
    """
    Downloads GHCN-Daily weather data for US Corn Belt stations.

    Output schema (bronze/weather partition):
      date         : object  (YYYY-MM-DD)
      station      : str
      datatype     : str     (TMAX | TMIN | PRCP | SNOW)
      value        : float   (raw NOAA units — tenths of °C or mm)
      attributes   : str     (quality flag, source flag, etc.)
      _ingest_date : str     (partition key)
      _ingest_ts   : str     (UTC timestamp of this ingestion run)
    """

    BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"

    def __init__(self, config_path: str | None = None):
        self.cfg = load_config(config_path)
        self.src_cfg = self.cfg["sources"]["weather"]
        self.store = ParquetStore(self.cfg["storage"]["bronze"] + "/weather")
        self.session = self._build_session()

    def _build_session(self) -> requests.Session:

        token = self.src_cfg["Token"]

        if not token:
            raise OSError(
                "NOAA_API_TOKEN not set. "
                "Get a free token at https://www.ncdc.noaa.gov/cdo-web/token"
            )
        session = requests.Session()
        session.headers.update({"token": token})
        return session

    # ── Core fetch ────────────────────────────────────────────────

    @retry(max_attempts=3, exceptions=(requests.RequestException, ValueError))
    def _fetch_period(
        self, station_id: str, start_date: date, end_date: date
    ) -> list[dict]:
        """
        Ingest weather data for a date range, fetching in 6-month chunks.

        NOAA limit: 1000 rows per request
        Reality: 1 year × 365 days × 4 attributes = 1,460 rows (exceeds limit!)
        Solution: Fetch 6-month periods (182 days × 5 attributes = ~730 rows)

        Args:
            station_id: NOAA station ID (e.g., 'USW00014933')
            start_date: Period start date
            end_date: Period end date

        Returns:
            DataFrame with columns: date, station, datatype, value, attributes
            None if request fails or returns no data
        """
        params = {
            "datasetid": self.src_cfg["dataset"],
            "stationid": f"GHCND:{station_id}",
            "startdate": start_date.strftime("%Y-%m-%d"),
            "enddate": end_date.strftime("%Y-%m-%d"),
            "datatypeid": ",".join(self.src_cfg["datatypes"]),
            "limit": 1000,
            "units": "standard",
        }
        resp = self.session.get(
            self.BASE_URL,
            params=params,
            timeout=self.src_cfg["timeout_s"],
        )
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])

        if not results:
            log.debug(f"No results: station={station_id} year={start_date.year}")

        df = pd.DataFrame(results)

        # Verify limit wasn't hit (i.e. truncated data)
        if len(df) >= 1000:
            log.warning(
                f"Hit 1000-row limit for {station_id} {start_date.date()}! "
                f"Data may be truncated. Consider shorter periods."
            )

        return results

    def _generate_6month_periods(
        self, start_date: date, end_date: date
    ) -> list[tuple[date, date]]:
        """
        Generate list of 6-month periods between start and end dates.
        Periods are aligned to calendar half-years:
        - H1: January 1 to June 30
        - H2: July 1 to December 31

        Args:
            start_date: Overall start date
            end_date: Overall end date

        Returns:
            List of (period_start, period_end) tuples

        """

        periods = []

        # Start from the beginning of the half-year containing start_date
        if start_date.month <= 6:
            current_start = date(start_date.year, 1, 1)
        else:
            current_start = date(start_date.year, 7, 1)

        while current_start <= end_date:
            if current_start.month == 1:
                current_end = date(current_start.year, 6, 30)
            else:
                current_end = date(current_start.year, 12, 31)

            periods.append((current_start, current_end))

            current_start = current_end + timedelta(days=1)

        return periods

    # ── Public interface ──────────────────────────────────────────

    def ingest_date_range(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        force_refresh: bool = False,
    ) -> list[Path]:
        """
        Ingest weather for all configured stations over a date range.
        Partitions by station and calendar year.

        Args:
            start_date:     First date to ingest
            end_date:       Last date to ingest (inclusive)
            force_refresh:  If True, re-download even if partition exists

        Returns:
            List of Parquet file paths written
        """
        log.info("Starting weather ingestion")

        written_paths: list[Path] = []
        stations = self.src_cfg["stations"]

        lookback = self.cfg["pipeline"]["lookback_years"]

        today = date.today()

        if end_date is None:
            end_date = today
        if start_date is None:
            start_date = date(today.year - lookback, 1, 1)

        log.info(f"Using lookback years={lookback}")

        periods = self._generate_6month_periods(start_date, end_date)

        log.info(
            f"Generated {len(periods)} six-month periods across {len(stations)} stations"
        )

        for region, station_id in stations.items():
            for period_start, period_end in periods:
                partition_key = period_start.strftime("%Y-%m-%d")

                # Idempotency check — skip if already ingested
                existing = self.store.list_bronze_partitions(region)

                partition_exists = partition_key in existing
                if partition_exists and not force_refresh:
                    log.info(f"Skipping {region}/{partition_key} (already ingested)")
                    continue

                # Fetch from NOAA API
                log.info(
                    f"Fetching {region} ({station_id}) {period_start} to {period_end}"
                )
                records = self._fetch_period(station_id, period_start, period_end)

                if not records:
                    log.warning(f"Empty response: {region} {partition_key}")
                    continue

                df = pd.DataFrame(records)
                df["station_id"] = station_id
                df["region"] = region
                df["year"] = period_start.year

                # Write one partition per station per year
                path = self.store.write_bronze_partition(
                    df,
                    source=region,
                    ingest_date=partition_key,
                )
                written_paths.append(path)

                time.sleep(self.src_cfg["rate_limit_sleep_s"])

        log.info(f"Weather ingestion complete: {len(written_paths)} partitions written")

        return written_paths

    def ingest_incremental(self) -> list[Path]:
        """
        Ingest only the most recent year (handles ongoing pipeline runs).
        Called by the daily Prefect flow.
        """
        today = date.today()
        start = date(today.year, 1, 1)
        return self.ingest_date_range(start, today, force_refresh=True)

    def read_all(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Read all ingested bronze partitions into a single DataFrame.
        """
        frames = []
        for region in self.src_cfg["stations"]:
            try:
                df = self.store.read_bronze(region, start_date, end_date)
                frames.append(df)
            except FileNotFoundError:
                log.warning(f"No bronze data found for region: {region}")

        if not frames:
            raise RuntimeError(
                "No weather data found in bronze layer. "
                "Run ingest_date_range() first."
            )

        combined = pd.concat(frames, ignore_index=True)
        log.info(
            f"Bronze weather: {len(combined):,} rows across {len(frames)} stations"
        )
        return combined
