"""
agrisignal/utils.py
────────────────────
Shared infrastructure utilities used across all pipeline layers.

Covers:
  - Config loading (YAML + env var overrides)
  - Structured logging setup
  - Parquet storage helpers with partition management
  - Retry decorator for external API calls
"""

from __future__ import annotations

import functools
import logging
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml


# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────

_CFG_CACHE: dict = {}

def load_config(path: str | None = None) -> dict:
    """
    Load YAML config from multiple possible locations.
    
    Search order:
    1. Explicit path argument
    2. AGRISIGNAL_CONFIG environment variable
    3. ./configs/config.yaml (current directory)
    4. ../configs/config.yaml (parent directory)
    5. {project_root}/configs/config.yaml (package location)
    """
    global _CFG_CACHE
    
    if path is None:
        # Explicit path provided
        candidates = [Path(path)]
    else:
        # Build list of candidate locations
        candidates = []

        # Check environment variable
        # env_path = os.getenv("AGRISIGNAL_CONFIG")
        # if env_path:
        #     candidates.append(Path(env_path))
        
        # Current directory
        candidates.append(Path.cwd() / "agrisignal" / "configs" / "config.yaml")        
        
        # Current directory
        candidates.append(Path.cwd() / "configs" / "config.yaml")
        
        # Parent directory (in case running from agrisignal/ subdirectory)
        candidates.append(Path.cwd().parent / "configs" / "config.yaml")
        
        # Project root (relative to this file)
        utils_dir = Path(__file__).parent
        project_root = utils_dir.parent
        candidates.append(project_root / "configs" / "config.yaml")
        
    # Try each candidate
    for candidate in candidates:
        path_str = str(candidate)
        
        # Check cache first
        if path_str in _CFG_CACHE:
            return _CFG_CACHE[path_str]
        
        # Check if file exists
        if candidate.exists():
            with open(candidate) as f:
                cfg = yaml.safe_load(f)
            _CFG_CACHE[path_str] = cfg
            return cfg
    
    # No config found anywhere
    tried = '\n  '.join(str(c.absolute()) for c in candidates)
    raise FileNotFoundError(
        f"Config file not found. Tried:\n  {tried}\n\n"
        f"Current directory: {Path.cwd()}\n"
        f"Set AGRISIGNAL_CONFIG environment variable or pass explicit path."
    )


def cfg_get(cfg: dict, *keys: str, default: Any = None) -> Any:
    """Safe nested key lookup: cfg_get(cfg, 'model', 'params', 'n_estimators')"""
    val = cfg
    for key in keys:
        if not isinstance(val, dict):
            return default
        val = val.get(key, default)
        if val is default:
            return default
    return val


# ─────────────────────────────────────────────────────────────────
# Structured Logging
# ─────────────────────────────────────────────────────────────────

def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Returns a logger with a consistent, structured format.
    Format: 2024-01-15 07:32:11 | INFO  | agrisignal.ingestion.weather | msg
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


# ─────────────────────────────────────────────────────────────────
# Retry decorator
# ─────────────────────────────────────────────────────────────────

def retry(
    max_attempts: int = 3,
    backoff_base: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Exponential-backoff retry decorator for external API calls.

    Usage:
        @retry(max_attempts=3, exceptions=(requests.RequestException,))
        def fetch_noaa_data(...): ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log = get_logger(func.__module__)
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts:
                        log.error(f"{func.__name__} failed after {max_attempts} attempts: {exc}")
                        raise
                    wait = backoff_base ** attempt
                    log.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed: {exc}. "
                        f"Retrying in {wait:.1f}s..."
                    )
                    time.sleep(wait)
        return wrapper
    return decorator


# ─────────────────────────────────────────────────────────────────
# Parquet Storage Helpers
# ─────────────────────────────────────────────────────────────────

class ParquetStore:
    """
    Manages partitioned Parquet storage for the medallion layers.

    Partitioning strategy:
      Bronze: by ingest_date (YYYY-MM-DD) — append-only, raw source data
      Silver: single file, replaced on each run
      Gold:   single file, replaced on each run

    All writes use snappy compression and preserve schema metadata.
    """

    def __init__(self, base_path: str):
        self.base = Path(base_path)
        self.base.mkdir(parents=True, exist_ok=True)
        self.log = get_logger(f"{__name__}.ParquetStore")

    # ── Bronze (partitioned append) ────────────────────────────────

    def write_bronze_partition(
        self,
        df: pd.DataFrame,
        source: str,
        ingest_date: str,
    ) -> Path:
        """
        Write a bronze partition. Idempotent: re-running the same
        ingest_date simply overwrites the existing partition.

        Layout: base/bronze/<source>/ingest_date=<date>/data.parquet
        """
        partition_dir = self.base / source / f"ingest_date={ingest_date}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        path = partition_dir / "data.parquet"

        df = df.copy()
        df["_ingest_date"] = ingest_date
        df["_ingest_ts"] = pd.Timestamp.utcnow().isoformat()

        df.to_parquet(path, index=False, compression="snappy")
        self.log.info(f"Bronze write: {path} ({len(df):,} rows)")
        return path

    def read_bronze(
        self,
        source: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Read bronze partitions, optionally filtered by date range.
        Scans partition directories and concatenates matching files.
        """
        source_dir = self.base / source
        if not source_dir.exists():
            raise FileNotFoundError(f"Bronze source not found: {source_dir}")

        frames = []
        for partition_dir in sorted(source_dir.iterdir()):
            if not partition_dir.is_dir():
                continue
            part_date = partition_dir.name.replace("ingest_date=", "")
            if start_date and part_date < start_date:
                continue
            if end_date and part_date > end_date:
                continue
            pq_file = partition_dir / "data.parquet"
            if pq_file.exists():
                frames.append(pd.read_parquet(pq_file))

        if not frames:
            raise FileNotFoundError(
                f"No bronze partitions found for {source} between {start_date} and {end_date}"
            )

        combined = pd.concat(frames, ignore_index=True)
        self.log.info(f"Bronze read: {source} → {len(combined):,} rows")
        return combined

    def list_bronze_partitions(self, source: str) -> list[str]:
        """Return sorted list of available partition dates for a source."""
        source_dir = self.base / source
        if not source_dir.exists():
            return []
        return sorted([
            d.name.replace("ingest_date=", "")
            for d in source_dir.iterdir()
            if d.is_dir() and d.name.startswith("ingest_date=")
        ])

    # ── Silver / Gold (single file) ────────────────────────────────

    def write(self, df: pd.DataFrame, name: str) -> Path:
        """Write a silver or gold dataset (single parquet file)."""
        path = self.base / f"{name}.parquet"
        df.to_parquet(path, index=False, compression="snappy")
        self.log.info(f"Write: {path} ({len(df):,} rows, {len(df.columns)} cols)")
        return path

    def read(self, name: str) -> pd.DataFrame:
        """Read a silver or gold dataset."""
        path = self.base / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        df = pd.read_parquet(path)
        self.log.info(f"Read: {path} ({len(df):,} rows)")
        return df

    def exists(self, name: str) -> bool:
        return (self.base / f"{name}.parquet").exists()

    def metadata(self, name: str) -> dict:
        """Return row count, column list, and file size for a dataset."""
        path = self.base / f"{name}.parquet"
        if not path.exists():
            return {}
        df = pd.read_parquet(path)
        return {
            "rows": len(df),
            "columns": list(df.columns),
            "size_mb": round(path.stat().st_size / 1e6, 2),
            "date_range": {
                "min": str(df.get("date", df.iloc[:, 0]).min()),
                "max": str(df.get("date", df.iloc[:, 0]).max()),
            } if len(df) > 0 else {},
        }

