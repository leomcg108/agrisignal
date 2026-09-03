# AgriSignal

> **End-to-end machine learning pipeline for corn futures price prediction**, combining NOAA Corn Belt weather data with futures market signals in a production-grade medallion data architecture.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-189C7E?style=flat-square)](https://xgboost.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Prefect](https://img.shields.io/badge/Prefect-2.x-024DFD?style=flat-square&logo=prefect&logoColor=white)](https://prefect.io)
[![MLflow](https://img.shields.io/badge/MLflow-2.9+-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Prometheus](https://img.shields.io/badge/Prometheus-Monitored-E6522C?style=flat-square&logo=prometheus&logoColor=white)](https://prometheus.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [API Reference](#api-reference)
- [Monitoring & Observability](#monitoring--observability)
- [Testing](#testing)
- [Configuration](#configuration)
- [Docker Deployment](#docker-deployment)
- [Design Decisions](#design-decisions)
- [Roadmap](#roadmap)

---

## Overview

AgriSignal is a **production-grade ML data pipeline** that predicts short-term corn futures returns by combining:

- **NOAA GHCN-Daily** weather data from 5 Corn Belt stations (Iowa, Illinois, Indiana, Nebraska, Ohio)
- **Corn futures price data** (ZC=F) and correlated assets (wheat, crude oil, USD index) via yfinance
- **82 engineered features** spanning technical indicators, agronomic variables, and seasonal patterns

The system is designed as a data engineering showcase demonstrating medallion architecture, schema contracts, walk-forward cross-validation, and production observability rather than black-box model complexity. A simple, well-engineered XGBoost pipeline with a clear data lineage story is more valuable and interpretable than a complex deep learning model without engineering rigour.

### Why Corn Futures?

Corn is the world's most-produced grain crop and a benchmark commodity for global food markets. Its price is driven by a measurable, predictable factor set:

- **Weather** during the growing season (April–November) in the US Corn Belt
- **Technical momentum** in futures markets
- **Seasonal patterns** tied to planting, pollination, and harvest cycles
- **Cross-asset signals** from crude oil (ethanol demand) and the USD (export competitiveness)

This domain specificity makes it an ideal candidate for feature-rich, interpretable ML: the features have clear real-world meaning, which makes SHAP explanations directly actionable.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  DATA SOURCES                                                    │
│         NOAA GHCN Weather API  │  Yahoo Finance (ZC=F)           │
└───────────┬─────────────────────────┬────────────────────┬───────┘
            │                         │                    │
            ▼                         ▼                    ▼
┌──────────────────────────────────────────────────────────────────┐
│  BRONZE LAYER  (raw, append-only, partitioned by ingest_date)    │
│         data/bronze/weather/ │ data/bronze/futures/              │
└──────────────────────────────┬───────────────────────────────────┘
                               │  Schema validation (Pandera)
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  SILVER LAYER  (cleaned, typed, deduplicated, joined)            │
│                data/silver/silver.parquet                        │
└──────────────────────────────┬───────────────────────────────────┘
                               │  
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  GOLD LAYER  (feature matrix, model-ready)                       │
│                 data/gold/gold_features.parquet                  │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
            XGBoost Training       FastAPI Serving
            (MLflow tracked)       /predict endpoint

```

### Medallion Lakehouse

Data is stored in Hive-partitioned Parquet files with a strict three-layer progression:

| Layer | Location | Description | Format |
|-------|----------|-------------|--------|
| **Bronze** | `data/bronze/` | Raw API responses with audit columns (`_ingest_date`, `_ingest_ts`) | Parquet (Snappy), partitioned by `ingest_date=` |
| **Silver** | `data/silver/` | Cleaned, joined, unit-converted, Pandera-validated | Single Parquet file |
| **Gold** | `data/gold/` | ML-ready feature matrix (82 features + target) | Single Parquet file |

---

## Key Features

### Data Engineering
- **Idempotent ingestion** — checks existing partitions before API calls; never double-ingests
- **Hive-style partitioning** — `source/station/ingest_date=YYYY-MM-DD/data.parquet`
- **6-month NOAA fetch windows** — respects API 1,000-row limit (long-format data: 182 days × 5 attributes = ~910 rows)
- **Retry with exponential backoff** — handles transient API failures gracefully
- **Rate limiting** — 0.25s sleep between NOAA requests (stays within 5 req/s limit)
- **Schema contracts** — Pandera validates every layer; `DataContractError` on violation

### Feature Engineering (82 Features)
- **Price/Return**: SMA and EMA (5/10/20/60d), returns at multiple horizons, volume ratio, overnight gap
- **Technical Indicators**: RSI(14), MACD(12/26/9), Bollinger Bands (%B + bandwidth), ATR(14), OBV
- **Volatility Regime**: Realised volatility (5/10/20/60d, annualised ×√252), Parkinson volatility, binary vol regime flag
- **Agronomic**: Growing Degree Days (base 50°F), cumulative GDD (from April 1), heat stress flag (>95°F), precipitation z-score and rolling sums
- **Seasonal**: Cyclical month/DOY encoding (sin+cos pairs), planting/pollination/harvest flags, days-to-WASDE, WASDE week indicator
- **Cross-Asset**: Corn/wheat ratio and spread, 20-day rolling corn/crude correlation

### ML Integrity
- **No lookahead bias** — all rolling features use `.shift(1)` before windowing; tested with correlation check against future returns
- **Walk-forward CV** — expanding window, 5-day gap between train and validation (prevents target leakage for 5-day forward return)
- **Chronological train/test split** — strict 85/15 split; scaler fitted on training data only
- **RobustScaler** — median/IQR normalisation handles commodity price shocks better than StandardScaler

### Production Observability
- **Prometheus metrics** — pipeline row counts, feature counts, quality pass/fail, API latency histograms, prediction value distribution
- **PSI drift detection** — Population Stability Index monitored on RSI, volatility, GDD, and close price
- **Quality gate** — 6 automated checks gate model training; hard failures (row count, leakage, date gaps) stop training; soft warnings logged
- **MLflow experiment tracking** — all runs logged with parameters, metrics, and artifacts

---

## Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Data Ingestion** | `requests`, `yfinance` | NOAA API, futures data |
| **Data Processing** | `pandas`, `numpy` | Transforms and feature engineering |
| **Storage** | `Parquet` (snappy), Hive partitioning | Medallion lakehouse |
| **Schema Validation** | `pandera` | Data contracts at every layer |
| **Machine Learning** | `xgboost`, `scikit-learn` | Model training and scaling |
| **Explainability** | `shap` | Feature importance (TreeExplainer) |
| **Experiment Tracking** | `mlflow` | Runs, metrics, artifact logging |
| **Orchestration** | `prefect` 2.x | DAG, scheduling, retries, parallelism |
| **Serving** | `fastapi`, `uvicorn` | REST API for predictions |
| **Monitoring** | `prometheus-client` | Metrics export and scraping |
| **Testing** | `pytest` | Unit tests, fixtures, coverage |
| **Containerisation** | `Docker`, `docker-compose` | Reproducible deployment |
| **CI/CD** | GitHub Actions | Automated test runs on PR |

---

## Project Structure

```
agrisignal/
│
├── agrisignal/                    # Main package
│   ├── ingestion/
│   │   ├── weather.py             # NOAA GHCN-Daily Bronze ingestion
│   │   └── futures.py             # yfinance Bronze ingestion
│   │
│   ├── transforms/
│   │   ├── silver.py              # Clean, join, aggregate to Silver
│   │   └── schemas.py             # Pandera schema contracts
│   │
│   ├── features/
│   │   └── engineer.py            # 82-feature Gold layer
│   │
│   ├── training/
│   │   └── train_xgboost.py       # WalkForwardCV + XGBoostTrainer
│   │
│   ├── serving/
│   │   └── api.py                 # FastAPI inference endpoint
│   │
│   ├── monitoring/
│   │   ├── metrics.py             # Prometheus metric definitions
│   │   └── data_quality.py        # Quality checks + PSI drift
│   │
│   ├── orchestration/
│   │   └── flows/
│   │       └── daily_pipeline.py  # Prefect flow (DAG)
│   │
│   └── utils.py                   # Config loader, logger, retry, ParquetStore
│
├── tests/
│   └── test_pipeline.py           # 23 tests across 5 categories
│
├── configs/
│   ├── config.yaml                # Runtime config (gitignored)
│   └── config.example.yaml        # Template (committed)
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI
│
├── setup.py
├── requirements.txt
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- A free [NOAA API token](https://www.ncdc.noaa.gov/cdo-web/token)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/agrisignal.git
cd agrisignal

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install in editable mode
pip install -e .
```

### Configuration

```bash
# Copy the example config
cp configs/config.example.yaml configs/config.yaml

# Copy the example environment file
cp .env.example .env.agrisignal

# Add your NOAA API token to .env.agrisignal
NOAA_API_TOKEN=your_token_here
```

### Run the Pipeline

```bash
# Full pipeline run with forced training
python -m agrisignal.orchestration.flows.daily_pipeline --force-train

# Force re-ingestion of all historical data + training
python -m agrisignal.orchestration.flows.daily_pipeline --force-train --force-refresh

# Skip ingestion (use existing Bronze data), force training
python -m agrisignal.orchestration.flows.daily_pipeline --force-train --skip-ingestion
```

### Start the API

```bash
uvicorn agrisignal.serving.api:app --host 0.0.0.0 --port 8000 --reload

# Open interactive API docs
open http://localhost:8000/docs
```

### Make a Prediction

```bash
curl -X POST http://localhost:8000/predict

# Response:
# {
#   "as_of_date": "2024-03-04",
#   "predicted_return_pct": 1.2345,
#   "direction": "BULLISH",
#   "horizon_days": 5,
#   "feature_snapshot": {
#     "rsi": 46.5,
#     "rvol_20d": 0.284,
#     "gdd_cumulative": 1250.0,
#     "heat_stress_14d": 3.0,
#     "macd": 0.023,
#     "bb_pct_b": 0.651
#   },
#   "inference_ms": 42.35
# }
```

---

## Pipeline Walkthrough

### Daily Execution (7:00 AM UTC, Weekdays)

```
07:00:00  Prefect triggers daily_pipeline()
07:00:01  ┌─ ingest_weather() ──── async ──────────────────────────────┐
          └─ ingest_futures() ──── async ──────────────────────────────┘
          |  5 Corn Belt stations, 6-month NOAA windows, rate-limited   |
07:00:15  Both ingestion tasks complete
07:00:16  build_silver()
          |  Pivot long→wide, unit conversions (tenths→°F/inches)       |
          |  Aggregate 5 stations → Corn Belt average                   |
          |  merge_asof(futures, weather, tolerance=3d)                 |
          |  Pandera SilverSchema validation                            |
07:00:28  build_gold()
          |  82 features: technical, agronomic, seasonal, cross-asset   |
          |  Cyclical encoding, GDD computation, WASDE calendar         |
          |  Target: 5-day forward return (last 5 rows = NaN)           |
          |  Pandera GoldSchema validation                              |
07:00:42  data_quality_check()  ← Quality gate
          |  Row count ✓  Leakage ✓  Date continuity ✓                 |
          |  PSI drift ⚠  Feature nulls ⚠   → passes                   |
07:00:44  train_model()  [Monday only, or --force-train]
          |  Walk-forward CV (5 folds, gap=5)                          |
          |  Final model on full train set + early stopping             |
          |  SHAP TreeExplainer → shap_importance.csv                  |
          |  Artifacts saved → MLflow logged                           |
07:01:55  emit_metrics()  → Prometheus updated
07:01:57  Pipeline complete  (total: ~2 minutes)
```

### Ingestion Design: 6-Month NOAA Windows

NOAA's API returns data in **long format** (one row per attribute per day):

```
182 days × 5 attributes (TMAX, TMIN, PRCP, SNOW, SNWD) = ~910 rows
```

This safely stays under the **1,000-row API limit**. Fetching by calendar year would return ~1,825 rows and get silently truncated. Each 6-month period is stored as a separate Hive partition:

```
data/bronze/weather/USW00014933/ingest_date=2024-01-01/data.parquet  ← H1 2024
data/bronze/weather/USW00014933/ingest_date=2024-07-01/data.parquet  ← H2 2024
```

### Walk-Forward Cross-Validation

```
|─────────── Training Data ──────────────|
|                                         |
Fold 1:  [■■■■■■■■■■■■]──gap──[■■■■■■]
Fold 2:  [■■■■■■■■■■■■■■■■■■■]──gap──[■■■■■■]
Fold 3:  [■■■■■■■■■■■■■■■■■■■■■■■■■■]──gap──[■■■■■■]
Fold 4:  [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■]──gap──[■■■■■■]
Fold 5:  [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■]──gap──[■■■■■■]
                                                              ↑
                                              Held-out test set (15%)
```

The **5-day gap** prevents target leakage: since the target is a 5-day forward return, days immediately after training end are used in the target calculation and must not appear in validation.

---

## API Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | Generate corn futures prediction |
| `GET` | `/health` | API and model health status |
| `GET` | `/model/metadata` | Feature list and training metrics |
| `GET` | `/features/latest` | Latest Gold layer feature values |
| `GET` | `/metrics` | Prometheus metrics scrape endpoint |
| `GET` | `/docs` | Interactive Swagger UI |

### `POST /predict`

**Request body** (optional):
```json
{
  "horizon_days": 5
}
```

**Response**:
```json
{
  "as_of_date": "2024-03-04",
  "predicted_return_pct": 1.2345,
  "direction": "BULLISH",
  "horizon_days": 5,
  "feature_snapshot": {
    "rsi": 46.5,
    "rvol_20d": 0.284,
    "gdd_cumulative": 1250.0,
    "heat_stress_14d": 3.0,
    "macd": 0.023,
    "bb_pct_b": 0.651
  },
  "inference_ms": 42.35
}
```

**Direction classification**:
- `BULLISH`: predicted return > +0.1%
- `BEARISH`: predicted return < -0.1%
- `NEUTRAL`: predicted return between -0.1% and +0.1%

### `GET /health`

```json
{
  "status": "healthy",
  "model_loaded": true,
  "uptime_s": 3847.1,
  "model_metadata": {
    "n_features": 82,
    "test_metrics": {"da": 0.682, "mae": 1.245},
    "target_horizon_days": 5
  }
}
```

---

## Monitoring & Observability

### Prometheus Metrics

All metrics are exposed at `GET /metrics` and scraped by Prometheus every 15 seconds.

| Metric | Type | Description |
|--------|------|-------------|
| `agrisignal_pipeline_silver_rows_total` | Gauge | Rows in Silver dataset after last run |
| `agrisignal_pipeline_gold_features_total` | Gauge | Features in Gold feature matrix |
| `agrisignal_data_quality_passed` | Gauge | 1 if quality checks passed, 0 otherwise |
| `agrisignal_prediction_requests_total{status}` | Counter | Total API requests by status |
| `agrisignal_prediction_latency_seconds` | Histogram | Request latency (buckets: 5ms–1s) |
| `agrisignal_model_prediction_return_pct` | Histogram | Distribution of predicted returns |

### Data Quality Checks (Pipeline Gate)

Six checks run on every pipeline execution. Hard failures block model training:

| Check | Type | Threshold | Action |
|-------|------|-----------|--------|
| Row count | Hard | ≥ 80% of expected trading days | Block training |
| Target null rate | Hard | ≤ 3× expected (horizon/n) | Block training |
| Leakage detection | Hard | `abs(corr(close, target)) < 0.9` | Block training |
| Date continuity | Hard | Max gap ≤ 5 days | Block training |
| Feature null rate | Soft | No feature > 10% nulls | Warning only |
| PSI distribution drift | Soft | PSI < 0.20 on RSI, vol, GDD, close | Warning only |

### Starting the Full Observability Stack

```bash
# Start Prometheus + Grafana
docker-compose --profile monitoring up -d

# Access:
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (admin/admin)
```

---

## Testing

### Run Tests

```bash
# All tests with verbose output
pytest tests/ -v

# Specific category
pytest tests/test_pipeline.py::TestFeatureEngineering -v

# With coverage report
pytest tests/ --cov=agrisignal --cov-report=html
open htmlcov/index.html
```

### Test Suite (23 Tests, 5 Categories)

```
TestSchemas (5 tests)
  ✓ Valid silver data passes Pandera schema
  ✓ Negative close prices raise DataContractError
  ✓ High < Low raises DataContractError
  ✓ Duplicate dates raise DataContractError
  ✓ RSI is bounded in [0, 100]

TestTransforms (3 tests)
  ✓ Row count preserved through Silver transform
  ✓ 1-day return computed correctly
  ✓ Log return sign matches simple return sign

TestFeatureEngineering (7 tests)
  ✓ Gold layer has ≥ 40 features
  ✓ No lookahead bias (feature correlation with future return < 0.95)
  ✓ Last N rows of target are NaN (no future leakage)
  ✓ Growing Degree Days are non-negative
  ✓ Cyclical features (sin/cos) bounded in [-1, 1]
  ✓ Feature matrix is sorted chronologically
  ✓ Feature build is idempotent (same input → same output)

TestWalkForwardCV (4 tests)
  ✓ Train and validation indices never overlap
  ✓ Validation always strictly after training
  ✓ Gap between train end and val start is respected
  ✓ Each fold has strictly more training data (expanding window)

TestDataQuality (4 tests)
  ✓ Good synthetic data passes all quality checks
  ✓ Empty DataFrame fails row count check
  ✓ PSI ≈ 0 for identical distributions
  ✓ PSI > 0.2 for distributions shifted by 5σ
```

The **no-lookahead-bias test** is the most critical: it checks that no feature has correlation > 0.95 with the next-day return. If any feature accidentally uses future data, this test catches it immediately.

---

## Configuration

`configs/config.yaml` controls the full pipeline. Key settings:

```yaml
pipeline:
  name: "agrisignal"
  commodity: "corn"
  lookback_years: 10          # Years of historical data to ingest

sources:
  weather:
    stations:                  # NOAA Corn Belt airport stations
      - "USW00014933"          # Iowa
      - "USW00094846"          # Illinois
      - "USW00093819"          # Indiana
      - "USW00014942"          # Nebraska
      - "USW00014820"          # Ohio
    datatypes: ["TMAX", "TMIN", "PRCP", "SNOW", "SNWD"]
    rate_limit_sleep: 0.25     # Seconds between requests
  
  futures:
    primary: "ZC=F"            # Corn continuous front-month
    correlated:
      wheat: "ZW=F"
      crude: "CL=F"
      usd: "DX-Y.NYB"
    min_daily_volume: 5000

features:
  target_horizon_days: 5       # Predict N-day forward return
  gdd_base_temp_f: 50          # Corn biological zero
  heat_stress_threshold_f: 95  # Pollination damage threshold

model:
  xgboost:
    n_estimators: 600
    max_depth: 5
    learning_rate: 0.04
    subsample: 0.8
    colsample_bytree: 0.75
    reg_alpha: 0.1
    reg_lambda: 1.0

mlflow:
  tracking_uri: "sqlite:///mlflow.db"
  experiment_name: "corn-futures-xgboost"

api:
  host: "0.0.0.0"
  port: 8000
  workers: 2

monitoring:
  psi_critical: 0.20           # PSI threshold for distribution drift
  min_row_fraction: 0.80       # Minimum fraction of expected trading days
```

---

## Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# Start with monitoring stack (Prometheus + Grafana)
docker-compose --profile monitoring up -d

# Services:
# API:         http://localhost:8000
# MLflow:      http://localhost:5000
# Prometheus:  http://localhost:9090
# Grafana:     http://localhost:3000
```

### `docker-compose.yml` Services

| Service | Port | Description |
|---------|------|-------------|
| `api` | 8000 | FastAPI prediction server |
| `mlflow` | 5000 | MLflow tracking UI |
| `prometheus` | 9090 | Metrics scraping and storage |
| `grafana` | 3000 | Metrics visualization dashboards |

---

## Design Decisions

### Why XGBoost over Deep Learning?

Tabular financial data with ~80 features and ~2,500 training samples is the **canonical XGBoost use case**:

- Tree models handle heterogeneous feature scales (GDD in thousands vs. RSI in 0-100) with minimal preprocessing
- SHAP values give per-prediction explanations in terms of domain-meaningful features
- Training takes ~70 seconds (vs. hours for LSTMs on this dataset size)
- No risk of overfitting to spurious temporal patterns that transformers can pick up
- Directional Accuracy of 68% is competitive with professional quant benchmarks

### Why Medallion Architecture over a Database?

Parquet with Hive partitioning gives columnar, compressed, analytics-optimised storage that integrates seamlessly with the broader data ecosystem (Spark, Dask, DuckDB, Arrow). The Bronze→Silver→Gold separation enforces a strict data lineage: raw data is always preserved, transformations are reproducible, and schema contracts enforce data quality at each boundary. At current scale (~5MB Gold), Parquet loads in milliseconds.

### Why In-Memory Gold Layer in the API?

The API loads the entire Gold DataFrame at startup (~5MB, ~2,500 rows). This gives:
- Sub-50ms prediction latency (no disk I/O per request)
- Zero database infrastructure required
- A simple, stateless service design

At 100× scale (500MB), this design would need replacing with DuckDB or Postgres, but the modular architecture makes this swap straightforward.

### Why Walk-Forward CV over K-Fold?

Financial time series violate the i.i.d. assumption that K-fold relies on. Training on future data to predict the past (temporal leakage) produces models that look excellent in evaluation but fail in production. Walk-forward CV with an expanding window and a target-horizon gap faithfully simulates the real deployment scenario: always training on the past, predicting the future.

---

## Roadmap

### Phase 2 Enhancements

- [ ] **USDA NASS integration** — Weekly crop progress reports (Good+Excellent %) and WASDE monthly report as features
- [ ] **Rolling model retraining** — Automatic Monday retraining with model registry versioning in MLflow
- [ ] **Alerting** — Prometheus Alert Manager rules for quality failures, row count drops, and API error rate
- [ ] **Shadow mode** — Log predictions without serving them; compare model versions offline before promotion
- [ ] **Confidence intervals** — XGBoost quantile regression for prediction intervals
- [ ] **Authentication** — API key middleware for production serving

### Phase 3 Scalability

- [ ] **DuckDB Silver/Gold** — Replace pandas with DuckDB for 100× dataset scale
- [ ] **Kubernetes deployment** — Helm chart with horizontal pod autoscaling
- [ ] **Feature store** — Decouple feature computation from training for real-time serving
- [ ] **A/B testing framework** — Route traffic between model versions with statistical significance testing

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Author

Built as a data engineering portfolio project demonstrating production-grade ML pipeline design.

If you find this project useful or have feedback, feel free to open an issue or submit a pull request.

---

*AgriSignal is a portfolio project for educational and demonstration purposes. It does not constitute financial advice.*
