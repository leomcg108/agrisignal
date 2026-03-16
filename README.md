# AgriSignal — Corn Futures ML Pipeline

> A production-grade data engineering portfolio project demonstrating
> end-to-end pipeline design for agricultural commodities price prediction.



| **Medallion architecture** | Bronze → Silver → Gold Parquet layers |

| **Schema contracts** | Pandera validation at every layer boundary |

| **Idempotent ingestion** | Partition-aware, safe to re-run any date range |

| **Orchestration** | Prefect 2 flows with retries, caching, alerting |

| **Containerization** | Multi-stage Docker + docker-compose full stack |

| **Observability** | Structured logging, data quality metrics, Prometheus |

| **Testing** | Unit tests + great_expectations data quality suite |

| **CI/CD** | GitHub Actions: lint → test → docker build |

| **Config-driven** | Zero hardcoded values; YAML + env vars |


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

## Quick Start

```bash
# 1. Clone, configure
cp .env.example .env          # Add your NOAA token
pip install -r requirements.txt

# 2. Run the full pipeline
python -m orchestration.flows.daily_pipeline

# 3. Or run via Docker
docker-compose up --build

# 4. Trigger prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"horizon_days": 5}'
```

## Project Layout

```
agrisignal/
├── ingestion/          # Bronze layer: raw data collectors
│   ├── weather.py      #   NOAA GHCN-Daily API
│   └── futures.py      #   yfinance ZC=F corn futures
├── transforms/         # Silver layer: cleaning & joining
│   ├── schemas.py      #   Pandera data contracts
│   └── silver.py       #   Typed, validated transforms
├── features/           # Gold layer: feature engineering
│   └── engineer.py     #   Growing degree days, technicals, seasonality
├── training/           # Model training
│   └── train_xgboost.py #  Walk-forward CV, MLflow, SHAP
├── serving/            # Real-time inference
│   └── api.py          #   FastAPI prediction endpoint
├── orchestration/      # Pipeline scheduling
│   └── flows/
│       └── daily_pipeline.py  # Prefect 2 flow
├── monitoring/         # Observability
│   ├── data_quality.py #   Row counts, null rates, PSI drift
│   └── metrics.py      #   Prometheus-compatible endpoint
├── tests/              # Test suite
│   ├── test_schemas.py
│   ├── test_transforms.py
│   └── test_features.py
├── configs/
│   └── config.yaml     # All pipeline configuration
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```
