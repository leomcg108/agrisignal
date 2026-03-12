"""
monitoring/metrics.py
──────────────────────
Prometheus-compatible metrics for pipeline observability.
Exposes a /metrics endpoint in the API for scraping.
"""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# Shared registry
REGISTRY = CollectorRegistry(auto_describe=True)

# Pipeline metrics
pipeline_rows = Gauge(
    "agrisignal_pipeline_silver_rows_total",
    "Number of rows in the silver dataset after last run",
    registry=REGISTRY,
)
gold_features = Gauge(
    "agrisignal_pipeline_gold_features_total",
    "Number of features in the gold feature matrix",
    registry=REGISTRY,
)
quality_check_passed = Gauge(
    "agrisignal_data_quality_passed",
    "1 if latest data quality checks passed, 0 otherwise",
    registry=REGISTRY,
)

# API metrics
prediction_requests = Counter(
    "agrisignal_prediction_requests_total",
    "Total prediction requests served",
    ["status"],
    registry=REGISTRY,
)
prediction_latency = Histogram(
    "agrisignal_prediction_latency_seconds",
    "Prediction request latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=REGISTRY,
)
model_prediction_value = Histogram(
    "agrisignal_model_prediction_return_pct",
    "Distribution of model prediction values (% return)",
    buckets=[-5, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5],
    registry=REGISTRY,
)


def emit_pipeline_metrics(stats: dict) -> None:
    """Update Prometheus gauges after a pipeline run."""
    if "silver_rows" in stats:
        pipeline_rows.set(stats["silver_rows"])
    if "gold_features" in stats:
        gold_features.set(stats["gold_features"])
    if "quality_passed" in stats:
        quality_check_passed.set(stats["quality_passed"])


def get_metrics_output() -> tuple[bytes, str]:
    """Return Prometheus text format metrics for /metrics endpoint."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
