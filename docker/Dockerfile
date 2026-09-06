# ═══════════════════════════════════════════════════════════════
# Stage 1: dependency builder
# ═══════════════════════════════════════════════════════════════
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Install the agrisignal package itself (without re-installing dependencies)
COPY setup.py .
COPY agrisignal/ agrisignal/
RUN pip install --no-cache-dir --prefix=/install --no-deps .


# ═══════════════════════════════════════════════════════════════
# Stage 2: lean runtime image
# ═══════════════════════════════════════════════════════════════
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="AgriSignal"
LABEL org.opencontainers.image.description="Corn Futures ML Pipeline API"
LABEL org.opencontainers.image.version="1.0.0"

WORKDIR /app

# Runtime-only system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# Create non-root user (security best practice)
RUN useradd -m -u 1000 -s /bin/bash agrisignal \
    && mkdir -p data/bronze data/silver data/gold data/models \
    && chown -R agrisignal:agrisignal /app

USER agrisignal

# Verify app is importable
RUN python -c "from agrisignal.utils import load_config; print('Import OK')"

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Default: run the API server
CMD ["python", "-m", "uvicorn", "agrisignal.serving.api:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
