# Multi-stage Dockerfile for Electricity Demand Forecasting Service
# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir build && \
    pip wheel --no-cache-dir --wheel-dir /app/wheels -e .

# Stage 2: Runtime
FROM python:3.11-slim as runtime

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash appuser

# Copy wheels from builder
COPY --from=builder /app/wheels /app/wheels

# Install application
RUN pip install --no-cache-dir /app/wheels/* && \
    rm -rf /app/wheels

# Copy source code
COPY src/ /app/src/
COPY configs/ /app/configs/

# Create directories for data and models
RUN mkdir -p /app/data /app/models /app/results && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    FORECAST_USE_SYNTHETIC_DATA=true \
    FORECAST_MODELS_DIR=/app/models \
    FORECAST_DATA_DIR=/app/data \
    FORECAST_RESULTS_DIR=/app/results

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "forecast_service.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
