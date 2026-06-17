# System Architecture

## Overview

The Electricity Demand Forecasting Service is designed as a modular, production-ready system for short-term load forecasting. This document describes the architectural decisions, component interactions, and deployment considerations.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Client Layer                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Web Client  │  │ API Client  │  │ CLI Client  │  │ Scheduled Jobs      │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
└─────────┼────────────────┼────────────────┼────────────────────┼────────────┘
          │                │                │                    │
          └────────────────┼────────────────┼────────────────────┘
                           │                │
                           ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           API Gateway / Load Balancer                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FastAPI Service                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                           Endpoints                                     │ │
│  │  ┌─────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────────────┐  │ │
│  │  │ /health │  │  /predict   │  │   /model    │  │      /docs        │  │ │
│  │  └─────────┘  └──────┬──────┘  └─────────────┘  └───────────────────┘  │ │
│  └──────────────────────┼─────────────────────────────────────────────────┘ │
│                         │                                                    │
│  ┌──────────────────────▼─────────────────────────────────────────────────┐ │
│  │                      ForecastService                                    │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────────────┐  │ │
│  │  │ Model Manager   │  │ Feature Engine  │  │   Data Manager         │  │ │
│  │  │ (Load/Predict)  │  │ (Transform)     │  │   (Recent History)     │  │ │
│  │  └────────┬────────┘  └────────┬────────┘  └───────────┬────────────┘  │ │
│  └───────────┼────────────────────┼───────────────────────┼────────────────┘ │
└──────────────┼────────────────────┼───────────────────────┼──────────────────┘
               │                    │                       │
               ▼                    ▼                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              Core Libraries                                   │
│  ┌────────────────┐  ┌─────────────────┐  ┌──────────────────────────────┐   │
│  │    Models      │  │    Features     │  │         Data                 │   │
│  │ ┌────────────┐ │  │ ┌─────────────┐ │  │ ┌────────────┐ ┌───────────┐ │   │
│  │ │ LightGBM   │ │  │ │ Lags        │ │  │ │ PJM Loader │ │ Synthetic │ │   │
│  │ │ LSTM       │ │  │ │ Rolling     │ │  │ └────────────┘ │ Generator │ │   │
│  │ │ TCN        │ │  │ │ Fourier     │ │  │                └───────────┘ │   │
│  │ │ SARIMA     │ │  │ │ Calendar    │ │  │                              │   │
│  │ │ Naive      │ │  │ │ Holiday     │ │  │                              │   │
│  │ └────────────┘ │  │ │ Temperature │ │  │                              │   │
│  └────────────────┘  │ └─────────────┘ │  └──────────────────────────────┘   │
│                      └─────────────────┘                                      │
└──────────────────────────────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           External Services                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐   │
│  │ MLflow Server   │  │ Model Registry  │  │ Monitoring (Prometheus)    │   │
│  │ (Experiments)   │  │ (Artifacts)     │  │ (Optional)                 │   │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. API Layer (`src/forecast_service/api/`)

**FastAPI Application** (`app.py`)
- Handles HTTP requests
- Request validation with Pydantic
- CORS middleware
- Lifespan management (startup/shutdown)
- Error handling

**Schemas** (`schemas.py`)
- `PredictRequest`: Forecast input validation
- `PredictResponse`: Structured forecast output
- `HealthResponse`: Service status
- `ModelInfoResponse`: Model metadata

### 2. Model Layer (`src/forecast_service/models/`)

**Base Classes** (`base.py`)
- `BaseForecaster`: Abstract interface for all models
- `ForecastResult`: Standardized prediction container
- `CrossValidationResult`: CV metrics aggregation

**Implementations**
| Model | File | Use Case |
|-------|------|----------|
| Seasonal Naive | `baselines.py` | Baseline, fast |
| SARIMA | `baselines.py` | Statistical baseline |
| LightGBM | `lightgbm_model.py` | Primary production model |
| LSTM | `pytorch_model.py` | Sequence modeling |
| TCN | `pytorch_model.py` | Alternative to LSTM |

### 3. Feature Engineering (`src/forecast_service/features/`)

**FeatureEngineer** (`engineering.py`)
- Configurable via `FeatureConfig`
- Deterministic transformation
- Handles missing values
- Groups features by type

**Feature Pipeline**
```
Raw Data → Lag Features → Rolling Stats → Calendar → Fourier → Holiday → Temperature → Output
```

### 4. Data Layer (`src/forecast_service/data/`)

**DataLoader** (`loader.py`)
- Loads PJM CSV files
- Validates data quality
- Handles missing values
- Ensures hourly frequency

**SyntheticDataGenerator** (`synthetic.py`)
- Creates realistic test data
- Reproducible with seed
- Includes temperature effects
- Clearly labeled as SYNTHETIC

### 5. Training Pipeline (`src/forecast_service/training/`)

**ExperimentTracker** (`experiment.py`)
- MLflow integration
- Parameter logging
- Metric tracking
- Artifact storage
- Model registry

**TrainingPipeline** (`pipeline.py`)
- End-to-end orchestration
- Cross-validation
- Model comparison
- Result persistence

### 6. Utilities (`src/forecast_service/utils/`)

**Configuration** (`config.py`)
- Pydantic Settings
- Environment variable support
- Validation with defaults

**Logging** (`logging.py`)
- Structured logging with structlog
- JSON format for production
- Colored console for development

**Metrics** (`metrics.py`)
- MAPE, RMSE, MAE
- Pinball loss for quantiles
- Coverage metrics

## Data Flow

### Prediction Flow
```
1. Client POST /predict
2. Validate request (PredictRequest)
3. ForecastService.predict()
   a. Get recent data
   b. Engineer features
   c. Model.predict()
   d. Format response
4. Return PredictResponse
```

### Training Flow
```
1. Load/Generate data
2. Engineer features
3. For each model:
   a. Cross-validate
   b. Log to MLflow
   c. Save model
4. Compare results
5. Save best model
```

## Design Decisions

### 1. Multi-Quantile Approach
**Decision**: Train separate models for each quantile (LightGBM) or multi-output networks (LSTM/TCN).

**Rationale**:
- Provides uncertainty estimates
- Supports risk-aware decision making
- Industry standard for probabilistic forecasting

### 2. Feature-Based vs End-to-End
**Decision**: Explicit feature engineering + tree models as primary.

**Rationale**:
- Interpretable features
- SHAP explainability
- Robust to data shifts
- Faster training

### 3. Synthetic Data Generator
**Decision**: Include synthetic data generation as fallback.

**Rationale**:
- Enable testing without data downloads
- Ensure reproducible tests
- Support CI/CD pipelines

### 4. Rolling-Origin CV
**Decision**: Use rolling-origin instead of random splits.

**Rationale**:
- Respects temporal ordering
- Simulates production deployment
- More realistic error estimates

## Scalability Considerations

### Horizontal Scaling
- Stateless API servers
- Load balancer distribution
- Shared model artifacts (S3/GCS)

### Vertical Scaling
- LightGBM: Linear with features
- LSTM/TCN: GPU acceleration
- Memory: Proportional to history length

### Caching
- Feature engineering caches
- Model inference caching
- Redis for distributed caching

## Security

### API Security
- Input validation
- Rate limiting
- CORS configuration
- No sensitive data in logs

### Model Security
- Model versioning
- Artifact signing (optional)
- Audit logging

## Deployment Options

### Local Development
```bash
uvicorn forecast_service.api.app:app --reload
```

### Docker
```bash
docker-compose up -d
```

### Kubernetes
```yaml
# Example deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: forecast-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: forecast-service:latest
        ports:
        - containerPort: 8000
```

## Monitoring

### Health Checks
- `/health` endpoint
- Docker HEALTHCHECK
- Kubernetes liveness/readiness probes

### Metrics (Prometheus)
- Request latency
- Prediction error distribution
- Model inference time
- Feature engineering time

### Logging
- Structured JSON logs
- Request tracing
- Error aggregation

## Future Enhancements

1. **Online Learning**: Continuous model updates
2. **Ensemble Methods**: Model averaging
3. **Weather Integration**: Real-time weather API
4. **A/B Testing**: Model comparison in production
5. **Feature Store**: Centralized feature management
