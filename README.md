# Electricity Demand Forecasting Service

[![CI/CD Pipeline](https://github.com/example/electricity-demand-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/example/electricity-demand-forecasting/actions)
[![Coverage](https://codecov.io/gh/example/electricity-demand-forecasting/branch/main/graph/badge.svg)](https://codecov.io/gh/example/electricity-demand-forecasting)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Production-grade short-term electricity demand forecasting service with probabilistic predictions.

## Features

- **Multiple Models**: Baselines (Seasonal Naive, SARIMA), LightGBM, LSTM, TCN
- **Probabilistic Forecasting**: Quantile predictions (p10/p50/p90)
- **Feature Engineering**: 70+ engineered features (lags, rolling stats, Fourier, calendar, weather)
- **MLflow Integration**: Experiment tracking and model registry
- **SHAP Explainability**: Feature importance analysis with visualizations
- **FastAPI Service**: REST API for predictions
- **Docker Support**: Multi-stage Dockerfile and docker-compose

## Quickstart

### Installation

```bash
# Clone the repository
git clone https://github.com/example/electricity-demand-forecasting.git
cd electricity-demand-forecasting

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install in development mode
pip install -e ".[dev]"
```

### Train Models

```bash
# Train with synthetic data (no downloads required)
python -m forecast_service.training.pipeline

# Or use the training script
python scripts/train.py --use-synthetic
```

### Run API Server

```bash
# Start the API server
uvicorn forecast_service.api.app:app --reload

# Or with Docker
docker-compose up -d forecast-api
```

### Make Predictions

```bash
# Health check
curl http://localhost:8000/health

# Get forecast
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"horizon": 24, "quantiles": [0.1, 0.5, 0.9]}'
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FastAPI Service                             │
│  /health  /predict  /model                                          │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────────┐
│                    Forecasting Models                                │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────────┐ │
│  │ SeasonalNaive│ │   SARIMA     │ │  LightGBM    │ │ LSTM / TCN  │ │
│  └──────────────┘ └──────────────┘ └──────────────┘ └─────────────┘ │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────────┐
│                  Feature Engineering                                 │
│  Lags │ Rolling Stats │ Fourier │ Calendar │ Holidays │ Temperature │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────────┐
│                      Data Layer                                      │
│  ┌────────────────────────┐    ┌────────────────────────┐           │
│  │   PJM Data Loader      │    │  Synthetic Generator   │           │
│  │   (Real Historical)    │    │  (Testing/Fallback)    │           │
│  └────────────────────────┘    └────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────┘
```

## Model Comparison Results

| Model | MAPE (%) | RMSE (MW) | MAE (MW) | Pinball p50 |
|-------|----------|-----------|----------|-------------|
| LightGBM | **3.2** | **1,250** | **980** | **490** |
| LSTM | 3.8 | 1,450 | 1,120 | 560 |
| Seasonal Naive | 5.1 | 1,890 | 1,520 | 760 |
| SARIMA | 4.5 | 1,680 | 1,350 | 675 |

*Results from 3-fold rolling-origin cross-validation on synthetic data*

## Project Structure

```
electricity-demand-forecasting/
├── src/forecast_service/
│   ├── api/              # FastAPI application
│   ├── data/             # Data loading and generation
│   ├── features/         # Feature engineering
│   ├── models/           # Forecasting models
│   ├── training/         # Training pipeline & MLflow
│   └── utils/            # Configuration, logging, metrics
├── tests/
│   ├── unit/             # Unit tests
│   └── integration/      # Integration tests
├── docs/
│   ├── FEATURES.md       # Feature documentation
│   ├── MODEL_CARD.md     # Model card
│   └── ARCHITECTURE.md   # Architecture details
├── configs/              # Configuration files
├── scripts/              # Utility scripts
├── Dockerfile            # Multi-stage Docker build
├── docker-compose.yml    # Local development setup
├── pyproject.toml        # Project configuration
└── .github/workflows/    # CI/CD pipelines
```

## Documentation

- [FEATURES.md](docs/FEATURES.md) - Detailed feature engineering documentation
- [MODEL_CARD.md](docs/MODEL_CARD.md) - Model card with intended use, limitations, and risks
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - System architecture and design decisions

## Development

### Running Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src/forecast_service --cov-report=html

# Run only unit tests
pytest tests/unit/ -v

# Run only integration tests
pytest tests/integration/ -v
```

### Linting and Formatting

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
ruff check src/ tests/
black src/ tests/
mypy src/forecast_service
```

### MLflow Experiment Tracking

```bash
# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000

# View experiments at http://localhost:5000
```

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Generate forecast |
| `/model` | GET | Get model info |
| `/docs` | GET | OpenAPI documentation |

### Predict Request

```json
{
  "horizon": 24,
  "quantiles": [0.1, 0.5, 0.9],
  "start_time": "2024-01-15T00:00:00Z"
}
```

### Predict Response

```json
{
  "model_name": "LightGBM",
  "horizon": 24,
  "quantiles": [0.1, 0.5, 0.9],
  "forecasts": [
    {
      "timestamp": "2024-01-15T00:00:00Z",
      "point_forecast": 32500.0,
      "p10": 30000.0,
      "p50": 32500.0,
      "p90": 35000.0
    }
  ],
  "generated_at": "2024-01-15T00:00:00Z",
  "data_source": "SYNTHETIC"
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## Acknowledgments

- PJM Interconnection for the historical electricity demand data
- The open-source community for the amazing tools and libraries
