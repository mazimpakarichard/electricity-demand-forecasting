"""
Forecast Service - Production-grade short-term electricity demand forecasting.

This package provides:
- Data loading and synthetic data generation
- Feature engineering for time series
- Multiple forecasting models (baselines, LightGBM, PyTorch)
- Probabilistic forecasting with quantiles
- FastAPI service for predictions
- MLflow experiment tracking
"""

from forecast_service._version import __version__

__all__ = ["__version__"]
