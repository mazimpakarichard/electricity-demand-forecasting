"""Forecasting models."""

from forecast_service.models.base import BaseForecaster
from forecast_service.models.baselines import SARIMAForecaster, SeasonalNaive
from forecast_service.models.lightgbm_model import LightGBMForecaster

__all__ = [
    "BaseForecaster",
    "SeasonalNaive",
    "SARIMAForecaster",
    "LightGBMForecaster",
]

# Optional PyTorch models (lazy import)
try:
    from forecast_service.models.pytorch_model import LSTMForecaster, TCNForecaster

    __all__.extend(["LSTMForecaster", "TCNForecaster"])
except ImportError:
    LSTMForecaster = None  # type: ignore[misc, assignment]
    TCNForecaster = None  # type: ignore[misc, assignment]
