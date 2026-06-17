"""Pydantic schemas for API request/response validation."""

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Service health status")
    version: str = Field(description="Service version")
    model_loaded: bool = Field(description="Whether a model is loaded")
    model_name: str | None = Field(description="Name of loaded model")
    timestamp: datetime = Field(description="Response timestamp")


class PredictRequest(BaseModel):
    """Forecast prediction request."""

    horizon: Annotated[int, Field(ge=1, le=168, description="Forecast horizon in hours")] = 24
    quantiles: list[float] | None = Field(
        default=None,
        description="Quantiles to forecast (e.g., [0.1, 0.5, 0.9])",
    )
    start_time: datetime | None = Field(
        default=None,
        description="Start time for forecast (defaults to current time)",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "horizon": 24,
            "quantiles": [0.1, 0.5, 0.9],
            "start_time": "2024-01-15T00:00:00",
        }
    }}


class ForecastPoint(BaseModel):
    """Single forecast point with quantiles."""

    timestamp: datetime = Field(description="Forecast timestamp")
    point_forecast: float = Field(description="Point forecast (median/p50)")
    p10: float | None = Field(default=None, description="10th percentile forecast")
    p50: float | None = Field(default=None, description="50th percentile forecast")
    p90: float | None = Field(default=None, description="90th percentile forecast")


class PredictResponse(BaseModel):
    """Forecast prediction response."""

    model_name: str = Field(description="Name of model used for prediction")
    horizon: int = Field(description="Number of hours forecasted")
    quantiles: list[float] = Field(description="Quantiles included in forecast")
    forecasts: list[ForecastPoint] = Field(description="Forecast values")
    generated_at: datetime = Field(description="Timestamp when forecast was generated")
    data_source: str = Field(description="Data source (REAL or SYNTHETIC)")

    model_config = {"json_schema_extra": {
        "example": {
            "model_name": "LightGBM",
            "horizon": 24,
            "quantiles": [0.1, 0.5, 0.9],
            "forecasts": [
                {
                    "timestamp": "2024-01-15T00:00:00",
                    "point_forecast": 32500.0,
                    "p10": 30000.0,
                    "p50": 32500.0,
                    "p90": 35000.0,
                }
            ],
            "generated_at": "2024-01-15T00:00:00",
            "data_source": "SYNTHETIC",
        }
    }}


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(description="Error type")
    detail: str = Field(description="Error details")
    timestamp: datetime = Field(description="Error timestamp")


class ModelInfoResponse(BaseModel):
    """Model information response."""

    name: str = Field(description="Model name")
    quantiles: list[float] = Field(description="Supported quantiles")
    is_fitted: bool = Field(description="Whether model is trained")
    feature_count: int = Field(description="Number of features used")
    features: list[str] | None = Field(default=None, description="Feature names")
