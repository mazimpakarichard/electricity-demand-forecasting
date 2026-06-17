"""FastAPI application for electricity demand forecasting service."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from forecast_service._version import __version__
from forecast_service.api.schemas import (
    ErrorResponse,
    ForecastPoint,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from forecast_service.data.synthetic import SyntheticDataGenerator
from forecast_service.features.engineering import FeatureEngineer
from forecast_service.models.base import BaseForecaster
from forecast_service.models.lightgbm_model import LightGBMForecaster
from forecast_service.utils.config import get_settings
from forecast_service.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


class ForecastService:
    """Service class managing model and prediction logic."""

    def __init__(self) -> None:
        self.model: BaseForecaster | None = None
        self.feature_engineer: FeatureEngineer | None = None
        self.recent_data: pd.DataFrame | None = None
        self.data_source: str = "UNKNOWN"

    def load_model(self, model_path: Path) -> None:
        """Load a trained model from disk."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Determine model type from filename
        model_name = model_path.stem.lower()

        if "lightgbm" in model_name or "lgb" in model_name:
            self.model = LightGBMForecaster()
        else:
            # Default to LightGBM
            self.model = LightGBMForecaster()

        self.model.load(model_path)
        logger.info("Model loaded", path=str(model_path), name=self.model.name)

    def initialize_data(self, use_synthetic: bool = True) -> None:
        """Initialize recent data for predictions."""
        if use_synthetic:
            generator = SyntheticDataGenerator(seed=42)
            # Generate recent data
            end_date = datetime.now(timezone.utc)
            start_date = end_date - pd.Timedelta(days=30)

            self.recent_data = generator.generate(
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                include_temperature=True,
            )
            self.data_source = "SYNTHETIC"
        else:
            self.data_source = "REAL"

        self.feature_engineer = FeatureEngineer()

        logger.info(
            "Data initialized",
            source=self.data_source,
            rows=len(self.recent_data) if self.recent_data is not None else 0,
        )

    def predict(
        self,
        horizon: int = 24,
        quantiles: list[float] | None = None,
        start_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Generate a forecast."""
        if self.model is None:
            raise ValueError("No model loaded")

        if self.recent_data is None or self.feature_engineer is None:
            raise ValueError("Data not initialized")

        # Use model's quantiles if not specified
        if quantiles is None:
            quantiles = self.model.quantiles

        # Generate features
        features = self.feature_engineer.transform(
            self.recent_data, target_col="load_mw", drop_na=True
        )

        # Get last portion for prediction
        feature_cols = [c for c in features.columns if c != "load_mw"]
        X = features[feature_cols].tail(horizon + 168)  # Extra for context

        # Generate predictions
        result = self.model.predict(X, horizon=horizon)

        # Format response
        forecasts = []
        for i in range(len(result.timestamps)):
            point = ForecastPoint(
                timestamp=result.timestamps[i].to_pydatetime(),
                point_forecast=float(result.point_forecast[i]),
                p10=(
                    float(result.quantile_forecasts.get(0.1, [0])[i])
                    if 0.1 in result.quantile_forecasts
                    else None
                ),
                p50=(
                    float(result.quantile_forecasts.get(0.5, [0])[i])
                    if 0.5 in result.quantile_forecasts
                    else None
                ),
                p90=(
                    float(result.quantile_forecasts.get(0.9, [0])[i])
                    if 0.9 in result.quantile_forecasts
                    else None
                ),
            )
            forecasts.append(point)

        return {
            "model_name": self.model.name,
            "horizon": horizon,
            "quantiles": quantiles,
            "forecasts": forecasts,
            "generated_at": datetime.now(timezone.utc),
            "data_source": self.data_source,
        }


# Global service instance
_service: ForecastService | None = None


def get_service() -> ForecastService:
    """Get the global service instance."""
    global _service
    if _service is None:
        _service = ForecastService()
    return _service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    configure_logging()
    logger.info("Starting Forecast Service", version=__version__)

    service = get_service()
    settings = get_settings()

    # Try to load a model
    model_path = settings.models_dir / "lightgbm.pkl"
    if model_path.exists():
        try:
            service.load_model(model_path)
        except Exception as e:
            logger.warning("Failed to load model", error=str(e))

    # Initialize data
    service.initialize_data(use_synthetic=settings.use_synthetic_data)

    yield

    logger.info("Shutting down Forecast Service")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Electricity Demand Forecasting Service",
        description="""
        Production-grade short-term electricity demand forecasting API.

        Features:
        - Probabilistic forecasts with quantiles (p10/p50/p90)
        - Multiple model support (LightGBM, LSTM, baselines)
        - Synthetic data generation for testing
        - MLflow experiment tracking integration
        """,
        version=__version__,
        lifespan=lifespan,
        responses={
            500: {"model": ErrorResponse, "description": "Internal server error"},
        },
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check() -> HealthResponse:
        """
        Health check endpoint.

        Returns service status, version, and model information.
        """
        service = get_service()

        return HealthResponse(
            status="healthy",
            version=__version__,
            model_loaded=service.model is not None,
            model_name=service.model.name if service.model else None,
            timestamp=datetime.now(timezone.utc),
        )

    @app.post("/predict", response_model=PredictResponse, tags=["Forecast"])
    async def predict(request: PredictRequest) -> PredictResponse:
        """
        Generate electricity demand forecast.

        Args:
            request: Forecast request with horizon and quantiles.

        Returns:
            Probabilistic forecast for the requested horizon.
        """
        service = get_service()

        if service.model is None:
            raise HTTPException(
                status_code=503,
                detail="No model loaded. Please train a model first.",
            )

        try:
            result = service.predict(
                horizon=request.horizon,
                quantiles=request.quantiles,
                start_time=request.start_time,
            )
            return PredictResponse(**result)
        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/model", response_model=ModelInfoResponse, tags=["Model"])
    async def get_model_info() -> ModelInfoResponse:
        """
        Get information about the loaded model.

        Returns model name, quantiles, and feature information.
        """
        service = get_service()

        if service.model is None:
            raise HTTPException(
                status_code=404,
                detail="No model loaded",
            )

        feature_names = None
        feature_count = 0

        if hasattr(service.model, "feature_names"):
            feature_names = service.model.feature_names
            feature_count = len(feature_names)

        return ModelInfoResponse(
            name=service.model.name,
            quantiles=service.model.quantiles,
            is_fitted=service.model.is_fitted,
            feature_count=feature_count,
            features=feature_names,
        )

    @app.get("/", tags=["Root"])
    async def root() -> dict[str, str]:
        """Root endpoint with service information."""
        return {
            "service": "Electricity Demand Forecasting",
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
        }

    return app


# Create app instance
app = create_app()
