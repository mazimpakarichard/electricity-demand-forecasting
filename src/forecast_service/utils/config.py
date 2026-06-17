"""Configuration management using Pydantic Settings."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="FORECAST_",
        case_sensitive=False,
    )

    # Application
    app_name: str = "Electricity Demand Forecasting Service"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Data paths
    data_dir: Path = Field(default=Path("data"))
    models_dir: Path = Field(default=Path("models"))
    results_dir: Path = Field(default=Path("results"))

    # Data source
    use_synthetic_data: bool = Field(
        default=False,
        description="Use synthetic data instead of real PJM data",
    )
    pjm_series: str = Field(
        default="PJME",
        description="PJM series to use (AEP, COMED, PJME, etc.)",
    )

    # Model configuration
    default_horizon: int = Field(default=24, ge=1, le=168)
    quantiles: list[float] = Field(default=[0.1, 0.5, 0.9])

    # Training
    train_test_split_date: str = Field(default="2017-01-01")
    n_cv_folds: int = Field(default=5, ge=2)
    random_seed: int = Field(default=42)

    # MLflow
    mlflow_tracking_uri: str = Field(default="sqlite:///mlflow.db")
    mlflow_experiment_name: str = Field(default="electricity-demand-forecast")

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=1)

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Get application settings (cached)."""
    return Settings()
