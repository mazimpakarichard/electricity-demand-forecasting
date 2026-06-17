"""Pytest configuration and fixtures."""

from pathlib import Path

import pandas as pd
import pytest

from forecast_service.data.synthetic import SyntheticDataGenerator
from forecast_service.features.engineering import FeatureEngineer


@pytest.fixture(scope="session")
def tmp_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a temporary data directory."""
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="session")
def tmp_models_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a temporary models directory."""
    return tmp_path_factory.mktemp("models")


@pytest.fixture(scope="session")
def synthetic_data() -> pd.DataFrame:
    """Generate synthetic data for testing."""
    generator = SyntheticDataGenerator(seed=42)
    return generator.generate(
        start_date="2017-01-01",
        end_date="2017-03-31",
        include_temperature=True,
    )


@pytest.fixture(scope="session")
def engineered_features(synthetic_data: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features from synthetic data."""
    engineer = FeatureEngineer()
    return engineer.transform(synthetic_data, target_col="load_mw")


@pytest.fixture
def small_synthetic_data() -> pd.DataFrame:
    """Generate small synthetic dataset for quick tests."""
    generator = SyntheticDataGenerator(seed=123)
    return generator.generate(
        start_date="2017-01-01",
        end_date="2017-01-14",  # 2 weeks
        include_temperature=True,
    )


@pytest.fixture
def feature_engineer() -> FeatureEngineer:
    """Create a FeatureEngineer instance."""
    return FeatureEngineer()
