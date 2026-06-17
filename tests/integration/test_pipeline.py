"""Integration tests for training pipeline."""

from pathlib import Path

import pandas as pd
import pytest

from forecast_service.data.synthetic import SyntheticDataGenerator
from forecast_service.features.engineering import FeatureEngineer
from forecast_service.models.baselines import SeasonalNaive
from forecast_service.models.lightgbm_model import LightGBMForecaster


class TestEndToEndPipeline:
    """End-to-end integration tests."""

    @pytest.fixture
    def setup_data(self) -> pd.DataFrame:
        """Generate data for tests."""
        generator = SyntheticDataGenerator(seed=42)
        return generator.generate(
            start_date="2017-01-01",
            end_date="2017-06-30",
            include_temperature=True,
        )

    def test_data_to_features(self, setup_data: pd.DataFrame) -> None:
        """Test data generation and feature engineering."""
        engineer = FeatureEngineer()
        features = engineer.transform(setup_data, target_col="load_mw")

        assert len(features) > 0
        assert "load_mw" in features.columns
        assert len(engineer.feature_names_) > 50  # Should have many features

    def test_train_seasonal_naive(self, setup_data: pd.DataFrame) -> None:
        """Test training Seasonal Naive model."""
        engineer = FeatureEngineer()
        features = engineer.transform(setup_data, target_col="load_mw")

        feature_cols = [c for c in features.columns if c != "load_mw"]
        X = features[feature_cols]
        y = features["load_mw"]

        model = SeasonalNaive()
        model.fit(X, y)

        assert model.is_fitted

        # Make predictions
        result = model.predict(X.tail(48), horizon=24)
        assert len(result.point_forecast) == 24

    def test_train_lightgbm(self, setup_data: pd.DataFrame) -> None:
        """Test training LightGBM model."""
        engineer = FeatureEngineer()
        features = engineer.transform(setup_data, target_col="load_mw")

        feature_cols = [c for c in features.columns if c != "load_mw"]
        X = features[feature_cols]
        y = features["load_mw"]

        model = LightGBMForecaster(params={"n_estimators": 50, "verbose": -1})
        model.fit(X, y)

        assert model.is_fitted

        # Make predictions
        result = model.predict(X.tail(48), horizon=24)
        assert len(result.point_forecast) == 24

        # Check quantiles
        assert 0.1 in result.quantile_forecasts
        assert 0.5 in result.quantile_forecasts
        assert 0.9 in result.quantile_forecasts

    def test_model_persistence(self, setup_data: pd.DataFrame, tmp_path: Path) -> None:
        """Test model save and load cycle."""
        engineer = FeatureEngineer()
        features = engineer.transform(setup_data, target_col="load_mw")

        feature_cols = [c for c in features.columns if c != "load_mw"]
        X = features[feature_cols]
        y = features["load_mw"]

        # Train model
        model = LightGBMForecaster(params={"n_estimators": 20, "verbose": -1})
        model.fit(X, y)

        # Save model
        model_path = tmp_path / "model.pkl"
        model.save(model_path)

        # Load model
        loaded_model = LightGBMForecaster()
        loaded_model.load(model_path)

        # Predictions should match
        X_test = X.tail(24)
        orig_pred = model.predict(X_test, horizon=24)
        loaded_pred = loaded_model.predict(X_test, horizon=24)

        # Point forecasts should be identical
        assert (orig_pred.point_forecast == loaded_pred.point_forecast).all()
