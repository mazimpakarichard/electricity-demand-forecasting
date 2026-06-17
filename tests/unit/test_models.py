"""Tests for forecasting models."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecast_service.models.base import CrossValidationResult, ForecastResult
from forecast_service.models.baselines import SeasonalNaive
from forecast_service.models.lightgbm_model import LightGBMForecaster
from forecast_service.utils.metrics import ForecastMetrics


class TestForecastResult:
    """Tests for ForecastResult dataclass."""

    def test_to_dataframe(self) -> None:
        """Test conversion to DataFrame."""
        result = ForecastResult(
            timestamps=pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 01:00"]),
            point_forecast=np.array([100.0, 110.0]),
            quantile_forecasts={
                0.1: np.array([90.0, 100.0]),
                0.5: np.array([100.0, 110.0]),
                0.9: np.array([110.0, 120.0]),
            },
            model_name="TestModel",
        )

        df = result.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert "forecast_p50" in df.columns
        assert "forecast_p10" in df.columns
        assert "forecast_p90" in df.columns
        assert len(df) == 2


class TestCrossValidationResult:
    """Tests for CrossValidationResult dataclass."""

    def test_mean_std_calculation(self) -> None:
        """Test automatic mean/std calculation."""
        fold_metrics = [
            ForecastMetrics(mape=5.0, rmse=100.0, mae=80.0, pinball_loss={}),
            ForecastMetrics(mape=10.0, rmse=150.0, mae=120.0, pinball_loss={}),
        ]

        result = CrossValidationResult(fold_metrics=fold_metrics)

        assert result.mean_metrics["mape"] == pytest.approx(7.5)
        assert result.mean_metrics["rmse"] == pytest.approx(125.0)

    def test_to_dataframe(self) -> None:
        """Test conversion to DataFrame."""
        fold_metrics = [
            ForecastMetrics(mape=5.0, rmse=100.0, mae=80.0, pinball_loss={}),
            ForecastMetrics(mape=10.0, rmse=150.0, mae=120.0, pinball_loss={}),
        ]

        result = CrossValidationResult(fold_metrics=fold_metrics)
        df = result.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # 2 folds + 1 mean row
        assert "fold" in df.columns


class TestSeasonalNaive:
    """Tests for SeasonalNaive model."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        model = SeasonalNaive()

        assert model.name == "SeasonalNaive"
        assert model.seasonal_period == 168
        assert model.is_fitted is False

    def test_init_custom(self) -> None:
        """Test custom initialization."""
        model = SeasonalNaive(seasonal_period=24, quantiles=[0.25, 0.75])

        assert model.seasonal_period == 24
        assert model.quantiles == [0.25, 0.75]

    def test_fit(self, engineered_features: pd.DataFrame) -> None:
        """Test model fitting."""
        model = SeasonalNaive()
        feature_cols = [c for c in engineered_features.columns if c != "load_mw"]

        X = engineered_features[feature_cols]
        y = engineered_features["load_mw"]

        model.fit(X, y)

        assert model.is_fitted is True

    def test_predict(self, engineered_features: pd.DataFrame) -> None:
        """Test model prediction."""
        model = SeasonalNaive()
        feature_cols = [c for c in engineered_features.columns if c != "load_mw"]

        X = engineered_features[feature_cols]
        y = engineered_features["load_mw"]

        model.fit(X, y)
        result = model.predict(X.tail(48), horizon=24)

        assert isinstance(result, ForecastResult)
        assert len(result.point_forecast) == 24
        assert result.model_name == "SeasonalNaive"

    def test_predict_quantiles(self, engineered_features: pd.DataFrame) -> None:
        """Test quantile predictions."""
        model = SeasonalNaive(quantiles=[0.1, 0.5, 0.9])
        feature_cols = [c for c in engineered_features.columns if c != "load_mw"]

        X = engineered_features[feature_cols]
        y = engineered_features["load_mw"]

        model.fit(X, y)
        result = model.predict(X.tail(48), horizon=24)

        assert 0.1 in result.quantile_forecasts
        assert 0.5 in result.quantile_forecasts
        assert 0.9 in result.quantile_forecasts

    def test_save_load(self, engineered_features: pd.DataFrame, tmp_path: Path) -> None:
        """Test model save and load."""
        model = SeasonalNaive()
        feature_cols = [c for c in engineered_features.columns if c != "load_mw"]

        X = engineered_features[feature_cols]
        y = engineered_features["load_mw"]

        model.fit(X, y)

        # Save
        model_path = tmp_path / "seasonal_naive.pkl"
        model.save(model_path)

        assert model_path.exists()

        # Load
        loaded_model = SeasonalNaive()
        loaded_model.load(model_path)

        assert loaded_model.is_fitted is True
        assert loaded_model.seasonal_period == model.seasonal_period


class TestLightGBMForecaster:
    """Tests for LightGBM forecaster."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        model = LightGBMForecaster()

        assert model.name == "LightGBM"
        assert model.is_fitted is False
        assert len(model.quantiles) == 3

    def test_init_custom_params(self) -> None:
        """Test custom parameters."""
        model = LightGBMForecaster(
            quantiles=[0.25, 0.5, 0.75],
            params={"n_estimators": 100},
        )

        assert model.quantiles == [0.25, 0.5, 0.75]
        assert model.params["n_estimators"] == 100

    def test_fit(self, engineered_features: pd.DataFrame) -> None:
        """Test model fitting."""
        model = LightGBMForecaster(params={"n_estimators": 10, "verbose": -1})
        feature_cols = [c for c in engineered_features.columns if c != "load_mw"]

        X = engineered_features[feature_cols]
        y = engineered_features["load_mw"]

        model.fit(X, y)

        assert model.is_fitted is True
        assert len(model.models) == len(model.quantiles)

    def test_predict(self, engineered_features: pd.DataFrame) -> None:
        """Test model prediction."""
        model = LightGBMForecaster(params={"n_estimators": 10, "verbose": -1})
        feature_cols = [c for c in engineered_features.columns if c != "load_mw"]

        X = engineered_features[feature_cols]
        y = engineered_features["load_mw"]

        model.fit(X, y)
        result = model.predict(X.tail(48), horizon=24)

        assert isinstance(result, ForecastResult)
        assert len(result.point_forecast) == 24

    def test_get_feature_importance(self, engineered_features: pd.DataFrame) -> None:
        """Test feature importance retrieval."""
        model = LightGBMForecaster(params={"n_estimators": 10, "verbose": -1})
        feature_cols = [c for c in engineered_features.columns if c != "load_mw"]

        X = engineered_features[feature_cols]
        y = engineered_features["load_mw"]

        model.fit(X, y)
        importance = model.get_feature_importance()

        assert isinstance(importance, pd.DataFrame)
        assert "feature" in importance.columns
        assert "importance" in importance.columns
        assert len(importance) == len(feature_cols)

    def test_save_load(self, engineered_features: pd.DataFrame, tmp_path: Path) -> None:
        """Test model save and load."""
        model = LightGBMForecaster(params={"n_estimators": 10, "verbose": -1})
        feature_cols = [c for c in engineered_features.columns if c != "load_mw"]

        X = engineered_features[feature_cols]
        y = engineered_features["load_mw"]

        model.fit(X, y)

        # Save
        model_path = tmp_path / "lightgbm.pkl"
        model.save(model_path)

        assert model_path.exists()

        # Load
        loaded_model = LightGBMForecaster()
        loaded_model.load(model_path)

        assert loaded_model.is_fitted is True
        assert loaded_model.feature_names == model.feature_names

        # Predictions should match
        result1 = model.predict(X.tail(24), horizon=24)
        result2 = loaded_model.predict(X.tail(24), horizon=24)

        np.testing.assert_array_almost_equal(result1.point_forecast, result2.point_forecast)
