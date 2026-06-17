"""Tests for feature engineering."""

import pandas as pd
import pytest

from forecast_service.features.engineering import (
    FeatureConfig,
    FeatureEngineer,
    create_features,
)


class TestFeatureConfig:
    """Tests for FeatureConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = FeatureConfig()

        assert config.lag_hours == [1, 2, 3, 6, 12, 24, 48, 168]
        assert config.rolling_windows == [6, 12, 24, 48, 168]
        assert config.fourier_daily_terms == 3
        assert config.include_holidays is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = FeatureConfig(
            lag_hours=[1, 24],
            rolling_windows=[24],
            fourier_daily_terms=2,
            include_holidays=False,
        )

        assert config.lag_hours == [1, 24]
        assert config.rolling_windows == [24]
        assert config.fourier_daily_terms == 2
        assert config.include_holidays is False


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        engineer = FeatureEngineer()
        assert engineer.config is not None
        assert engineer.feature_names_ == []

    def test_init_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = FeatureConfig(lag_hours=[1, 24])
        engineer = FeatureEngineer(config=config)
        assert engineer.config.lag_hours == [1, 24]

    def test_transform_basic(self, small_synthetic_data: pd.DataFrame) -> None:
        """Test basic feature transformation."""
        engineer = FeatureEngineer()
        features = engineer.transform(small_synthetic_data, target_col="load_mw")

        assert isinstance(features, pd.DataFrame)
        assert "load_mw" in features.columns
        assert len(engineer.feature_names_) > 0
        assert len(features) > 0

    def test_transform_creates_lag_features(
        self, small_synthetic_data: pd.DataFrame
    ) -> None:
        """Test that lag features are created."""
        config = FeatureConfig(lag_hours=[1, 24])
        engineer = FeatureEngineer(config=config)
        features = engineer.transform(small_synthetic_data, target_col="load_mw")

        assert "lag_1h" in features.columns
        assert "lag_24h" in features.columns

    def test_transform_creates_rolling_features(
        self, small_synthetic_data: pd.DataFrame
    ) -> None:
        """Test that rolling features are created."""
        config = FeatureConfig(rolling_windows=[24])
        engineer = FeatureEngineer(config=config)
        features = engineer.transform(small_synthetic_data, target_col="load_mw")

        assert "rolling_mean_24h" in features.columns
        assert "rolling_std_24h" in features.columns
        assert "rolling_min_24h" in features.columns
        assert "rolling_max_24h" in features.columns

    def test_transform_creates_calendar_features(
        self, small_synthetic_data: pd.DataFrame
    ) -> None:
        """Test that calendar features are created."""
        engineer = FeatureEngineer()
        features = engineer.transform(small_synthetic_data, target_col="load_mw")

        assert "hour" in features.columns
        assert "day_of_week" in features.columns
        assert "month" in features.columns
        assert "is_weekend" in features.columns

    def test_transform_creates_fourier_features(
        self, small_synthetic_data: pd.DataFrame
    ) -> None:
        """Test that Fourier features are created."""
        engineer = FeatureEngineer()
        features = engineer.transform(small_synthetic_data, target_col="load_mw")

        assert "daily_sin_1" in features.columns
        assert "daily_cos_1" in features.columns
        assert "weekly_sin_1" in features.columns
        assert "annual_sin_1" in features.columns

    def test_transform_creates_holiday_features(
        self, small_synthetic_data: pd.DataFrame
    ) -> None:
        """Test that holiday features are created."""
        config = FeatureConfig(include_holidays=True)
        engineer = FeatureEngineer(config=config)
        features = engineer.transform(small_synthetic_data, target_col="load_mw")

        assert "is_holiday" in features.columns

    def test_transform_creates_temperature_features(
        self, small_synthetic_data: pd.DataFrame
    ) -> None:
        """Test that temperature features are created when available."""
        engineer = FeatureEngineer()
        features = engineer.transform(small_synthetic_data, target_col="load_mw")

        assert "temperature" in features.columns
        assert "heating_degree" in features.columns
        assert "cooling_degree" in features.columns

    def test_transform_drops_na(self, small_synthetic_data: pd.DataFrame) -> None:
        """Test that NaN values are dropped."""
        engineer = FeatureEngineer()
        features = engineer.transform(
            small_synthetic_data, target_col="load_mw", drop_na=True
        )

        assert features.isna().sum().sum() == 0

    def test_transform_keeps_na(self, small_synthetic_data: pd.DataFrame) -> None:
        """Test that NaN values can be kept."""
        engineer = FeatureEngineer()
        features = engineer.transform(
            small_synthetic_data, target_col="load_mw", drop_na=False
        )

        # Should have NaN from lagging
        assert features.isna().sum().sum() > 0

    def test_transform_invalid_index(self) -> None:
        """Test error on non-datetime index."""
        df = pd.DataFrame({"load_mw": [1, 2, 3]})
        engineer = FeatureEngineer()

        with pytest.raises(ValueError, match="DatetimeIndex"):
            engineer.transform(df, target_col="load_mw")

    def test_transform_missing_target(
        self, small_synthetic_data: pd.DataFrame
    ) -> None:
        """Test error on missing target column."""
        engineer = FeatureEngineer()

        with pytest.raises(ValueError, match="not found"):
            engineer.transform(small_synthetic_data, target_col="nonexistent")

    def test_get_feature_names(self, small_synthetic_data: pd.DataFrame) -> None:
        """Test getting feature names after transform."""
        engineer = FeatureEngineer()
        engineer.transform(small_synthetic_data, target_col="load_mw")

        names = engineer.get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 0
        assert "load_mw" not in names  # Target not in features

    def test_get_feature_groups(self, small_synthetic_data: pd.DataFrame) -> None:
        """Test getting features organized by group."""
        engineer = FeatureEngineer()
        engineer.transform(small_synthetic_data, target_col="load_mw")

        groups = engineer.get_feature_groups()
        assert "lag" in groups
        assert "rolling" in groups
        assert "calendar" in groups
        assert "fourier" in groups


class TestCreateFeatures:
    """Tests for create_features convenience function."""

    def test_create_features(self, small_synthetic_data: pd.DataFrame) -> None:
        """Test convenience function."""
        features, engineer = create_features(small_synthetic_data)

        assert isinstance(features, pd.DataFrame)
        assert isinstance(engineer, FeatureEngineer)
        assert len(engineer.feature_names_) > 0
