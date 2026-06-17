"""
Feature engineering for electricity demand forecasting.

This module provides comprehensive feature engineering capabilities including:
- Lag features (historical load values)
- Rolling statistics (mean, std, min, max)
- Calendar features (hour, day, month, year)
- Fourier features (cyclic patterns)
- Holiday indicators
- Temperature features (if available)

See FEATURES.md for detailed documentation of all features.
"""

from dataclasses import dataclass, field

import holidays
import numpy as np
import pandas as pd

from forecast_service.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    # Lag features
    lag_hours: list[int] = field(default_factory=lambda: [1, 2, 3, 6, 12, 24, 48, 168])

    # Rolling statistics windows (in hours)
    rolling_windows: list[int] = field(default_factory=lambda: [6, 12, 24, 48, 168])

    # Fourier features
    fourier_daily_terms: int = 3
    fourier_weekly_terms: int = 2
    fourier_annual_terms: int = 4

    # Calendar features
    include_hour: bool = True
    include_day_of_week: bool = True
    include_day_of_month: bool = True
    include_month: bool = True
    include_quarter: bool = True
    include_year: bool = True
    include_is_weekend: bool = True

    # Holiday features
    include_holidays: bool = True
    holiday_country: str = "US"

    # Temperature features
    include_temperature_features: bool = True


class FeatureEngineer:
    """
    Generate features for electricity demand forecasting.

    This class creates a comprehensive feature set from raw time series data,
    suitable for tree-based models (LightGBM) and sequence models (LSTM/TCN).

    The feature engineering is reproducible and all features are documented
    in FEATURES.md.

    Attributes:
        config: Feature engineering configuration.
        feature_names_: List of generated feature names (available after transform).
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        """
        Initialize the feature engineer.

        Args:
            config: Feature configuration. Uses defaults if not provided.
        """
        self.config = config or FeatureConfig()
        self.feature_names_: list[str] = []
        self._holidays: holidays.HolidayBase | None = None

        if self.config.include_holidays:
            self._holidays = holidays.country_holidays(self.config.holiday_country)

        logger.info("Initialized FeatureEngineer", config=self.config)

    def transform(
        self,
        df: pd.DataFrame,
        target_col: str = "load_mw",
        drop_na: bool = True,
    ) -> pd.DataFrame:
        """
        Generate all features from the input DataFrame.

        Args:
            df: Input DataFrame with datetime index and target column.
            target_col: Name of the target variable column.
            drop_na: Whether to drop rows with NaN values (from lagging).

        Returns:
            DataFrame with all generated features and target column.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex")

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")

        logger.info("Starting feature engineering", rows=len(df))

        # Start with a copy of the input
        result = df[[target_col]].copy()

        # Generate each feature group
        result = self._add_lag_features(result, target_col)
        result = self._add_rolling_features(result, target_col)
        result = self._add_calendar_features(result)
        result = self._add_fourier_features(result)

        if self.config.include_holidays:
            result = self._add_holiday_features(result)

        # Add temperature features if temperature column exists
        if "temperature_f" in df.columns and self.config.include_temperature_features:
            result = self._add_temperature_features(result, df["temperature_f"])

        # Drop rows with NaN values
        if drop_na:
            n_before = len(result)
            result = result.dropna()
            n_dropped = n_before - len(result)
            if n_dropped > 0:
                logger.info("Dropped rows with NaN", count=n_dropped)

        # Store feature names (exclude target)
        self.feature_names_ = [col for col in result.columns if col != target_col]

        logger.info(
            "Feature engineering complete",
            n_features=len(self.feature_names_),
            n_rows=len(result),
        )

        return result

    def _add_lag_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Add lag features."""
        for lag in self.config.lag_hours:
            df[f"lag_{lag}h"] = df[target_col].shift(lag)
        return df

    def _add_rolling_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Add rolling statistics features."""
        for window in self.config.rolling_windows:
            # Rolling mean
            df[f"rolling_mean_{window}h"] = (
                df[target_col].shift(1).rolling(window=window, min_periods=1).mean()
            )

            # Rolling std
            df[f"rolling_std_{window}h"] = (
                df[target_col].shift(1).rolling(window=window, min_periods=1).std()
            )

            # Rolling min
            df[f"rolling_min_{window}h"] = (
                df[target_col].shift(1).rolling(window=window, min_periods=1).min()
            )

            # Rolling max
            df[f"rolling_max_{window}h"] = (
                df[target_col].shift(1).rolling(window=window, min_periods=1).max()
            )

        # Same hour previous day / week statistics
        df["same_hour_prev_day"] = df[target_col].shift(24)
        df["same_hour_prev_week"] = df[target_col].shift(168)

        # Difference features
        df["diff_1h"] = df[target_col].diff(1)
        df["diff_24h"] = df[target_col].diff(24)

        return df

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar-based features."""
        idx = df.index

        if self.config.include_hour:
            df["hour"] = idx.hour

        if self.config.include_day_of_week:
            df["day_of_week"] = idx.dayofweek

        if self.config.include_day_of_month:
            df["day_of_month"] = idx.day

        if self.config.include_month:
            df["month"] = idx.month

        if self.config.include_quarter:
            df["quarter"] = idx.quarter

        if self.config.include_year:
            df["year"] = idx.year

        if self.config.include_is_weekend:
            df["is_weekend"] = (idx.dayofweek >= 5).astype(int)

        return df

    def _add_fourier_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Fourier features for cyclic patterns."""
        idx = df.index

        # Daily cycle (24-hour period)
        hour_fraction = idx.hour / 24
        for i in range(1, self.config.fourier_daily_terms + 1):
            df[f"daily_sin_{i}"] = np.sin(2 * np.pi * i * hour_fraction)
            df[f"daily_cos_{i}"] = np.cos(2 * np.pi * i * hour_fraction)

        # Weekly cycle (168-hour period)
        week_fraction = (idx.dayofweek * 24 + idx.hour) / 168
        for i in range(1, self.config.fourier_weekly_terms + 1):
            df[f"weekly_sin_{i}"] = np.sin(2 * np.pi * i * week_fraction)
            df[f"weekly_cos_{i}"] = np.cos(2 * np.pi * i * week_fraction)

        # Annual cycle (365.25 days)
        day_of_year = idx.dayofyear
        year_fraction = day_of_year / 365.25
        for i in range(1, self.config.fourier_annual_terms + 1):
            df[f"annual_sin_{i}"] = np.sin(2 * np.pi * i * year_fraction)
            df[f"annual_cos_{i}"] = np.cos(2 * np.pi * i * year_fraction)

        return df

    def _add_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add holiday indicator features."""
        if self._holidays is None:
            return df

        dates = df.index.date

        # Binary holiday indicator
        df["is_holiday"] = [1 if d in self._holidays else 0 for d in dates]

        # Day before holiday
        df["is_day_before_holiday"] = df["is_holiday"].shift(-24).fillna(0).astype(int)

        # Day after holiday
        df["is_day_after_holiday"] = df["is_holiday"].shift(24).fillna(0).astype(int)

        return df

    def _add_temperature_features(self, df: pd.DataFrame, temperature: pd.Series) -> pd.DataFrame:
        """Add temperature-based features."""
        df["temperature"] = temperature.values

        # Temperature lags
        df["temp_lag_1h"] = temperature.shift(1).values
        df["temp_lag_24h"] = temperature.shift(24).values

        # Rolling temperature stats
        df["temp_rolling_mean_24h"] = temperature.rolling(window=24, min_periods=1).mean().values
        df["temp_rolling_std_24h"] = temperature.rolling(window=24, min_periods=1).std().values

        # Heating and cooling degree features
        # Reference temperature for HVAC load
        ref_temp = 65.0
        df["heating_degree"] = np.maximum(0, ref_temp - temperature.values)
        df["cooling_degree"] = np.maximum(0, temperature.values - ref_temp)

        # Squared terms (for non-linear relationship)
        df["heating_degree_sq"] = df["heating_degree"] ** 2
        df["cooling_degree_sq"] = df["cooling_degree"] ** 2

        return df

    def get_feature_names(self) -> list[str]:
        """Get the list of generated feature names."""
        return self.feature_names_.copy()

    def get_feature_groups(self) -> dict[str, list[str]]:
        """Get features organized by group."""
        groups: dict[str, list[str]] = {
            "lag": [],
            "rolling": [],
            "calendar": [],
            "fourier": [],
            "holiday": [],
            "temperature": [],
        }

        for name in self.feature_names_:
            if name.startswith("lag_") or name.startswith("same_hour") or name.startswith("diff_"):
                groups["lag"].append(name)
            elif name.startswith("rolling_"):
                groups["rolling"].append(name)
            elif name in (
                "hour",
                "day_of_week",
                "day_of_month",
                "month",
                "quarter",
                "year",
                "is_weekend",
            ):
                groups["calendar"].append(name)
            elif (
                name.endswith("_sin_")
                or name.endswith("_cos_")
                or "_sin_" in name
                or "_cos_" in name
            ):
                groups["fourier"].append(name)
            elif "holiday" in name:
                groups["holiday"].append(name)
            elif "temp" in name or "degree" in name:
                groups["temperature"].append(name)

        return groups


def create_features(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
    target_col: str = "load_mw",
) -> tuple[pd.DataFrame, FeatureEngineer]:
    """
    Convenience function to create features from a DataFrame.

    Args:
        df: Input DataFrame with datetime index.
        config: Feature configuration.
        target_col: Name of target column.

    Returns:
        Tuple of (feature DataFrame, fitted FeatureEngineer).
    """
    engineer = FeatureEngineer(config=config)
    features = engineer.transform(df, target_col=target_col)
    return features, engineer
