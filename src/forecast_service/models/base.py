"""Base classes for forecasting models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from forecast_service.utils.metrics import ForecastMetrics, calculate_metrics


@dataclass
class ForecastResult:
    """Container for forecast results with quantiles."""

    timestamps: pd.DatetimeIndex
    point_forecast: npt.NDArray[np.floating[Any]]  # p50
    quantile_forecasts: dict[float, npt.NDArray[np.floating[Any]]]
    model_name: str

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame format."""
        data = {
            "timestamp": self.timestamps,
            "forecast_p50": self.point_forecast,
        }
        for q, values in self.quantile_forecasts.items():
            data[f"forecast_p{int(q * 100)}"] = values

        return pd.DataFrame(data).set_index("timestamp")


@dataclass
class CrossValidationResult:
    """Results from time series cross-validation."""

    fold_metrics: list[ForecastMetrics]
    mean_metrics: dict[str, float] = field(default_factory=dict)
    std_metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Calculate mean and std of metrics across folds."""
        if self.fold_metrics:
            all_metrics = [m.to_dict() for m in self.fold_metrics]
            metric_names = all_metrics[0].keys()

            for name in metric_names:
                values = [m[name] for m in all_metrics]
                self.mean_metrics[name] = float(np.mean(values))
                self.std_metrics[name] = float(np.std(values))

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with one row per fold."""
        records = []
        for i, metrics in enumerate(self.fold_metrics):
            record = {"fold": i + 1, **metrics.to_dict()}
            records.append(record)

        df = pd.DataFrame(records)

        # Add summary row
        summary = {"fold": "mean", **self.mean_metrics}
        df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

        return df


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting models.

    All forecasters must implement:
    - fit(): Train the model
    - predict(): Generate forecasts
    - save()/load(): Model persistence

    Attributes:
        name: Model name for identification.
        quantiles: Quantiles to forecast (e.g., [0.1, 0.5, 0.9]).
        is_fitted: Whether the model has been trained.
    """

    def __init__(
        self,
        name: str = "BaseForecaster",
        quantiles: list[float] | None = None,
    ) -> None:
        """
        Initialize the forecaster.

        Args:
            name: Model name.
            quantiles: Quantiles to forecast.
        """
        self.name = name
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.is_fitted = False

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "BaseForecaster":
        """
        Train the model.

        Args:
            X: Feature matrix.
            y: Target values.

        Returns:
            Self for method chaining.
        """
        pass

    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame,
        horizon: int = 24,
    ) -> ForecastResult:
        """
        Generate forecasts.

        Args:
            X: Feature matrix for forecast period.
            horizon: Number of hours to forecast.

        Returns:
            ForecastResult with point and quantile forecasts.
        """
        pass

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """Save the model to disk."""
        pass

    @abstractmethod
    def load(self, path: Path | str) -> "BaseForecaster":
        """Load a model from disk."""
        pass

    def evaluate(
        self,
        y_true: pd.Series,
        y_pred: ForecastResult,
    ) -> ForecastMetrics:
        """
        Evaluate forecast accuracy.

        Args:
            y_true: Actual values.
            y_pred: Forecast results.

        Returns:
            ForecastMetrics with MAPE, RMSE, MAE, and pinball losses.
        """
        return calculate_metrics(
            y_true=y_true.values,
            y_pred=y_pred.point_forecast,
            quantile_predictions=y_pred.quantile_forecasts,
        )

    def cross_validate(
        self,
        df: pd.DataFrame,
        target_col: str = "load_mw",
        n_folds: int = 5,
        test_size: int = 168,  # 1 week
        gap: int = 24,  # 1 day gap to prevent leakage
    ) -> CrossValidationResult:
        """
        Perform rolling-origin time series cross-validation.

        This implements a rolling origin approach where:
        1. Training set grows with each fold
        2. A gap period separates training and test
        3. Test period has fixed size

        Args:
            df: DataFrame with features and target.
            target_col: Name of target column.
            n_folds: Number of CV folds.
            test_size: Size of each test period (hours).
            gap: Gap between training and test (hours).

        Returns:
            CrossValidationResult with metrics for each fold.
        """
        from forecast_service.utils.logging import get_logger
        logger = get_logger(__name__)

        feature_cols = [c for c in df.columns if c != target_col]
        n = len(df)

        # Calculate fold boundaries
        # Reserve enough data for all folds
        total_test_space = n_folds * (test_size + gap)
        min_train_size = max(168 * 4, int(n * 0.3))  # At least 4 weeks or 30%

        if n < min_train_size + total_test_space:
            raise ValueError(
                f"Not enough data for {n_folds} folds. "
                f"Have {n} rows, need at least {min_train_size + total_test_space}"
            )

        fold_metrics = []

        for fold in range(n_folds):
            # Calculate split points
            test_end = n - fold * (test_size + gap)
            test_start = test_end - test_size
            train_end = test_start - gap

            if train_end < min_train_size:
                logger.warning(f"Skipping fold {fold + 1}: insufficient training data")
                continue

            # Split data
            train_df = df.iloc[:train_end]
            test_df = df.iloc[test_start:test_end]

            X_train = train_df[feature_cols]
            y_train = train_df[target_col]
            X_test = test_df[feature_cols]
            y_test = test_df[target_col]

            logger.info(
                f"CV Fold {fold + 1}/{n_folds}",
                train_size=len(train_df),
                test_size=len(test_df),
            )

            # Train and predict
            self.fit(X_train, y_train)
            predictions = self.predict(X_test, horizon=len(X_test))

            # Evaluate
            metrics = self.evaluate(y_test, predictions)
            fold_metrics.append(metrics)

            logger.info(
                f"Fold {fold + 1} metrics",
                mape=f"{metrics.mape:.2f}%",
                rmse=f"{metrics.rmse:.0f}",
            )

        return CrossValidationResult(fold_metrics=fold_metrics)
