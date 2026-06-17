"""
Baseline forecasting models.

Includes:
- SeasonalNaive: Uses same hour from previous week
- SARIMAForecaster: Seasonal ARIMA model
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from forecast_service.models.base import BaseForecaster, ForecastResult
from forecast_service.utils.logging import get_logger

logger = get_logger(__name__)


class SeasonalNaive(BaseForecaster):
    """
    Seasonal Naive forecaster.

    This baseline uses the value from the same hour in the previous week
    as the forecast. For quantile forecasts, it uses historical prediction
    errors to estimate uncertainty.

    This is a strong baseline for electricity demand forecasting due to
    the strong weekly seasonality in load patterns.
    """

    def __init__(
        self,
        seasonal_period: int = 168,  # 1 week in hours
        quantiles: list[float] | None = None,
    ) -> None:
        """
        Initialize Seasonal Naive forecaster.

        Args:
            seasonal_period: Seasonality period in hours (default: 168 = 1 week).
            quantiles: Quantiles to forecast.
        """
        super().__init__(name="SeasonalNaive", quantiles=quantiles)
        self.seasonal_period = seasonal_period
        self._history: pd.Series | None = None
        self._error_quantiles: dict[float, float] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "SeasonalNaive":
        """
        Fit the model by storing history and computing error quantiles.

        Args:
            X: Feature matrix (not used but kept for API consistency).
            y: Target values (historical load).

        Returns:
            Self.
        """
        self._history = y.copy()

        # Compute in-sample prediction errors for quantile estimation
        predictions = y.shift(self.seasonal_period)
        errors = y - predictions
        errors = errors.dropna()

        for q in self.quantiles:
            self._error_quantiles[q] = float(errors.quantile(q))

        self.is_fitted = True
        logger.info("Fitted SeasonalNaive", history_size=len(y))

        return self

    def predict(
        self,
        X: pd.DataFrame,
        horizon: int = 24,
    ) -> ForecastResult:
        """
        Generate forecasts using seasonal naive method.

        Args:
            X: Feature matrix (uses index for timestamps).
            horizon: Number of hours to forecast.

        Returns:
            ForecastResult with point and quantile forecasts.
        """
        if not self.is_fitted or self._history is None:
            raise ValueError("Model must be fitted before prediction")

        timestamps = X.index[:horizon]
        n_pred = len(timestamps)

        # Get point forecast (median)
        point_forecast = np.zeros(n_pred)

        for i, ts in enumerate(timestamps):
            # Look back one seasonal period
            lookback_ts = ts - pd.Timedelta(hours=self.seasonal_period)

            if lookback_ts in self._history.index:
                point_forecast[i] = self._history.loc[lookback_ts]
            else:
                # Fallback to mean if history not available
                point_forecast[i] = self._history.mean()

        # Generate quantile forecasts using error distribution
        quantile_forecasts: dict[float, npt.NDArray[np.floating[Any]]] = {}
        for q in self.quantiles:
            error_adj = self._error_quantiles.get(q, 0)
            quantile_forecasts[q] = point_forecast + error_adj

        return ForecastResult(
            timestamps=pd.DatetimeIndex(timestamps),
            point_forecast=point_forecast,
            quantile_forecasts=quantile_forecasts,
            model_name=self.name,
        )

    def save(self, path: Path | str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "seasonal_period": self.seasonal_period,
            "quantiles": self.quantiles,
            "history": self._history,
            "error_quantiles": self._error_quantiles,
            "is_fitted": self.is_fitted,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

        logger.info("Saved SeasonalNaive model", path=str(path))

    def load(self, path: Path | str) -> "SeasonalNaive":
        """Load model from disk."""
        # Note: Only load models from trusted sources (our own saved models)
        with open(path, "rb") as f:
            state = pickle.load(f)  # nosec B301

        self.seasonal_period = state["seasonal_period"]
        self.quantiles = state["quantiles"]
        self._history = state["history"]
        self._error_quantiles = state["error_quantiles"]
        self.is_fitted = state["is_fitted"]

        logger.info("Loaded SeasonalNaive model", path=str(path))
        return self


class SARIMAForecaster(BaseForecaster):
    """
    Seasonal ARIMA (SARIMA) forecaster.

    Fits a SARIMAX model with specified order parameters.
    Uses prediction intervals for quantile forecasts.
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 0, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 24),
        quantiles: list[float] | None = None,
    ) -> None:
        """
        Initialize SARIMA forecaster.

        Args:
            order: ARIMA (p, d, q) order.
            seasonal_order: Seasonal (P, D, Q, s) order.
            quantiles: Quantiles to forecast.
        """
        super().__init__(name="SARIMA", quantiles=quantiles)
        self.order = order
        self.seasonal_order = seasonal_order
        self._model: Any = None
        self._fitted_model: Any = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "SARIMAForecaster":
        """
        Fit SARIMA model.

        Args:
            X: Feature matrix (can include exogenous variables).
            y: Target values.

        Returns:
            Self.
        """
        logger.info(
            "Fitting SARIMA model",
            order=self.order,
            seasonal_order=self.seasonal_order,
            n_obs=len(y),
        )

        # Use subset for faster fitting if data is very long
        max_obs = 8760  # 1 year
        if len(y) > max_obs:
            y = y.iloc[-max_obs:]
            logger.info("Using last year of data for SARIMA fitting")

        self._model = SARIMAX(
            y,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        self._fitted_model = self._model.fit(disp=False, maxiter=100)
        self.is_fitted = True

        logger.info(
            "SARIMA model fitted",
            aic=f"{self._fitted_model.aic:.0f}",
            bic=f"{self._fitted_model.bic:.0f}",
        )

        return self

    def predict(
        self,
        X: pd.DataFrame,
        horizon: int = 24,
    ) -> ForecastResult:
        """
        Generate SARIMA forecasts with prediction intervals.

        Args:
            X: Feature matrix (uses index for timestamps).
            horizon: Number of hours to forecast.

        Returns:
            ForecastResult with point and quantile forecasts.
        """
        if not self.is_fitted or self._fitted_model is None:
            raise ValueError("Model must be fitted before prediction")

        timestamps = X.index[:horizon]

        # Get forecast with confidence intervals
        forecast = self._fitted_model.get_forecast(steps=horizon)
        point_forecast = forecast.predicted_mean.values

        # Generate quantile forecasts from prediction intervals
        quantile_forecasts: dict[float, npt.NDArray[np.floating[Any]]] = {}

        for q in self.quantiles:
            if q == 0.5:
                quantile_forecasts[q] = point_forecast
            else:
                # Convert quantile to confidence level
                alpha = 2 * abs(q - 0.5)
                conf_int = forecast.conf_int(alpha=1 - alpha)

                if q < 0.5:
                    quantile_forecasts[q] = conf_int.iloc[:, 0].values
                else:
                    quantile_forecasts[q] = conf_int.iloc[:, 1].values

        return ForecastResult(
            timestamps=pd.DatetimeIndex(timestamps),
            point_forecast=point_forecast,
            quantile_forecasts=quantile_forecasts,
            model_name=self.name,
        )

    def save(self, path: Path | str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "quantiles": self.quantiles,
            "fitted_model": self._fitted_model,
            "is_fitted": self.is_fitted,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

        logger.info("Saved SARIMA model", path=str(path))

    def load(self, path: Path | str) -> "SARIMAForecaster":
        """Load model from disk."""
        # Note: Only load models from trusted sources (our own saved models)
        with open(path, "rb") as f:
            state = pickle.load(f)  # nosec B301

        self.order = state["order"]
        self.seasonal_order = state["seasonal_order"]
        self.quantiles = state["quantiles"]
        self._fitted_model = state["fitted_model"]
        self.is_fitted = state["is_fitted"]

        logger.info("Loaded SARIMA model", path=str(path))
        return self
