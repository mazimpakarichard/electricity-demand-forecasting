"""
LightGBM forecasting model with quantile regression support.
"""

import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd

from forecast_service.models.base import BaseForecaster, ForecastResult
from forecast_service.utils.logging import get_logger

logger = get_logger(__name__)


class LightGBMForecaster(BaseForecaster):
    """
    LightGBM forecaster with multi-quantile regression.

    Trains separate models for each quantile to provide probabilistic
    forecasts. Uses gradient boosting with tree learners optimized
    for tabular data.

    Attributes:
        models: Dictionary mapping quantiles to fitted LightGBM models.
        feature_names: Names of features used during training.
    """

    DEFAULT_PARAMS: dict[str, Any] = {
        "objective": "quantile",
        "metric": "quantile",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "n_estimators": 500,
        "early_stopping_rounds": 50,
        "verbose": -1,
        "random_state": 42,
    }

    def __init__(
        self,
        quantiles: list[float] | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize LightGBM forecaster.

        Args:
            quantiles: Quantiles to forecast.
            params: LightGBM parameters. Merged with DEFAULT_PARAMS.
        """
        super().__init__(name="LightGBM", quantiles=quantiles)

        self.params = {**self.DEFAULT_PARAMS}
        if params:
            self.params.update(params)

        self.models: dict[float, lgb.LGBMRegressor] = {}
        self.feature_names: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "LightGBMForecaster":
        """
        Train LightGBM models for each quantile.

        Args:
            X: Training features.
            y: Training target.
            X_val: Validation features (optional, for early stopping).
            y_val: Validation target (optional, for early stopping).

        Returns:
            Self.
        """
        self.feature_names = list(X.columns)

        logger.info(
            "Training LightGBM models",
            n_quantiles=len(self.quantiles),
            n_features=len(self.feature_names),
            n_samples=len(X),
        )

        # Create validation set if not provided (last 10% of data)
        if X_val is None or y_val is None:
            val_size = max(168, int(len(X) * 0.1))  # At least 1 week
            X_val = X.iloc[-val_size:]
            y_val = y.iloc[-val_size:]
            X_train = X.iloc[:-val_size]
            y_train = y.iloc[:-val_size]
        else:
            X_train = X
            y_train = y

        for q in self.quantiles:
            logger.info(f"Training quantile {q}")

            model_params = {**self.params}
            model_params["alpha"] = q

            model = lgb.LGBMRegressor(**model_params)

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
            )

            self.models[q] = model

            logger.info(
                f"Quantile {q} trained",
                best_iteration=model.best_iteration_,
            )

        self.is_fitted = True
        return self

    def predict(
        self,
        X: pd.DataFrame,
        horizon: int = 24,
    ) -> ForecastResult:
        """
        Generate multi-quantile forecasts.

        Args:
            X: Feature matrix.
            horizon: Number of hours to forecast.

        Returns:
            ForecastResult with point and quantile forecasts.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_pred = X.iloc[:horizon]
        timestamps = X_pred.index

        quantile_forecasts: dict[float, npt.NDArray[np.floating[Any]]] = {}

        for q, model in self.models.items():
            pred = model.predict(X_pred)
            quantile_forecasts[q] = pred

        # Point forecast is the median (p50)
        point_forecast = quantile_forecasts.get(
            0.5,
            np.median(list(quantile_forecasts.values()), axis=0)
        )

        return ForecastResult(
            timestamps=pd.DatetimeIndex(timestamps),
            point_forecast=point_forecast,
            quantile_forecasts=quantile_forecasts,
            model_name=self.name,
        )

    def get_feature_importance(
        self,
        importance_type: str = "gain",
    ) -> pd.DataFrame:
        """
        Get feature importance from the median model.

        Args:
            importance_type: Type of importance ('gain', 'split').

        Returns:
            DataFrame with feature names and importance scores.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # Use the median (p50) model
        model = self.models.get(0.5)
        if model is None:
            model = list(self.models.values())[0]

        importance = model.feature_importances_

        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        })

        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def get_model_for_quantile(self, quantile: float) -> lgb.LGBMRegressor:
        """Get the trained model for a specific quantile."""
        if quantile not in self.models:
            raise ValueError(f"No model for quantile {quantile}")
        return self.models[quantile]

    def save(self, path: Path | str) -> None:
        """Save models to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "quantiles": self.quantiles,
            "params": self.params,
            "feature_names": self.feature_names,
            "models": {},
            "is_fitted": self.is_fitted,
        }

        # Save each model's booster
        for q, model in self.models.items():
            state["models"][q] = model

        with open(path, "wb") as f:
            pickle.dump(state, f)

        logger.info("Saved LightGBM model", path=str(path))

    def load(self, path: Path | str) -> "LightGBMForecaster":
        """Load models from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)

        self.quantiles = state["quantiles"]
        self.params = state["params"]
        self.feature_names = state["feature_names"]
        self.models = state["models"]
        self.is_fitted = state["is_fitted"]

        logger.info("Loaded LightGBM model", path=str(path))
        return self
