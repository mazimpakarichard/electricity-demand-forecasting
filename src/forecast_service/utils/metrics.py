"""Evaluation metrics for time series forecasting."""

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

ArrayLike: TypeAlias = npt.NDArray[np.floating[Any]]


@dataclass
class ForecastMetrics:
    """Container for forecast evaluation metrics."""

    mape: float
    rmse: float
    mae: float
    pinball_loss: dict[float, float]  # quantile -> loss

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to a flat dictionary."""
        result: dict[str, float] = {
            "mape": self.mape,
            "rmse": self.rmse,
            "mae": self.mae,
        }
        for quantile, loss in self.pinball_loss.items():
            result[f"pinball_p{int(quantile * 100)}"] = loss
        return result


def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        MAPE as a percentage (0-100 scale).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Avoid division by zero
    mask = y_true != 0
    if not np.any(mask):
        return float("inf")

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Calculate Root Mean Squared Error.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        RMSE in the same units as the target variable.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Calculate Mean Absolute Error.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        MAE in the same units as the target variable.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def pinball_loss(y_true: ArrayLike, y_pred: ArrayLike, quantile: float) -> float:
    """
    Calculate pinball loss for quantile forecasting.

    The pinball loss is asymmetric and penalizes over/under-predictions
    differently based on the quantile.

    Args:
        y_true: Actual values.
        y_pred: Predicted quantile values.
        quantile: The quantile (0-1) for which the prediction was made.

    Returns:
        Pinball loss value.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    diff = y_true - y_pred
    loss = np.where(diff >= 0, quantile * diff, (quantile - 1) * diff)
    return float(np.mean(loss))


def calculate_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    quantile_predictions: dict[float, ArrayLike] | None = None,
) -> ForecastMetrics:
    """
    Calculate comprehensive forecast metrics.

    Args:
        y_true: Actual values.
        y_pred: Point predictions (typically p50/median).
        quantile_predictions: Dict mapping quantiles to their predictions.

    Returns:
        ForecastMetrics containing all calculated metrics.
    """
    pinball_losses: dict[float, float] = {}

    if quantile_predictions is not None:
        for q, q_pred in quantile_predictions.items():
            pinball_losses[q] = pinball_loss(y_true, q_pred, q)

    return ForecastMetrics(
        mape=mape(y_true, y_pred),
        rmse=rmse(y_true, y_pred),
        mae=mae(y_true, y_pred),
        pinball_loss=pinball_losses,
    )


def coverage(
    y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike
) -> float:
    """
    Calculate prediction interval coverage.

    Args:
        y_true: Actual values.
        lower: Lower bound of prediction interval.
        upper: Upper bound of prediction interval.

    Returns:
        Proportion of actuals falling within the interval.
    """
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    within = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(within))
