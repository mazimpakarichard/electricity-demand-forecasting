"""Tests for evaluation metrics."""

import numpy as np
import pytest

from forecast_service.utils.metrics import (
    ForecastMetrics,
    calculate_metrics,
    coverage,
    mae,
    mape,
    pinball_loss,
    rmse,
)


class TestMAPE:
    """Tests for MAPE metric."""

    def test_mape_perfect(self) -> None:
        """Test MAPE with perfect predictions."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([100, 200, 300])

        result = mape(y_true, y_pred)
        assert result == pytest.approx(0.0)

    def test_mape_known_value(self) -> None:
        """Test MAPE with known value."""
        y_true = np.array([100, 100])
        y_pred = np.array([90, 110])

        # |10/100| + |10/100| = 0.2, mean = 0.1 = 10%
        result = mape(y_true, y_pred)
        assert result == pytest.approx(10.0)

    def test_mape_handles_zeros(self) -> None:
        """Test MAPE handles zero actual values."""
        y_true = np.array([100, 0, 200])
        y_pred = np.array([100, 50, 200])

        # Should skip zeros
        result = mape(y_true, y_pred)
        assert result == pytest.approx(0.0)

    def test_mape_all_zeros(self) -> None:
        """Test MAPE with all zeros."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 2, 3])

        result = mape(y_true, y_pred)
        assert result == float("inf")


class TestRMSE:
    """Tests for RMSE metric."""

    def test_rmse_perfect(self) -> None:
        """Test RMSE with perfect predictions."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([100, 200, 300])

        result = rmse(y_true, y_pred)
        assert result == pytest.approx(0.0)

    def test_rmse_known_value(self) -> None:
        """Test RMSE with known value."""
        y_true = np.array([100, 100, 100])
        y_pred = np.array([90, 100, 110])

        # sqrt((100 + 0 + 100) / 3) = sqrt(200/3) ≈ 8.16
        result = rmse(y_true, y_pred)
        assert result == pytest.approx(np.sqrt(200 / 3))


class TestMAE:
    """Tests for MAE metric."""

    def test_mae_perfect(self) -> None:
        """Test MAE with perfect predictions."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([100, 200, 300])

        result = mae(y_true, y_pred)
        assert result == pytest.approx(0.0)

    def test_mae_known_value(self) -> None:
        """Test MAE with known value."""
        y_true = np.array([100, 100, 100])
        y_pred = np.array([90, 100, 110])

        # (10 + 0 + 10) / 3 ≈ 6.67
        result = mae(y_true, y_pred)
        assert result == pytest.approx(20 / 3)


class TestPinballLoss:
    """Tests for pinball loss metric."""

    def test_pinball_median(self) -> None:
        """Test pinball loss at median (q=0.5)."""
        y_true = np.array([100, 100])
        y_pred = np.array([90, 110])

        # At q=0.5, pinball = MAE / 2
        result = pinball_loss(y_true, y_pred, quantile=0.5)
        assert result == pytest.approx(5.0)  # (10 + 10) / 2 / 2

    def test_pinball_low_quantile(self) -> None:
        """Test pinball loss at low quantile."""
        y_true = np.array([100])
        y_pred = np.array([110])  # Over-predict

        # q=0.1, diff=-10 (negative), loss = (0.1-1)*(-10) = 9
        result = pinball_loss(y_true, y_pred, quantile=0.1)
        assert result == pytest.approx(9.0)

    def test_pinball_high_quantile(self) -> None:
        """Test pinball loss at high quantile."""
        y_true = np.array([100])
        y_pred = np.array([90])  # Under-predict

        # q=0.9, diff=10 (positive), loss = 0.9*10 = 9
        result = pinball_loss(y_true, y_pred, quantile=0.9)
        assert result == pytest.approx(9.0)


class TestCoverage:
    """Tests for prediction interval coverage."""

    def test_coverage_perfect(self) -> None:
        """Test coverage when all values are covered."""
        y_true = np.array([100, 200, 300])
        lower = np.array([90, 190, 290])
        upper = np.array([110, 210, 310])

        result = coverage(y_true, lower, upper)
        assert result == pytest.approx(1.0)

    def test_coverage_none(self) -> None:
        """Test coverage when no values are covered."""
        y_true = np.array([100, 200, 300])
        lower = np.array([200, 300, 400])
        upper = np.array([210, 310, 410])

        result = coverage(y_true, lower, upper)
        assert result == pytest.approx(0.0)

    def test_coverage_partial(self) -> None:
        """Test partial coverage."""
        y_true = np.array([100, 200, 300, 400])
        lower = np.array([90, 190, 350, 450])  # 2 covered (100, 200)
        upper = np.array([110, 210, 360, 460])  # 300, 400 NOT covered

        result = coverage(y_true, lower, upper)
        assert result == pytest.approx(0.5)


class TestForecastMetrics:
    """Tests for ForecastMetrics dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        metrics = ForecastMetrics(
            mape=5.0,
            rmse=100.0,
            mae=80.0,
            pinball_loss={0.1: 10.0, 0.5: 8.0, 0.9: 12.0},
        )

        result = metrics.to_dict()

        assert result["mape"] == 5.0
        assert result["rmse"] == 100.0
        assert result["mae"] == 80.0
        assert result["pinball_p10"] == 10.0
        assert result["pinball_p50"] == 8.0
        assert result["pinball_p90"] == 12.0


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    def test_calculate_metrics_basic(self) -> None:
        """Test basic metrics calculation."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([100, 200, 300])

        result = calculate_metrics(y_true, y_pred)

        assert isinstance(result, ForecastMetrics)
        assert result.mape == pytest.approx(0.0)
        assert result.rmse == pytest.approx(0.0)
        assert result.mae == pytest.approx(0.0)

    def test_calculate_metrics_with_quantiles(self) -> None:
        """Test metrics calculation with quantile predictions."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([100, 200, 300])
        quantiles = {
            0.1: np.array([90, 190, 290]),
            0.5: np.array([100, 200, 300]),
            0.9: np.array([110, 210, 310]),
        }

        result = calculate_metrics(y_true, y_pred, quantile_predictions=quantiles)

        assert 0.1 in result.pinball_loss
        assert 0.5 in result.pinball_loss
        assert 0.9 in result.pinball_loss
