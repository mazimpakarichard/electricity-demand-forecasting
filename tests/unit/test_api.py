"""Tests for FastAPI endpoints."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from forecast_service.api.app import ForecastService, app
from forecast_service.api.schemas import PredictRequest


@pytest.fixture
def client() -> TestClient:
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_service() -> MagicMock:
    """Create mock forecast service."""
    service = MagicMock(spec=ForecastService)
    service.model = MagicMock()
    service.model.name = "TestModel"
    service.model.is_fitted = True
    service.model.quantiles = [0.1, 0.5, 0.9]
    service.recent_data = MagicMock()
    service.feature_engineer = MagicMock()
    service.data_source = "SYNTHETIC"
    return service


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self, client: TestClient) -> None:
        """Test health check returns 200."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "model_loaded" in data
        assert "timestamp" in data

    def test_health_check_response_schema(self, client: TestClient) -> None:
        """Test health response matches schema."""
        response = client.get("/health")
        data = response.json()

        assert isinstance(data["status"], str)
        assert isinstance(data["version"], str)
        assert isinstance(data["model_loaded"], bool)


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root(self, client: TestClient) -> None:
        """Test root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "docs" in data


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def test_predict_without_model(self, client: TestClient) -> None:
        """Test predict fails gracefully without model."""
        # Service may not have a model loaded
        response = client.post(
            "/predict",
            json={"horizon": 24},
        )

        # Should return 503 if no model loaded
        assert response.status_code in [200, 503]

    def test_predict_request_validation(self, client: TestClient) -> None:
        """Test request validation."""
        # Invalid horizon
        response = client.post(
            "/predict",
            json={"horizon": 0},  # Must be >= 1
        )
        assert response.status_code == 422

        # Horizon too large
        response = client.post(
            "/predict",
            json={"horizon": 1000},  # Max is 168
        )
        assert response.status_code == 422


class TestPredictRequest:
    """Tests for PredictRequest schema."""

    def test_default_values(self) -> None:
        """Test default values."""
        request = PredictRequest()

        assert request.horizon == 24
        assert request.quantiles is None
        assert request.start_time is None

    def test_custom_values(self) -> None:
        """Test custom values."""
        request = PredictRequest(
            horizon=48,
            quantiles=[0.1, 0.9],
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        assert request.horizon == 48
        assert request.quantiles == [0.1, 0.9]
        assert request.start_time is not None


class TestModelEndpoint:
    """Tests for /model endpoint."""

    def test_model_info_without_model(self, client: TestClient) -> None:
        """Test model info when no model loaded."""
        response = client.get("/model")

        # Should return 404 if no model
        assert response.status_code in [200, 404]


class TestForecastService:
    """Tests for ForecastService class."""

    def test_init(self) -> None:
        """Test service initialization."""
        service = ForecastService()

        assert service.model is None
        assert service.feature_engineer is None
        assert service.recent_data is None

    def test_initialize_data_synthetic(self) -> None:
        """Test synthetic data initialization."""
        service = ForecastService()
        service.initialize_data(use_synthetic=True)

        assert service.recent_data is not None
        assert service.feature_engineer is not None
        assert service.data_source == "SYNTHETIC"
        assert len(service.recent_data) > 0
