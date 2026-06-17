"""
PyTorch sequence models for electricity demand forecasting.

Includes:
- LSTMForecaster: LSTM-based sequence model
- TCNForecaster: Temporal Convolutional Network
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from forecast_service.models.base import BaseForecaster, ForecastResult
from forecast_service.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SequenceConfig:
    """Configuration for sequence models."""

    sequence_length: int = 168  # 1 week lookback
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 50
    early_stopping_patience: int = 10


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series sequences."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: int,
    ) -> None:
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.X) - self.sequence_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        X_seq = self.X[idx : idx + self.sequence_length]
        y_val = self.y[idx + self.sequence_length]
        return X_seq, y_val


class QuantileLoss(nn.Module):
    """Quantile loss (pinball loss) for probabilistic forecasting."""

    def __init__(self, quantiles: list[float]) -> None:
        super().__init__()
        self.quantiles = quantiles

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = targets - predictions[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors))
        return torch.mean(torch.stack(losses))


class LSTMNetwork(nn.Module):
    """LSTM network for sequence forecasting."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM output shape: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        # Take the last timestep
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)


class TCNBlock(nn.Module):
    """Temporal Convolutional Network block with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, seq_len)
        residual = x

        out = self.conv1(x)
        out = out[:, :, : x.size(2)]  # Causal: trim future
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[:, :, : x.size(2)]  # Causal: trim future
        out = self.relu(out)
        out = self.dropout(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        return self.relu(out + residual)


class TCNNetwork(nn.Module):
    """Temporal Convolutional Network for sequence forecasting."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        layers = []
        for i in range(num_layers):
            in_ch = input_size if i == 0 else hidden_size
            dilation = 2**i
            layers.append(TCNBlock(in_ch, hidden_size, kernel_size, dilation, dropout))

        self.tcn = nn.Sequential(*layers)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, seq_len, features)
        # TCN expects: (batch, features, seq_len)
        x = x.transpose(1, 2)

        out = self.tcn(x)

        # Take the last timestep
        last_out = out[:, :, -1]
        return self.fc(last_out)


class LSTMForecaster(BaseForecaster):
    """
    LSTM-based forecaster with quantile regression.

    Uses a multi-output LSTM to predict multiple quantiles simultaneously.
    """

    def __init__(
        self,
        config: SequenceConfig | None = None,
        quantiles: list[float] | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(name="LSTM", quantiles=quantiles)
        self.config = config or SequenceConfig()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: LSTMNetwork | None = None
        self.feature_names: list[str] = []
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        self._target_mean: float = 0.0
        self._target_std: float = 1.0

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "LSTMForecaster":
        """Train LSTM model."""
        self.feature_names = list(X.columns)
        n_features = len(self.feature_names)
        n_quantiles = len(self.quantiles)

        # Normalize features
        self._feature_mean = X.values.mean(axis=0)
        self._feature_std = X.values.std(axis=0) + 1e-8
        X_norm = (X.values - self._feature_mean) / self._feature_std

        self._target_mean = float(y.mean())
        self._target_std = float(y.std()) + 1e-8
        y_norm = (y.values - self._target_mean) / self._target_std

        # Create dataset
        dataset = TimeSeriesDataset(X_norm, y_norm, self.config.sequence_length)

        # Split for validation
        val_size = max(168, int(len(dataset) * 0.1))
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)

        # Initialize model
        self.model = LSTMNetwork(
            input_size=n_features,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            output_size=n_quantiles,
            dropout=self.config.dropout,
        ).to(self.device)

        criterion = QuantileLoss(self.quantiles)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        logger.info(
            "Training LSTM model",
            n_features=n_features,
            device=str(self.device),
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch.unsqueeze(1).expand(-1, n_quantiles))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    output = self.model(X_batch)
                    loss = criterion(
                        output, y_batch.unsqueeze(1).expand(-1, n_quantiles)
                    )
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs}",
                    train_loss=f"{train_loss:.4f}",
                    val_loss=f"{val_loss:.4f}",
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        self.is_fitted = True
        return self

    def predict(
        self,
        X: pd.DataFrame,
        horizon: int = 24,
    ) -> ForecastResult:
        """Generate LSTM forecasts."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")

        self.model.eval()
        timestamps = X.index[:horizon]

        # Normalize features
        X_norm = (X.values - self._feature_mean) / self._feature_std

        predictions = []
        with torch.no_grad():
            for i in range(horizon):
                start_idx = max(0, i - self.config.sequence_length)
                if i < self.config.sequence_length:
                    # Pad with zeros if not enough history
                    pad_size = self.config.sequence_length - i
                    X_seq = np.vstack(
                        [np.zeros((pad_size, X_norm.shape[1])), X_norm[: i + 1]]
                    )
                else:
                    X_seq = X_norm[start_idx : i + 1]

                X_tensor = torch.FloatTensor(X_seq).unsqueeze(0).to(self.device)
                output = self.model(X_tensor)
                predictions.append(output.cpu().numpy()[0])

        predictions = np.array(predictions)

        # Denormalize
        predictions = predictions * self._target_std + self._target_mean

        quantile_forecasts: dict[float, npt.NDArray[np.floating[Any]]] = {}
        for i, q in enumerate(self.quantiles):
            quantile_forecasts[q] = predictions[:, i]

        point_forecast = quantile_forecasts.get(
            0.5, np.median(predictions, axis=1)
        )

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
            "config": self.config,
            "quantiles": self.quantiles,
            "feature_names": self.feature_names,
            "model_state_dict": self.model.state_dict() if self.model else None,
            "feature_mean": self._feature_mean,
            "feature_std": self._feature_std,
            "target_mean": self._target_mean,
            "target_std": self._target_std,
            "is_fitted": self.is_fitted,
        }

        torch.save(state, path)
        logger.info("Saved LSTM model", path=str(path))

    def load(self, path: Path | str) -> "LSTMForecaster":
        """Load model from disk."""
        state = torch.load(path, map_location=self.device, weights_only=False)

        self.config = state["config"]
        self.quantiles = state["quantiles"]
        self.feature_names = state["feature_names"]
        self._feature_mean = state["feature_mean"]
        self._feature_std = state["feature_std"]
        self._target_mean = state["target_mean"]
        self._target_std = state["target_std"]
        self.is_fitted = state["is_fitted"]

        if state["model_state_dict"] is not None:
            self.model = LSTMNetwork(
                input_size=len(self.feature_names),
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                output_size=len(self.quantiles),
                dropout=self.config.dropout,
            ).to(self.device)
            self.model.load_state_dict(state["model_state_dict"])

        logger.info("Loaded LSTM model", path=str(path))
        return self


class TCNForecaster(BaseForecaster):
    """
    Temporal Convolutional Network forecaster with quantile regression.
    """

    def __init__(
        self,
        config: SequenceConfig | None = None,
        quantiles: list[float] | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(name="TCN", quantiles=quantiles)
        self.config = config or SequenceConfig()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: TCNNetwork | None = None
        self.feature_names: list[str] = []
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        self._target_mean: float = 0.0
        self._target_std: float = 1.0

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "TCNForecaster":
        """Train TCN model."""
        self.feature_names = list(X.columns)
        n_features = len(self.feature_names)
        n_quantiles = len(self.quantiles)

        # Normalize
        self._feature_mean = X.values.mean(axis=0)
        self._feature_std = X.values.std(axis=0) + 1e-8
        X_norm = (X.values - self._feature_mean) / self._feature_std

        self._target_mean = float(y.mean())
        self._target_std = float(y.std()) + 1e-8
        y_norm = (y.values - self._target_mean) / self._target_std

        # Create dataset
        dataset = TimeSeriesDataset(X_norm, y_norm, self.config.sequence_length)

        val_size = max(168, int(len(dataset) * 0.1))
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)

        # Initialize model
        self.model = TCNNetwork(
            input_size=n_features,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            output_size=n_quantiles,
            dropout=self.config.dropout,
        ).to(self.device)

        criterion = QuantileLoss(self.quantiles)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )

        logger.info(
            "Training TCN model",
            n_features=n_features,
            device=str(self.device),
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.epochs):
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch.unsqueeze(1).expand(-1, n_quantiles))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    output = self.model(X_batch)
                    loss = criterion(
                        output, y_batch.unsqueeze(1).expand(-1, n_quantiles)
                    )
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs}",
                    train_loss=f"{train_loss:.4f}",
                    val_loss=f"{val_loss:.4f}",
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        self.is_fitted = True
        return self

    def predict(
        self,
        X: pd.DataFrame,
        horizon: int = 24,
    ) -> ForecastResult:
        """Generate TCN forecasts."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")

        self.model.eval()
        timestamps = X.index[:horizon]

        X_norm = (X.values - self._feature_mean) / self._feature_std

        predictions = []
        with torch.no_grad():
            for i in range(horizon):
                if i < self.config.sequence_length:
                    pad_size = self.config.sequence_length - i
                    X_seq = np.vstack(
                        [np.zeros((pad_size, X_norm.shape[1])), X_norm[: i + 1]]
                    )
                else:
                    start_idx = i - self.config.sequence_length
                    X_seq = X_norm[start_idx : i + 1]

                X_tensor = torch.FloatTensor(X_seq).unsqueeze(0).to(self.device)
                output = self.model(X_tensor)
                predictions.append(output.cpu().numpy()[0])

        predictions = np.array(predictions)
        predictions = predictions * self._target_std + self._target_mean

        quantile_forecasts: dict[float, npt.NDArray[np.floating[Any]]] = {}
        for i, q in enumerate(self.quantiles):
            quantile_forecasts[q] = predictions[:, i]

        point_forecast = quantile_forecasts.get(0.5, np.median(predictions, axis=1))

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
            "config": self.config,
            "quantiles": self.quantiles,
            "feature_names": self.feature_names,
            "model_state_dict": self.model.state_dict() if self.model else None,
            "feature_mean": self._feature_mean,
            "feature_std": self._feature_std,
            "target_mean": self._target_mean,
            "target_std": self._target_std,
            "is_fitted": self.is_fitted,
        }

        torch.save(state, path)
        logger.info("Saved TCN model", path=str(path))

    def load(self, path: Path | str) -> "TCNForecaster":
        """Load model from disk."""
        state = torch.load(path, map_location=self.device, weights_only=False)

        self.config = state["config"]
        self.quantiles = state["quantiles"]
        self.feature_names = state["feature_names"]
        self._feature_mean = state["feature_mean"]
        self._feature_std = state["feature_std"]
        self._target_mean = state["target_mean"]
        self._target_std = state["target_std"]
        self.is_fitted = state["is_fitted"]

        if state["model_state_dict"] is not None:
            self.model = TCNNetwork(
                input_size=len(self.feature_names),
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                output_size=len(self.quantiles),
                dropout=self.config.dropout,
            ).to(self.device)
            self.model.load_state_dict(state["model_state_dict"])

        logger.info("Loaded TCN model", path=str(path))
        return self
