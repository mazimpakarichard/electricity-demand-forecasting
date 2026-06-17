"""Training pipeline for electricity demand forecasting."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from forecast_service.data.synthetic import SyntheticDataGenerator
from forecast_service.features.engineering import FeatureConfig, FeatureEngineer
from forecast_service.models.base import BaseForecaster, CrossValidationResult
from forecast_service.models.baselines import SARIMAForecaster, SeasonalNaive
from forecast_service.models.lightgbm_model import LightGBMForecaster
from forecast_service.training.experiment import ExperimentTracker
from forecast_service.utils.logging import get_logger

# Optional PyTorch imports
try:
    from forecast_service.models.pytorch_model import LSTMForecaster, SequenceConfig, TCNForecaster

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    LSTMForecaster = None  # type: ignore[misc, assignment]
    TCNForecaster = None  # type: ignore[misc, assignment]

    @dataclass
    class SequenceConfig:  # type: ignore[no-redef]
        """Placeholder config when PyTorch not available."""

        epochs: int = 30


logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for the training pipeline."""

    # Data
    use_synthetic: bool = True
    synthetic_start: str = "2015-01-01"
    synthetic_end: str = "2018-12-31"
    train_end_date: str = "2017-12-31"

    # Features
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)

    # Models to train
    train_seasonal_naive: bool = True
    train_sarima: bool = True
    train_lightgbm: bool = True
    train_lstm: bool = True
    train_tcn: bool = False  # Optional, slower

    # Cross-validation
    n_cv_folds: int = 3
    cv_test_size: int = 168  # 1 week
    cv_gap: int = 24  # 1 day

    # Model params
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    lightgbm_params: dict[str, Any] = field(default_factory=dict)
    lstm_config: SequenceConfig = field(default_factory=lambda: SequenceConfig(epochs=30))

    # Output
    models_dir: Path = field(default_factory=lambda: Path("models"))
    results_dir: Path = field(default_factory=lambda: Path("results"))


class TrainingPipeline:
    """
    End-to-end training pipeline for demand forecasting.

    Orchestrates:
    1. Data loading/generation
    2. Feature engineering
    3. Model training with CV
    4. Experiment tracking
    5. Model comparison and selection
    """

    def __init__(
        self,
        config: TrainingConfig | None = None,
        experiment_tracker: ExperimentTracker | None = None,
    ) -> None:
        """
        Initialize training pipeline.

        Args:
            config: Training configuration.
            experiment_tracker: MLflow experiment tracker.
        """
        self.config = config or TrainingConfig()
        self.tracker = experiment_tracker

        # Create output directories
        self.config.models_dir.mkdir(parents=True, exist_ok=True)
        self.config.results_dir.mkdir(parents=True, exist_ok=True)

        self.data: pd.DataFrame | None = None
        self.features: pd.DataFrame | None = None
        self.feature_engineer: FeatureEngineer | None = None
        self.results: dict[str, CrossValidationResult] = {}
        self.best_model: BaseForecaster | None = None

    def run(self) -> dict[str, Any]:
        """
        Run the full training pipeline.

        Returns:
            Dictionary with training results and metrics.
        """
        logger.info("Starting training pipeline")

        # Step 1: Load or generate data
        self._load_data()

        # Step 2: Engineer features
        self._engineer_features()

        # Step 3: Train and evaluate models
        self._train_models()

        # Step 4: Compare and select best model
        comparison = self._compare_models()

        # Step 5: Save results
        self._save_results(comparison)

        logger.info("Training pipeline completed")

        return {
            "comparison": comparison,
            "best_model": self.best_model.name if self.best_model else None,
            "results": {name: result.mean_metrics for name, result in self.results.items()},
        }

    def _load_data(self) -> None:
        """Load or generate training data."""
        if self.config.use_synthetic:
            logger.info("Generating synthetic data")
            generator = SyntheticDataGenerator(seed=42)
            self.data = generator.generate(
                start_date=self.config.synthetic_start,
                end_date=self.config.synthetic_end,
                include_temperature=True,
            )
        else:
            # Try to load real data
            from forecast_service.data.loader import DataLoader

            try:
                loader = DataLoader()
                self.data = loader.load()
            except FileNotFoundError:
                logger.warning("Real data not found, falling back to synthetic")
                generator = SyntheticDataGenerator(seed=42)
                self.data = generator.generate()

        logger.info(
            "Data loaded",
            rows=len(self.data),
            start=str(self.data.index.min()),
            end=str(self.data.index.max()),
        )

    def _engineer_features(self) -> None:
        """Generate features from raw data."""
        if self.data is None:
            raise ValueError("Data must be loaded first")

        self.feature_engineer = FeatureEngineer(config=self.config.feature_config)
        self.features = self.feature_engineer.transform(self.data, target_col="load_mw")

        logger.info(
            "Features engineered",
            n_features=len(self.feature_engineer.feature_names_),
            n_rows=len(self.features),
        )

    def _train_models(self) -> None:
        """Train all configured models."""
        if self.features is None:
            raise ValueError("Features must be engineered first")

        # Split data
        train_end = pd.Timestamp(self.config.train_end_date)
        train_data = self.features[self.features.index <= train_end]

        feature_cols = [c for c in train_data.columns if c != "load_mw"]

        # Train each model type
        if self.config.train_seasonal_naive:
            self._train_model(
                "SeasonalNaive",
                SeasonalNaive(quantiles=self.config.quantiles),
                train_data,
                feature_cols,
            )

        if self.config.train_sarima:
            self._train_model(
                "SARIMA",
                SARIMAForecaster(quantiles=self.config.quantiles),
                train_data,
                feature_cols,
            )

        if self.config.train_lightgbm:
            self._train_model(
                "LightGBM",
                LightGBMForecaster(
                    quantiles=self.config.quantiles,
                    params=self.config.lightgbm_params,
                ),
                train_data,
                feature_cols,
            )

        if self.config.train_lstm and PYTORCH_AVAILABLE and LSTMForecaster is not None:
            self._train_model(
                "LSTM",
                LSTMForecaster(
                    config=self.config.lstm_config,
                    quantiles=self.config.quantiles,
                ),
                train_data,
                feature_cols,
            )
        elif self.config.train_lstm:
            logger.warning("LSTM training requested but PyTorch not available")

        if self.config.train_tcn and PYTORCH_AVAILABLE and TCNForecaster is not None:
            self._train_model(
                "TCN",
                TCNForecaster(
                    config=self.config.lstm_config,
                    quantiles=self.config.quantiles,
                ),
                train_data,
                feature_cols,
            )
        elif self.config.train_tcn:
            logger.warning("TCN training requested but PyTorch not available")

    def _train_model(
        self,
        name: str,
        model: BaseForecaster,
        train_data: pd.DataFrame,
        feature_cols: list[str],
    ) -> None:
        """Train a single model with cross-validation."""
        logger.info(f"Training {name}")

        if self.tracker:
            with self.tracker.start_run(run_name=name, tags={"model_type": name}):
                cv_result = model.cross_validate(
                    train_data,
                    target_col="load_mw",
                    n_folds=self.config.n_cv_folds,
                    test_size=self.config.cv_test_size,
                    gap=self.config.cv_gap,
                )

                self.tracker.log_cv_results(cv_result, name)
                self.tracker.log_params(
                    {
                        "quantiles": self.config.quantiles,
                        "n_features": len(feature_cols),
                        "train_size": len(train_data),
                    }
                )

                # Save model
                model_path = self.config.models_dir / f"{name.lower()}.pkl"
                model.save(model_path)
                self.tracker.log_artifact(str(model_path), "model")
        else:
            cv_result = model.cross_validate(
                train_data,
                target_col="load_mw",
                n_folds=self.config.n_cv_folds,
                test_size=self.config.cv_test_size,
                gap=self.config.cv_gap,
            )

            model_path = self.config.models_dir / f"{name.lower()}.pkl"
            model.save(model_path)

        self.results[name] = cv_result

        logger.info(
            f"{name} training complete",
            mape=f"{cv_result.mean_metrics.get('mape', 0):.2f}%",
            rmse=f"{cv_result.mean_metrics.get('rmse', 0):.0f}",
        )

    def _compare_models(self) -> pd.DataFrame:
        """Compare all trained models."""
        records = []

        for name, result in self.results.items():
            record = {
                "model": name,
                **{f"mean_{k}": v for k, v in result.mean_metrics.items()},
                **{f"std_{k}": v for k, v in result.std_metrics.items()},
            }
            records.append(record)

        comparison = pd.DataFrame(records)

        # Sort by MAPE
        if "mean_mape" in comparison.columns:
            comparison = comparison.sort_values("mean_mape")

            # Select best model
            best_name = comparison.iloc[0]["model"]
            best_path = self.config.models_dir / f"{best_name.lower()}.pkl"

            # Load best model
            if best_name == "SeasonalNaive":
                self.best_model = SeasonalNaive()
            elif best_name == "SARIMA":
                self.best_model = SARIMAForecaster()
            elif best_name == "LightGBM":
                self.best_model = LightGBMForecaster()
            elif best_name == "LSTM":
                self.best_model = LSTMForecaster()
            elif best_name == "TCN":
                self.best_model = TCNForecaster()

            if self.best_model:
                self.best_model.load(best_path)

        return comparison

    def _save_results(self, comparison: pd.DataFrame) -> None:
        """Save comparison results."""
        # Save comparison table
        comparison_path = self.config.results_dir / "model_comparison.csv"
        comparison.to_csv(comparison_path, index=False)

        logger.info("Results saved", path=str(comparison_path))

        # Print comparison
        print("\n" + "=" * 60)
        print("MODEL COMPARISON RESULTS")
        print("=" * 60)
        print(comparison.to_string(index=False))
        print("=" * 60 + "\n")


def run_training(
    config: TrainingConfig | None = None,
    use_mlflow: bool = True,
) -> dict[str, Any]:
    """
    Convenience function to run training pipeline.

    Args:
        config: Training configuration.
        use_mlflow: Whether to use MLflow tracking.

    Returns:
        Training results.
    """
    tracker = None
    if use_mlflow:
        tracker = ExperimentTracker()

    pipeline = TrainingPipeline(config=config, experiment_tracker=tracker)
    return pipeline.run()
