#!/usr/bin/env python3
"""Training script for electricity demand forecasting models."""

import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from forecast_service.training.pipeline import TrainingConfig, TrainingPipeline
from forecast_service.utils.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


def main() -> None:
    """Run training pipeline."""
    logger.info("Starting training pipeline")

    config = TrainingConfig(
        use_synthetic=True,
        train_seasonal_naive=True,
        train_sarima=True,
        train_lightgbm=True,
        train_lstm=False,  # Skip LSTM for faster training
        train_tcn=False,
        n_cv_folds=3,
        models_dir=Path("models"),
        results_dir=Path("results"),
    )

    # Run without MLflow for simplicity
    pipeline = TrainingPipeline(config=config, experiment_tracker=None)
    results = pipeline.run()

    logger.info("Training complete", best_model=results.get("best_model"))

    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for model_name, metrics in results.get("results", {}).items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
