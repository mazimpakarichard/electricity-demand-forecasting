"""MLflow experiment tracking and model registry."""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import mlflow
from mlflow.models.signature import infer_signature

from forecast_service.models.base import BaseForecaster, ForecastMetrics
from forecast_service.utils.logging import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """
    MLflow experiment tracking wrapper.

    Provides:
    - Experiment management
    - Parameter and metric logging
    - Model artifact storage
    - Model registry integration
    """

    def __init__(
        self,
        tracking_uri: str = "sqlite:///mlflow.db",
        experiment_name: str = "electricity-demand-forecast",
    ) -> None:
        """
        Initialize experiment tracker.

        Args:
            tracking_uri: MLflow tracking server URI.
            experiment_name: Name of the MLflow experiment.
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

        mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = experiment.experiment_id

        mlflow.set_experiment(experiment_name)

        logger.info(
            "Initialized ExperimentTracker",
            tracking_uri=tracking_uri,
            experiment=experiment_name,
        )

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> Generator[str, None, None]:
        """
        Start an MLflow run context.

        Args:
            run_name: Optional name for the run.
            tags: Optional tags for the run.

        Yields:
            Run ID.
        """
        with mlflow.start_run(run_name=run_name, tags=tags) as run:
            logger.info("Started MLflow run", run_id=run.info.run_id)
            yield run.info.run_id

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to current run."""
        # Flatten nested dicts
        flat_params = self._flatten_dict(params)
        mlflow.log_params(flat_params)

    def log_metrics(
        self,
        metrics: ForecastMetrics | dict[str, float],
        step: int | None = None,
    ) -> None:
        """Log metrics to current run."""
        if isinstance(metrics, ForecastMetrics):
            metric_dict = metrics.to_dict()
        else:
            metric_dict = metrics

        mlflow.log_metrics(metric_dict, step=step)

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        """Log a file or directory as an artifact."""
        mlflow.log_artifact(str(local_path), artifact_path)

    def log_figure(self, figure: Any, artifact_file: str) -> None:
        """Log a matplotlib figure."""
        mlflow.log_figure(figure, artifact_file)

    def log_model(
        self,
        model: BaseForecaster,
        artifact_path: str = "model",
        register_name: str | None = None,
        input_example: Any = None,
    ) -> None:
        """
        Log a forecaster model.

        Args:
            model: Trained forecaster model.
            artifact_path: Path within artifacts to store model.
            register_name: If provided, register model with this name.
            input_example: Example input for model signature inference.
        """
        # Save model to temporary file
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            model.save(model_path)

            # Log as artifact
            mlflow.log_artifact(str(model_path), artifact_path)

        # Log model metadata
        mlflow.log_param("model_name", model.name)
        mlflow.log_param("model_quantiles", str(model.quantiles))

        if register_name:
            self._register_model(artifact_path, register_name)

        logger.info("Logged model", name=model.name, path=artifact_path)

    def _register_model(self, artifact_path: str, name: str) -> None:
        """Register a model in the model registry."""
        run = mlflow.active_run()
        if run is None:
            logger.warning("No active run, cannot register model")
            return

        model_uri = f"runs:/{run.info.run_id}/{artifact_path}"

        try:
            result = mlflow.register_model(model_uri, name)
            logger.info(
                "Registered model",
                name=name,
                version=result.version,
            )
        except Exception as e:
            logger.error("Failed to register model", error=str(e))

    def log_cv_results(
        self,
        cv_results: Any,
        model_name: str,
    ) -> None:
        """Log cross-validation results."""
        mlflow.log_param("cv_model", model_name)
        mlflow.log_param("cv_n_folds", len(cv_results.fold_metrics))

        # Log mean metrics
        for name, value in cv_results.mean_metrics.items():
            mlflow.log_metric(f"cv_mean_{name}", value)

        # Log std metrics
        for name, value in cv_results.std_metrics.items():
            mlflow.log_metric(f"cv_std_{name}", value)

        # Save detailed results as artifact
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            cv_results.to_dataframe().to_csv(f.name, index=False)
            mlflow.log_artifact(f.name, "cv_results")

    def get_best_run(
        self,
        metric: str = "cv_mean_mape",
        ascending: bool = True,
    ) -> dict[str, Any] | None:
        """
        Get the best run based on a metric.

        Args:
            metric: Metric to optimize.
            ascending: If True, lower is better.

        Returns:
            Dict with run info, or None if no runs found.
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1,
        )

        if len(runs) == 0:
            return None

        best_run = runs.iloc[0]
        return {
            "run_id": best_run["run_id"],
            "metrics": {
                k.replace("metrics.", ""): v
                for k, v in best_run.items()
                if k.startswith("metrics.")
            },
            "params": {
                k.replace("params.", ""): v
                for k, v in best_run.items()
                if k.startswith("params.")
            },
        }

    def load_model(self, run_id: str, artifact_path: str = "model") -> Path:
        """
        Download model artifacts from a run.

        Args:
            run_id: MLflow run ID.
            artifact_path: Path to model artifacts.

        Returns:
            Local path to downloaded artifacts.
        """
        return Path(mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
        ))

    @staticmethod
    def _flatten_dict(d: dict[str, Any], parent_key: str = "") -> dict[str, Any]:
        """Flatten nested dictionary."""
        items: list[tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(ExperimentTracker._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, str(v) if not isinstance(v, (int, float, bool)) else v))
        return dict(items)
