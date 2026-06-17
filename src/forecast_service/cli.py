"""Command-line interface for the Forecast Service."""

from pathlib import Path
from typing import Optional

import typer

from forecast_service._version import __version__

app = typer.Typer(
    name="forecast-cli",
    help="Electricity Demand Forecasting CLI",
    add_completion=False,
)


@app.command()
def train(
    use_synthetic: bool = typer.Option(
        True, "--synthetic/--real", help="Use synthetic or real data"
    ),
    models_dir: Path = typer.Option(
        Path("models"), "--models-dir", "-m", help="Directory to save models"
    ),
    results_dir: Path = typer.Option(
        Path("results"), "--results-dir", "-r", help="Directory to save results"
    ),
    no_mlflow: bool = typer.Option(
        False, "--no-mlflow", help="Disable MLflow tracking"
    ),
    train_lstm: bool = typer.Option(
        False, "--lstm", help="Train LSTM model (slower)"
    ),
) -> None:
    """Train forecasting models."""
    from forecast_service.training.pipeline import TrainingConfig, TrainingPipeline
    from forecast_service.training.experiment import ExperimentTracker
    from forecast_service.utils.logging import configure_logging

    configure_logging()

    typer.echo(f"Training models (synthetic={use_synthetic})")

    config = TrainingConfig(
        use_synthetic=use_synthetic,
        models_dir=models_dir,
        results_dir=results_dir,
        train_lstm=train_lstm,
        train_tcn=False,
    )

    tracker = None if no_mlflow else ExperimentTracker()

    pipeline = TrainingPipeline(config=config, experiment_tracker=tracker)
    results = pipeline.run()

    typer.echo("\nTraining complete!")
    typer.echo(f"Best model: {results.get('best_model', 'N/A')}")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
) -> None:
    """Start the API server."""
    import uvicorn

    typer.echo(f"Starting API server on {host}:{port}")
    uvicorn.run(
        "forecast_service.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def predict(
    horizon: int = typer.Option(24, "--horizon", "-n", help="Forecast horizon"),
    model_path: Optional[Path] = typer.Option(
        None, "--model", "-m", help="Path to model file"
    ),
) -> None:
    """Generate a forecast."""
    from forecast_service.data.synthetic import SyntheticDataGenerator
    from forecast_service.features.engineering import FeatureEngineer
    from forecast_service.models.lightgbm_model import LightGBMForecaster
    from forecast_service.utils.logging import configure_logging

    configure_logging()

    if model_path is None:
        model_path = Path("models/lightgbm.pkl")

    if not model_path.exists():
        typer.echo(f"Model not found: {model_path}", err=True)
        raise typer.Exit(1)

    # Load model
    model = LightGBMForecaster()
    model.load(model_path)

    # Generate data and features
    generator = SyntheticDataGenerator(seed=42)
    data = generator.generate()
    engineer = FeatureEngineer()
    features = engineer.transform(data, target_col="load_mw")

    # Predict
    feature_cols = [c for c in features.columns if c != "load_mw"]
    X = features[feature_cols].tail(horizon + 168)

    result = model.predict(X, horizon=horizon)

    # Display results
    typer.echo(f"\nForecast for next {horizon} hours:")
    typer.echo("-" * 60)

    df = result.to_dataframe()
    typer.echo(df.to_string())


@app.command()
def version() -> None:
    """Show version information."""
    typer.echo(f"Forecast Service v{__version__}")


@app.command()
def explain(
    model_path: Path = typer.Option(
        Path("models/lightgbm.pkl"), "--model", "-m", help="Path to model"
    ),
    output_dir: Path = typer.Option(
        Path("results/shap"), "--output", "-o", help="Output directory"
    ),
) -> None:
    """Generate SHAP explainability report."""
    from forecast_service.data.synthetic import SyntheticDataGenerator
    from forecast_service.features.engineering import FeatureEngineer
    from forecast_service.models.lightgbm_model import LightGBMForecaster
    from forecast_service.training.explainability import SHAPExplainer
    from forecast_service.utils.logging import configure_logging

    configure_logging()

    if not model_path.exists():
        typer.echo(f"Model not found: {model_path}", err=True)
        raise typer.Exit(1)

    typer.echo("Loading model...")
    model = LightGBMForecaster()
    model.load(model_path)

    typer.echo("Generating data and features...")
    generator = SyntheticDataGenerator(seed=42)
    data = generator.generate()
    engineer = FeatureEngineer()
    features = engineer.transform(data, target_col="load_mw")
    feature_cols = [c for c in features.columns if c != "load_mw"]
    X = features[feature_cols]

    typer.echo("Running SHAP analysis...")
    explainer = SHAPExplainer(model, X_background=X.sample(n=500, random_state=42))
    report = explainer.generate_full_report(X.sample(n=1000, random_state=42), output_dir)

    typer.echo(f"\nSHAP report generated!")
    typer.echo(f"  Summary plot: {report['summary_plot']}")
    typer.echo(f"  Interpretation: {report['interpretation']}")


if __name__ == "__main__":
    app()
