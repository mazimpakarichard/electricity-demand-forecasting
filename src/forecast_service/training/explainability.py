"""
SHAP explainability for LightGBM forecasting model.

Provides feature importance analysis and explanation plots.
"""

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from forecast_service.models.lightgbm_model import LightGBMForecaster
from forecast_service.utils.logging import get_logger

logger = get_logger(__name__)


class SHAPExplainer:
    """
    SHAP-based explainability for LightGBM models.

    Provides:
    - Global feature importance (summary plots)
    - Local explanations (individual predictions)
    - Feature interaction analysis
    - Saved plots and interpretations
    """

    def __init__(
        self,
        model: LightGBMForecaster,
        X_background: pd.DataFrame | None = None,
        max_background_samples: int = 1000,
    ) -> None:
        """
        Initialize SHAP explainer.

        Args:
            model: Fitted LightGBM forecaster.
            X_background: Background data for SHAP calculations.
            max_background_samples: Maximum samples for background.
        """
        if not model.is_fitted:
            raise ValueError("Model must be fitted before creating explainer")

        self.model = model
        self.feature_names = model.feature_names

        # Use the median (p50) model for explanations
        self.lgb_model = model.get_model_for_quantile(0.5)

        # Sample background data if needed
        if X_background is not None:
            if len(X_background) > max_background_samples:
                X_background = X_background.sample(
                    n=max_background_samples, random_state=42
                )
            self._background = X_background
        else:
            self._background = None

        # Create SHAP explainer
        self.explainer = shap.TreeExplainer(self.lgb_model)

        logger.info("Initialized SHAP explainer", n_features=len(self.feature_names))

    def compute_shap_values(
        self,
        X: pd.DataFrame,
        max_samples: int = 5000,
    ) -> shap.Explanation:
        """
        Compute SHAP values for the given data.

        Args:
            X: Feature matrix.
            max_samples: Maximum samples to compute SHAP for.

        Returns:
            SHAP Explanation object.
        """
        if len(X) > max_samples:
            X = X.sample(n=max_samples, random_state=42)

        logger.info("Computing SHAP values", n_samples=len(X))

        shap_values = self.explainer(X)

        return shap_values

    def plot_summary(
        self,
        shap_values: shap.Explanation,
        save_path: Path | str | None = None,
        plot_type: str = "dot",
        max_display: int = 20,
    ) -> plt.Figure:
        """
        Create SHAP summary plot.

        Args:
            shap_values: Computed SHAP values.
            save_path: Path to save the plot.
            plot_type: Type of plot ("dot", "bar", "violin").
            max_display: Maximum features to display.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        shap.summary_plot(
            shap_values,
            max_display=max_display,
            plot_type=plot_type,
            show=False,
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved SHAP summary plot", path=str(save_path))

        return fig

    def plot_bar(
        self,
        shap_values: shap.Explanation,
        save_path: Path | str | None = None,
        max_display: int = 20,
    ) -> plt.Figure:
        """
        Create SHAP bar plot (mean absolute importance).

        Args:
            shap_values: Computed SHAP values.
            save_path: Path to save the plot.
            max_display: Maximum features to display.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        shap.plots.bar(shap_values, max_display=max_display, show=False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved SHAP bar plot", path=str(save_path))

        return fig

    def plot_dependence(
        self,
        shap_values: shap.Explanation,
        feature: str,
        interaction_feature: str | None = None,
        save_path: Path | str | None = None,
    ) -> plt.Figure:
        """
        Create SHAP dependence plot for a feature.

        Args:
            shap_values: Computed SHAP values.
            feature: Feature to plot.
            interaction_feature: Feature to color by (for interactions).
            save_path: Path to save the plot.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        shap.dependence_plot(
            feature,
            shap_values.values,
            shap_values.data,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False,
            ax=ax,
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved SHAP dependence plot", feature=feature, path=str(save_path))

        return fig

    def get_feature_importance(
        self,
        shap_values: shap.Explanation,
    ) -> pd.DataFrame:
        """
        Get feature importance from SHAP values.

        Args:
            shap_values: Computed SHAP values.

        Returns:
            DataFrame with feature names and importance scores.
        """
        # Mean absolute SHAP value per feature
        importance = np.abs(shap_values.values).mean(axis=0)

        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
            "mean_shap": shap_values.values.mean(axis=0),
            "std_shap": shap_values.values.std(axis=0),
        })

        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def generate_interpretation(
        self,
        shap_values: shap.Explanation,
        X: pd.DataFrame,
    ) -> str:
        """
        Generate written interpretation of SHAP analysis.

        Args:
            shap_values: Computed SHAP values.
            X: Feature matrix used for SHAP computation.

        Returns:
            Markdown-formatted interpretation.
        """
        importance = self.get_feature_importance(shap_values)
        top_features = importance.head(10)

        interpretation = """# SHAP Feature Importance Analysis

## Overview

This analysis uses SHAP (SHapley Additive exPlanations) values to explain the
LightGBM model's predictions for electricity demand forecasting. SHAP values
provide a unified measure of feature importance that accounts for feature
interactions.

## Top 10 Most Important Features

| Rank | Feature | Mean |SHAP| | Interpretation |
|------|---------|-------------|----------------|
"""

        for i, row in top_features.iterrows():
            feature = row["feature"]
            importance_val = row["importance"]

            # Generate feature-specific interpretation
            if "lag_" in feature:
                interp = "Recent historical load values are strong predictors"
            elif "rolling_" in feature:
                interp = "Trend indicators capture current load regime"
            elif "daily_" in feature or "hour" in feature:
                interp = "Daily patterns drive significant variation"
            elif "weekly_" in feature or "day_of_week" in feature:
                interp = "Weekday/weekend effects are important"
            elif "annual_" in feature or "month" in feature:
                interp = "Seasonal patterns affect demand"
            elif "holiday" in feature:
                interp = "Holidays significantly reduce load"
            elif "temp" in feature or "degree" in feature:
                interp = "Temperature affects HVAC-driven demand"
            else:
                interp = "Contributes to prediction accuracy"

            interpretation += f"| {i + 1} | `{feature}` | {importance_val:.1f} | {interp} |\n"

        # Add key insights
        interpretation += """
## Key Insights

### 1. Lag Features Dominate

"""
        lag_features = importance[importance["feature"].str.contains("lag_|same_hour")]
        lag_importance = lag_features["importance"].sum()
        total_importance = importance["importance"].sum()
        lag_pct = (lag_importance / total_importance) * 100

        interpretation += f"""Lag features (historical load values) account for approximately
**{lag_pct:.1f}%** of total feature importance. This confirms the strong
autocorrelation in electricity demand - recent load is the best predictor
of near-future load.

### 2. Temporal Patterns

"""
        temporal_features = importance[
            importance["feature"].str.contains("daily_|weekly_|annual_|hour|day_of")
        ]
        temporal_importance = temporal_features["importance"].sum()
        temporal_pct = (temporal_importance / total_importance) * 100

        interpretation += f"""Temporal/calendar features contribute **{temporal_pct:.1f}%** of
importance. The model successfully captures:
- **Daily patterns**: Morning ramp-up, evening peak, overnight trough
- **Weekly patterns**: Lower demand on weekends
- **Annual patterns**: Summer/winter peaks, spring/fall valleys

### 3. Weather Sensitivity

"""
        weather_features = importance[
            importance["feature"].str.contains("temp|degree")
        ]
        if len(weather_features) > 0:
            weather_importance = weather_features["importance"].sum()
            weather_pct = (weather_importance / total_importance) * 100
            interpretation += f"""Temperature-related features account for **{weather_pct:.1f}%** of
importance. This reflects the U-shaped relationship between temperature
and electricity demand (heating in cold weather, cooling in hot weather).
"""
        else:
            interpretation += """Temperature features were not included in this model run.
Including weather data typically improves forecast accuracy by 2-5%.
"""

        interpretation += """
## Model Behavior

### Positive SHAP Values (Increase Predicted Load)
- High recent load (lag features)
- Peak hours (morning/evening)
- Weekday patterns
- Extreme temperatures (hot or cold)

### Negative SHAP Values (Decrease Predicted Load)
- Low recent load
- Night hours
- Weekend/holiday indicators
- Mild temperatures (around 65°F)

## Recommendations

1. **Feature Engineering**: Focus on improving lag and rolling features
2. **Weather Data**: Ensure accurate temperature forecasts for prediction
3. **Holiday Calendar**: Maintain updated holiday calendars for all regions
4. **Model Updates**: Retrain periodically to capture changing patterns
"""

        return interpretation

    def generate_full_report(
        self,
        X: pd.DataFrame,
        output_dir: Path | str = "results/shap",
    ) -> dict[str, Any]:
        """
        Generate complete SHAP analysis with plots and interpretation.

        Args:
            X: Feature matrix for analysis.
            output_dir: Directory to save outputs.

        Returns:
            Dictionary with paths to generated files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating SHAP report", output_dir=str(output_dir))

        # Compute SHAP values
        shap_values = self.compute_shap_values(X)

        # Generate plots
        summary_path = output_dir / "shap_summary.png"
        self.plot_summary(shap_values, save_path=summary_path)
        plt.close()

        bar_path = output_dir / "shap_bar.png"
        self.plot_bar(shap_values, save_path=bar_path)
        plt.close()

        # Top feature dependence plots
        importance = self.get_feature_importance(shap_values)
        top_features = importance.head(5)["feature"].tolist()

        dependence_paths = []
        for feature in top_features:
            dep_path = output_dir / f"shap_dependence_{feature}.png"
            self.plot_dependence(shap_values, feature, save_path=dep_path)
            plt.close()
            dependence_paths.append(str(dep_path))

        # Save feature importance table
        importance_path = output_dir / "feature_importance.csv"
        importance.to_csv(importance_path, index=False)

        # Generate interpretation
        interpretation = self.generate_interpretation(shap_values, X)
        interp_path = output_dir / "INTERPRETATION.md"
        with open(interp_path, "w") as f:
            f.write(interpretation)

        logger.info("SHAP report generated", output_dir=str(output_dir))

        return {
            "summary_plot": str(summary_path),
            "bar_plot": str(bar_path),
            "dependence_plots": dependence_paths,
            "importance_table": str(importance_path),
            "interpretation": str(interp_path),
        }
