"""Data loading utilities for PJM electricity demand data."""

from pathlib import Path
from typing import Literal

import pandas as pd

from forecast_service.utils.logging import get_logger

logger = get_logger(__name__)

# PJM data series available on Kaggle and public sources
PJM_SERIES = Literal[
    "AEP", "COMED", "DAYTON", "DEOK", "DOM", "DUQ", "EKPC", "FE", "NI", "PJME", "PJMW"
]


class DataLoader:
    """
    Load and preprocess PJM hourly electricity consumption data.

    This loader handles:
    - Loading real PJM data from CSV files
    - Data validation and cleaning
    - Resampling to hourly frequency
    - Handling missing values

    The PJM Interconnection data contains hourly electricity consumption
    for various regions in the Eastern United States.

    Attributes:
        data_dir: Directory containing data files.
        series: PJM series to load (e.g., "PJME", "AEP").
    """

    # Expected columns in PJM data files
    DATETIME_COL = "Datetime"
    VALUE_COL_PATTERNS = ["MW", "LOAD", "DEMAND", "VALUE"]

    def __init__(self, data_dir: Path | str = "data", series: str = "PJME") -> None:
        """
        Initialize the data loader.

        Args:
            data_dir: Path to the data directory.
            series: PJM series name to load.
        """
        self.data_dir = Path(data_dir)
        self.series = series.upper()
        logger.info("Initialized DataLoader", series=self.series, data_dir=str(self.data_dir))

    def load(self) -> pd.DataFrame:
        """
        Load the PJM electricity demand data.

        Returns:
            DataFrame with datetime index and 'load_mw' column.

        Raises:
            FileNotFoundError: If the data file doesn't exist.
            ValueError: If the data format is unexpected.
        """
        file_path = self._find_data_file()
        logger.info("Loading data file", path=str(file_path))

        df = pd.read_csv(file_path)
        df = self._preprocess(df)

        logger.info(
            "Data loaded successfully",
            rows=len(df),
            start=str(df.index.min()),
            end=str(df.index.max()),
        )
        return df

    def _find_data_file(self) -> Path:
        """Find the data file for the specified series."""
        # Try common naming patterns
        patterns = [
            f"{self.series}_hourly.csv",
            f"{self.series}.csv",
            f"pjm_{self.series.lower()}_hourly.csv",
            f"{self.series.lower()}_hourly.csv",
        ]

        for pattern in patterns:
            path = self.data_dir / pattern
            if path.exists():
                return path

        # List available files
        available = list(self.data_dir.glob("*.csv"))
        raise FileNotFoundError(
            f"Data file for series '{self.series}' not found in {self.data_dir}. "
            f"Tried patterns: {patterns}. "
            f"Available files: {[f.name for f in available]}"
        )

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the raw data.

        - Parse datetime column
        - Identify and rename value column
        - Set datetime index
        - Handle missing values
        - Ensure hourly frequency
        """
        # Find datetime column (case-insensitive)
        datetime_col = self._find_column(df, ["datetime", "date", "timestamp", "time"])
        if datetime_col is None:
            raise ValueError(f"No datetime column found. Columns: {list(df.columns)}")

        # Find value column
        value_col = self._find_column(
            df, [self.series, "mw", "load", "demand", "value", "megawatts"]
        )
        if value_col is None:
            raise ValueError(f"No value column found. Columns: {list(df.columns)}")

        # Parse datetime
        df[datetime_col] = pd.to_datetime(df[datetime_col])

        # Create clean dataframe
        result = pd.DataFrame(
            {"load_mw": df[value_col].astype(float)},
            index=pd.DatetimeIndex(df[datetime_col], name="datetime"),
        )

        # Sort by datetime
        result = result.sort_index()

        # Remove duplicates (keep first)
        result = result[~result.index.duplicated(keep="first")]

        # Handle missing values
        result = self._handle_missing(result)

        # Ensure hourly frequency
        result = result.asfreq("h")

        # Final missing value handling (for any gaps from resampling)
        result = result.ffill().bfill()

        return result

    def _find_column(self, df: pd.DataFrame, patterns: list[str]) -> str | None:
        """Find a column matching any of the patterns (case-insensitive)."""
        columns_lower = {col.lower(): col for col in df.columns}
        for pattern in patterns:
            if pattern.lower() in columns_lower:
                return columns_lower[pattern.lower()]
        return None

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data."""
        missing_count = df["load_mw"].isna().sum()

        if missing_count > 0:
            logger.warning("Found missing values", count=missing_count)

            # For short gaps, interpolate
            df["load_mw"] = df["load_mw"].interpolate(method="time", limit=24)

            # For remaining gaps, forward fill then backward fill
            df["load_mw"] = df["load_mw"].ffill().bfill()

        return df

    @staticmethod
    def validate_data(df: pd.DataFrame) -> dict[str, bool | int | float]:
        """
        Validate the loaded data quality.

        Returns:
            Dictionary with validation results.
        """
        results: dict[str, bool | int | float] = {}

        # Check for required column
        results["has_load_column"] = "load_mw" in df.columns

        # Check index type
        results["has_datetime_index"] = isinstance(df.index, pd.DatetimeIndex)

        # Check for missing values
        results["missing_count"] = int(df["load_mw"].isna().sum())
        results["has_no_missing"] = results["missing_count"] == 0

        # Check for negative values (invalid for load)
        results["negative_count"] = int((df["load_mw"] < 0).sum())
        results["has_no_negative"] = results["negative_count"] == 0

        # Check for outliers (simple IQR method)
        q1 = df["load_mw"].quantile(0.25)
        q3 = df["load_mw"].quantile(0.75)
        iqr = q3 - q1
        outliers = ((df["load_mw"] < q1 - 3 * iqr) | (df["load_mw"] > q3 + 3 * iqr)).sum()
        results["outlier_count"] = int(outliers)

        # Check frequency
        if len(df) > 1:
            time_diffs = df.index.to_series().diff().dropna()
            expected_freq = pd.Timedelta(hours=1)
            results["consistent_hourly_freq"] = bool((time_diffs == expected_freq).all())
        else:
            results["consistent_hourly_freq"] = True

        # Summary stats
        results["min_load"] = float(df["load_mw"].min())
        results["max_load"] = float(df["load_mw"].max())
        results["mean_load"] = float(df["load_mw"].mean())

        return results


def load_pjm_data(
    data_dir: Path | str = "data",
    series: str = "PJME",
) -> pd.DataFrame:
    """
    Convenience function to load PJM data.

    Args:
        data_dir: Directory containing data files.
        series: PJM series to load.

    Returns:
        DataFrame with datetime index and load_mw column.
    """
    loader = DataLoader(data_dir=data_dir, series=series)
    return loader.load()
