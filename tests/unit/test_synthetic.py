"""Tests for synthetic data generation."""

import pandas as pd

from forecast_service.data.synthetic import (
    SyntheticDataGenerator,
    generate_synthetic_data,
    is_synthetic,
)


class TestSyntheticDataGenerator:
    """Tests for SyntheticDataGenerator class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        generator = SyntheticDataGenerator()
        assert generator.seed == 42
        assert generator.base_load == 32000.0

    def test_init_custom(self) -> None:
        """Test custom initialization."""
        generator = SyntheticDataGenerator(seed=123, base_load=30000.0, noise_std=300.0)
        assert generator.seed == 123
        assert generator.base_load == 30000.0
        assert generator.noise_std == 300.0

    def test_generate_basic(self) -> None:
        """Test basic data generation."""
        generator = SyntheticDataGenerator(seed=42)
        df = generator.generate(start_date="2017-01-01", end_date="2017-01-07")

        assert isinstance(df, pd.DataFrame)
        assert "load_mw" in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "datetime"

    def test_generate_with_temperature(self) -> None:
        """Test generation with temperature data."""
        generator = SyntheticDataGenerator(seed=42)
        df = generator.generate(
            start_date="2017-01-01",
            end_date="2017-01-07",
            include_temperature=True,
        )

        assert "temperature_f" in df.columns
        # Temperature should be reasonable
        assert df["temperature_f"].min() > -50
        assert df["temperature_f"].max() < 150

    def test_generate_without_temperature(self) -> None:
        """Test generation without temperature data."""
        generator = SyntheticDataGenerator(seed=42)
        df = generator.generate(
            start_date="2017-01-01",
            end_date="2017-01-07",
            include_temperature=False,
        )

        assert "temperature_f" not in df.columns

    def test_generate_reproducibility(self) -> None:
        """Test that same seed produces same data."""
        gen1 = SyntheticDataGenerator(seed=42)
        gen2 = SyntheticDataGenerator(seed=42)

        df1 = gen1.generate(start_date="2017-01-01", end_date="2017-01-07")
        df2 = gen2.generate(start_date="2017-01-01", end_date="2017-01-07")

        pd.testing.assert_frame_equal(df1, df2)

    def test_generate_different_seeds(self) -> None:
        """Test that different seeds produce different data."""
        gen1 = SyntheticDataGenerator(seed=42)
        gen2 = SyntheticDataGenerator(seed=123)

        df1 = gen1.generate(start_date="2017-01-01", end_date="2017-01-07")
        df2 = gen2.generate(start_date="2017-01-01", end_date="2017-01-07")

        # Data should be different (though structure is same)
        assert not df1["load_mw"].equals(df2["load_mw"])

    def test_generate_positive_load(self) -> None:
        """Test that all load values are positive."""
        generator = SyntheticDataGenerator(seed=42)
        df = generator.generate(start_date="2017-01-01", end_date="2017-12-31")

        assert (df["load_mw"] > 0).all()

    def test_generate_hourly_frequency(self) -> None:
        """Test that data has hourly frequency."""
        generator = SyntheticDataGenerator(seed=42)
        df = generator.generate(start_date="2017-01-01", end_date="2017-01-07")

        # Check time differences
        time_diffs = df.index.to_series().diff().dropna()
        assert (time_diffs == pd.Timedelta(hours=1)).all()

    def test_synthetic_label(self) -> None:
        """Test that generated data is labeled as synthetic."""
        generator = SyntheticDataGenerator(seed=42)
        df = generator.generate(start_date="2017-01-01", end_date="2017-01-07")

        assert df.attrs.get("data_source") == "SYNTHETIC"
        assert df.attrs.get("generator_seed") == 42


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_generate_synthetic_data(self) -> None:
        """Test generate_synthetic_data function."""
        df = generate_synthetic_data(
            start_date="2017-01-01",
            end_date="2017-01-07",
            seed=42,
        )

        assert isinstance(df, pd.DataFrame)
        assert "load_mw" in df.columns

    def test_is_synthetic_true(self) -> None:
        """Test is_synthetic returns True for synthetic data."""
        df = generate_synthetic_data(start_date="2017-01-01", end_date="2017-01-07")
        assert is_synthetic(df) is True

    def test_is_synthetic_false(self) -> None:
        """Test is_synthetic returns False for non-synthetic data."""
        df = pd.DataFrame({"load_mw": [1, 2, 3]})
        assert is_synthetic(df) is False
