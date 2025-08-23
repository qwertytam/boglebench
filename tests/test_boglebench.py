"""
Tests for BogleBenchAnalyzer core functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from boglebench.core.portfolio import BogleBenchAnalyzer, PerformanceResults
from boglebench.utils.config import ConfigManager


class TestBogleBenchAnalyzer:
    """Test suite for BogleBenchAnalyzer."""

    @pytest.fixture
    def sample_transactions(self):
        """Create sample transaction data for testing."""
        return pd.DataFrame(
            {
                "date": [
                    "2023-01-15",
                    "2023-02-15",
                    "2023-03-15",
                    "2023-04-15",
                ],
                "ticker": ["AAPL", "MSFT", "AAPL", "SPY"],
                "transaction_type": ["BUY", "BUY", "BUY", "BUY"],
                "shares": [100, 50, 50, 25],
                "price_per_share": [150.50, 240.25, 155.75, 380.00],
                "account": ["Schwab", "Fidelity", "Schwab", "Personal"],
            }
        )

    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create config
            config = ConfigManager()
            config.config["data"]["base_path"] = str(config_dir)

            # Create directories
            (config_dir / "transactions").mkdir()
            (config_dir / "market_data").mkdir()
            (config_dir / "output").mkdir()

            yield config

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = BogleBenchAnalyzer()
        assert analyzer.config is not None
        assert analyzer.transactions is None
        assert analyzer.market_data == {}
        assert analyzer.portfolio_history is None

    def test_load_transactions_success(self, temp_config, sample_transactions):
        """Test successful transaction loading."""
        # Create temporary CSV file
        csv_path = temp_config.get_data_path("transactions/test.csv")
        sample_transactions.to_csv(csv_path, index=False)

        analyzer = BogleBenchAnalyzer()
        analyzer.config = temp_config

        result = analyzer.load_transactions(str(csv_path))

        assert len(result) == 4
        assert analyzer.transactions is not None
        assert "total_value" in result.columns
        assert "account" in result.columns
        assert result["ticker"].tolist() == ["AAPL", "MSFT", "AAPL", "SPY"]

    def test_load_transactions_missing_file(self, temp_config):
        """Test loading transactions with missing file."""
        analyzer = BogleBenchAnalyzer()
        analyzer.config = temp_config

        with pytest.raises(FileNotFoundError):
            analyzer.load_transactions("/nonexistent/file.csv")

    def test_load_transactions_missing_columns(self, temp_config):
        """Test loading transactions with missing required columns."""
        # Create CSV with missing columns
        bad_data = pd.DataFrame(
            {
                "date": ["2023-01-15"],
                "ticker": ["AAPL"],
                # Missing: transaction_type, shares, price_per_share
            }
        )

        csv_path = temp_config.get_data_path("transactions/bad.csv")
        bad_data.to_csv(csv_path, index=False)

        analyzer = BogleBenchAnalyzer()
        analyzer.config = temp_config

        with pytest.raises(ValueError, match="Missing required columns"):
            analyzer.load_transactions(str(csv_path))

    def test_clean_transaction_data(self, temp_config):
        """Test transaction data cleaning with valid ISO8601 dates."""
        analyzer = BogleBenchAnalyzer()
        analyzer.config = temp_config

        # Test data with valid ISO8601 dates and other cleaning needs
        clean_data = pd.DataFrame(
            {
                "date": [
                    "2023-01-15",
                    "2023-01-16",
                    "2023-01-17",
                ],  # All ISO8601
                "ticker": [" aapl ", "MSFT", "spy "],
                "transaction_type": ["buy", "SELL", "BUY"],
                "shares": [100, 50, 25],
                "price_per_share": [150.50, 240.25, 380.00],
                "account": [" schwab ", "fidelity", "PERSONAL"],
            }
        )

        cleaned = analyzer._clean_transaction_data(clean_data)

        # Check cleaning results
        assert cleaned["ticker"].tolist() == ["AAPL", "MSFT", "SPY"]
        assert cleaned["transaction_type"].tolist() == ["BUY", "SELL", "BUY"]
        assert cleaned["account"].tolist() == ["Schwab", "Fidelity", "Personal"]
        assert cleaned.loc[1, "shares"] == -50  # SELL should be negative
        assert "total_value" in cleaned.columns
        assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])

    def test_clean_transaction_data_invalid_date_format(self, temp_config):
        """Test that non-ISO8601 date formats raise an error."""
        analyzer = BogleBenchAnalyzer()
        analyzer.config = temp_config

        # Test data with invalid date format
        invalid_date_data = pd.DataFrame(
            {
                "date": [
                    "2023-01-15",
                    "01/16/2023",
                    "2023-01-17",
                ],  # Mixed formats
                "ticker": ["AAPL", "MSFT", "SPY"],
                "transaction_type": ["BUY", "BUY", "BUY"],
                "shares": [100, 50, 25],
                "price_per_share": [150.50, 240.25, 380.00],
                "account": ["Schwab", "Fidelity", "Personal"],
            }
        )

        with pytest.raises(ValueError, match="is not in ISO8601 format"):
            analyzer._clean_transaction_data(invalid_date_data)

    def test_clean_transaction_data_various_invalid_formats(self, temp_config):
        """Test various invalid date formats."""
        analyzer = BogleBenchAnalyzer()
        analyzer.config = temp_config

        invalid_formats = [
            ["01/15/2023"],  # MM/DD/YYYY
            ["15-01-2023"],  # DD-MM-YYYY
            ["Jan 15, 2023"],  # Month name
            ["2023/01/15"],  # YYYY/MM/DD
            ["20230115"],  # YYYYMMDD
        ]

        for invalid_date in invalid_formats:
            invalid_data = pd.DataFrame(
                {
                    "date": invalid_date,
                    "ticker": ["AAPL"],
                    "transaction_type": ["BUY"],
                    "shares": [100],
                    "price_per_share": [150.50],
                    "account": ["Test"],
                }
            )

            with pytest.raises(ValueError, match="is not in ISO8601 format"):
                analyzer._clean_transaction_data(invalid_data)

    def test_account_column_backward_compatibility(self, temp_config):
        """Test that missing account column gets added automatically."""
        analyzer = BogleBenchAnalyzer()
        analyzer.config = temp_config

        # Create data without account column
        data_without_account = pd.DataFrame(
            {
                "date": ["2023-01-15"],
                "ticker": ["AAPL"],
                "transaction_type": ["BUY"],
                "shares": [100],
                "price_per_share": [150.50],
            }
        )

        cleaned = analyzer._clean_transaction_data(data_without_account)

        assert "account" in cleaned.columns
        assert cleaned["account"].iloc[0] == "Default"


# Simple integration test
def test_basic_workflow():
    """Test that the basic workflow doesn't crash."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create simple transaction data
        transactions = pd.DataFrame(
            {
                "date": ["2023-01-15", "2023-02-15"],
                "ticker": ["AAPL", "MSFT"],
                "transaction_type": ["BUY", "BUY"],
                "shares": [100, 50],
                "price_per_share": [150.50, 240.25],
                "account": ["Test", "Test"],
            }
        )

        # Save transactions
        csv_path = Path(temp_dir) / "transactions.csv"
        transactions.to_csv(csv_path, index=False)

        # Create analyzer
        analyzer = BogleBenchAnalyzer()

        # Test loading transactions
        loaded_transactions = analyzer.load_transactions(str(csv_path))

        assert len(loaded_transactions) == 2
        assert "total_value" in loaded_transactions.columns
        assert "account" in loaded_transactions.columns
