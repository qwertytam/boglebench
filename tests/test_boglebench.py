"""
Tests for BogleBenchAnalyzer core functionality.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from boglebench.core.portfolio import BogleBenchAnalyzer
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

    @pytest.fixture
    def test_data_dir(self):
        """Path to test data directory."""
        return Path(__file__).parent / "test_data"

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = BogleBenchAnalyzer()
        assert analyzer.config is not None
        assert analyzer.transactions is None
        assert not analyzer.market_data
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

    def test_load_transactions_missing_columns(
        self, test_data_dir, temp_config
    ):
        """Test loading transactions with missing required columns."""
        # Missing: transaction_type, shares, price_per_share
        csv_path = test_data_dir / "bad_data_missing_columns_pytest.csv"
        analyzer = BogleBenchAnalyzer()
        analyzer.config = temp_config

        with pytest.raises(ValueError, match="Missing required columns"):
            analyzer.load_transactions(str(csv_path))

    def test_clean_transaction_data(self, test_data_dir, temp_config):
        """Test transaction data cleaning with valid ISO8601 dates."""
        analyzer = BogleBenchAnalyzer()
        analyzer.config = temp_config

        csv_path = test_data_dir / "clean_data_with_ISO8601_dates_pytest.csv"
        cleaned = analyzer.load_transactions(csv_path)

        # Check cleaning results
        assert len(cleaned) == 4
        assert cleaned["ticker"].tolist() == ["AAPL", "MSFT", "SPY", "SPY"]
        assert cleaned["transaction_type"].tolist() == [
            "BUY",
            "BUY",
            "BUY",
            "SELL",
        ]
        assert cleaned["account"].tolist() == [
            "Schwab",
            "Personal",
            "Fidelity",
            "Fidelity",
        ]
        assert cleaned.loc[3, "shares"] == -25  # SELL should be negative
        assert "total_value" in cleaned.columns
        assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])

    def test_clean_transaction_data_invalid_date_format(
        self, test_data_dir, temp_config
    ):
        """Test that non-ISO8601 date formats raise an error."""
        analyzer = BogleBenchAnalyzer()
        analyzer.config = temp_config

        # Test data with invalid date format
        csv_path = test_data_dir / "clean_data_with_bad_dates_pytest.csv"

        with pytest.raises(ValueError, match="is not in ISO8601 format"):
            analyzer.load_transactions(csv_path)

    def test_account_column_backward_compatibility(
        self, test_data_dir, temp_config
    ):
        """Test that missing account column gets added automatically."""
        analyzer = BogleBenchAnalyzer()
        analyzer.config = temp_config

        # Test data without account column
        csv_path = test_data_dir / "data_without_account_col_pytest.csv"
        cleaned = analyzer.load_transactions(csv_path)

        assert "account" in cleaned.columns
        assert cleaned["account"].iloc[0] == "Default"

    def test_valid_transactions(self, test_data_dir, temp_config):
        """Test loading valid transactions from sample CSV."""
        analyzer = BogleBenchAnalyzer()
        analyzer.config = temp_config

        csv_file = test_data_dir / "default_simple_transactions_pytest.csv"
        result = analyzer.load_transactions(str(csv_file))

        assert len(result) == 5
        assert analyzer.transactions is not None
        assert "total_value" in result.columns
        assert "account" in result.columns
        assert result["ticker"].tolist() == [
            "AAPL",
            "MSFT",
            "SPY",
            "SPY",
            "SPY",
        ]


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
