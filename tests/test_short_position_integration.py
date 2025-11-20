"""
Integration test for short position handling in full portfolio workflow.

This test validates that the short position handling works correctly
when loading transactions through the BogleBenchAnalyzer.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from boglebench import BogleBenchAnalyzer, ShortPositionError


def create_test_config(
    config_path: Path, short_handling: str, transactions_file: str
):
    """Create a test configuration file."""
    config_content = f"""
data:
  base_path: "{config_path.parent}"
  transactions_file: "{transactions_file}"

validation:
  short_position_handling: "{short_handling}"

settings:
  cache_market_data: false

api:
  alpha_vantage_key: "demo"
"""
    config_path.write_text(config_content)


def create_transactions_csv(csv_path: Path, has_short: bool = False):
    """Create a test transactions CSV file."""
    if has_short:
        # Transactions with short position
        data = {
            "date": [
                "2023-01-15",
                "2023-02-15",
                "2023-03-15",
            ],
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "transaction_type": ["BUY", "BUY", "SELL"],
            "quantity": [100, 50, 200],  # Selling more than owned
            "value_per_share": [150.0, 155.0, 165.0],
            "total_value": [15000.0, 7750.0, 33000.0],
            "account": ["Test", "Test", "Test"],
        }
    else:
        # Normal transactions without short
        data = {
            "date": [
                "2023-01-15",
                "2023-02-15",
                "2023-03-15",
            ],
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "transaction_type": ["BUY", "BUY", "SELL"],
            "quantity": [100, 50, 25],  # Normal sell
            "value_per_share": [150.0, 155.0, 165.0],
            "total_value": [15000.0, 7750.0, 4125.0],
            "account": ["Test", "Test", "Test"],
        }

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)


class TestShortPositionIntegration:
    """Integration tests for short position handling."""

    def test_normal_transactions_load_successfully(self):
        """Test that normal transactions load without issues."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            config_path = tmppath / "config" / "config.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            transactions_path = tmppath / "transactions.csv"

            create_transactions_csv(transactions_path, has_short=False)
            create_test_config(
                config_path, "reject", str(transactions_path)
            )

            analyzer = BogleBenchAnalyzer(config_path=str(config_path))
            transactions = analyzer.load_transactions()

            # Should load successfully
            assert len(transactions) == 3
            # Verify SELL transaction has negative quantity
            assert transactions.iloc[2]["quantity"] == -25

    def test_reject_mode_raises_error_on_short(self):
        """Test that REJECT mode raises error when short position detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            config_path = tmppath / "config" / "config.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            transactions_path = tmppath / "transactions.csv"

            create_transactions_csv(transactions_path, has_short=True)
            create_test_config(
                config_path, "reject", str(transactions_path)
            )

            analyzer = BogleBenchAnalyzer(config_path=str(config_path))

            # Should raise ShortPositionError
            with pytest.raises(ShortPositionError) as exc_info:
                analyzer.load_transactions()

            error = exc_info.value
            assert error.symbol == "AAPL"
            assert error.account == "Test"
            assert error.current_position == 150.0
            assert error.transaction_quantity == -200.0

    def test_cap_mode_adjusts_short_position(self):
        """Test that CAP mode adjusts transaction when short detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            config_path = tmppath / "config" / "config.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            transactions_path = tmppath / "transactions.csv"

            create_transactions_csv(transactions_path, has_short=True)
            create_test_config(config_path, "cap", str(transactions_path))

            analyzer = BogleBenchAnalyzer(config_path=str(config_path))
            transactions = analyzer.load_transactions()

            # Should load successfully with adjusted quantity
            assert len(transactions) == 3
            # Third transaction should be capped to -150 (all available shares)
            assert transactions.iloc[2]["quantity"] == -150.0
            # Total value should be adjusted
            assert transactions.iloc[2]["total_value"] == -150.0 * 165.0

    def test_default_mode_is_reject(self):
        """Test that default mode is REJECT when not specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            config_path = tmppath / "config" / "config.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            transactions_path = tmppath / "transactions.csv"

            create_transactions_csv(transactions_path, has_short=True)
            
            # Create config without short_position_handling setting
            config_content = f"""
data:
  base_path: "{tmppath}"
  transactions_file: "{transactions_path}"

settings:
  cache_market_data: false

api:
  alpha_vantage_key: "demo"
"""
            config_path.write_text(config_content)

            analyzer = BogleBenchAnalyzer(config_path=str(config_path))

            # Should default to REJECT mode and raise error
            with pytest.raises(ShortPositionError):
                analyzer.load_transactions()

    def test_ignore_mode_allows_short_position(self):
        """Test that IGNORE mode allows short position with warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            config_path = tmppath / "config" / "config.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            transactions_path = tmppath / "transactions.csv"

            create_transactions_csv(transactions_path, has_short=True)
            create_test_config(config_path, "ignore", str(transactions_path))

            analyzer = BogleBenchAnalyzer(config_path=str(config_path))
            transactions = analyzer.load_transactions()

            # Should load successfully without adjustment
            assert len(transactions) == 3
            # Third transaction should remain as-is (short position allowed)
            assert transactions.iloc[2]["quantity"] == -200.0
            # Total value should remain unchanged
            assert transactions.iloc[2]["total_value"] == -33000.0
