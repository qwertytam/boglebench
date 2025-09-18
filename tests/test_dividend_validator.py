"""
Tests for the DividendValidator class.
"""

import tempfile
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import pytest

from boglebench.core.dividend_validator import DividendValidator
from boglebench.core.portfolio import BogleBenchAnalyzer
from boglebench.utils.config import ConfigManager

# --- Fixtures for Test Data ---


@pytest.fixture
def sample_market_data() -> Dict[str, pd.DataFrame]:
    """
    Creates a sample market data DataFrame with dividend information for two
    tickers:
    - VTI: Pays a $0.50 dividend on 2023-03-25
    - BND: Pays a $0.20 dividend on 2023-03-20
    """
    dates = pd.to_datetime(
        ["2023-03-20", "2023-03-21", "2023-03-25", "2023-03-26"], utc=True
    )

    data = {
        "VTI": pd.DataFrame(
            {
                "date": dates,
                "close": [100, 101, 102, 103],
                "adj_close": [100, 101, 102, 103],
                "dividend": [0.0, 0.0, 0.50, 0.0],
                "split_coefficient": [0.0, 0.0, 0.0, 0.0],
            },
        ),
        "BND": pd.DataFrame(
            {
                "date": dates,
                "close": [75, 76, 77, 78],
                "adj_close": [75, 76, 77, 78],
                "dividend": [0.20, 0.0, 0.0, 0.0],
                "split_coefficient": [0.0, 0.0, 0.0, 0.0],
            },
        ),
    }

    return data


@pytest.fixture
def transactions_perfect_match() -> pd.DataFrame:
    """
    User transactions that perfectly match the sample market data.
    - 20 shares of VTI * $0.50/share = $10.00 total
    - 50 shares of BND * $0.20/share = $10.00 total
    """
    data = [
        {
            "date": "2023-01-01",
            "ticker": "VTI",
            "transaction_type": "BUY",
            "quantity": 20,
            "value_per_share": 100.00,
            "total_value": 2000,
            "account": "Taxable",
        },
        {
            "date": "2023-01-01",
            "ticker": "BND",
            "transaction_type": "BUY",
            "quantity": 50,
            "value_per_share": 100.00,
            "total_value": 5000,
            "account": "Taxable",
        },
        {
            "date": "2023-03-20",
            "ticker": "BND",
            "transaction_type": "DIVIDEND",
            "quantity": 0,  # Cash dividend, no shares involved
            "value_per_share": 0,  # Not used for cash dividend
            # Negative as outflow (received dividend) and we are skipping the
            # loading which would make it a negative value
            "total_value": -10.00,
            "account": "Taxable",
        },
        {
            "date": "2023-03-25",
            "ticker": "VTI",
            "transaction_type": "DIVIDEND",
            "quantity": 0,  # Cash dividend, no shares involved
            "value_per_share": 0,  # Not used for cash dividend
            # Negative as outflow (received dividend) and we are skipping the
            # loading which would make it a negative value
            "total_value": -10.00,
            "account": "Taxable",
        },
    ]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def scenario_transactions_with_discrepancies() -> Tuple[str, pd.DataFrame]:
    """
    User transactions with various discrepancies compared to market data:
    1. Mismatch: VTI dividend amount is wrong ($9.00 instead of $10.00).
    2. Extra: User recorded a dividend for VTI on a day there was none.
    3. Missing: User did NOT record the dividend for BND.
    """
    data = [
        {
            "date": "2023-03-20",
            "ticker": "BND",
            "transaction_type": "BUY",
            "quantity": 50.0,
            "value_per_share": 75.00,
            "total_value": 3750.00,
            "account": "Taxable",
        },
        # Missing BND dividend from 2023-03-20
        {
            "date": "2023-03-20",
            "ticker": "VTI",
            "transaction_type": "BUY",
            "quantity": 20.0,
            "value_per_share": 100.00,
            "total_value": 2000.00,
            "account": "Taxable",
        },
        {
            "date": "2023-03-25",  # Mismatch
            "ticker": "VTI",
            "transaction_type": "DIVIDEND",
            "quantity": 0,  # Cash dividend, no shares involved
            "value_per_share": 0,  # Not used for cash dividend
            # Negative as outflow (received dividend) and we are skipping the
            # loading which would make it a negative value
            "total_value": -9.00,  # Should be $10.00
            "account": "Taxable",
        },
        {
            "date": "2023-03-26",  # Extra dividend
            "ticker": "VTI",
            "transaction_type": "DIVIDEND",
            "quantity": 0,  # Cash dividend, no shares involved
            "value_per_share": 0,  # Not used for cash dividend
            # Negative as outflow (received dividend) and we are skipping the
            # loading which would make it a negative value
            "total_value": -5.00,
            "account": "Taxable",
        },
    ]
    df = pd.DataFrame(data)
    return "discrepancies", df


@pytest.fixture
def transactions_with_discrepancies() -> pd.DataFrame:
    """User transactions with various discrepancies compared to market data."""
    _, df = scenario_transactions_with_discrepancies()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


@pytest.fixture
def transactions_no_dividends() -> pd.DataFrame:
    """User transactions with no dividends recorded."""
    data = [
        {
            "date": "2023-03-20",
            "ticker": "VTI",
            "transaction_type": "BUY",
            "quantity": 20.0,
            "value_per_share": 100.00,
            "total_value": 2000.00,
            "account": "Taxable",
        },
        {
            "date": "2023-03-20",
            "ticker": "BND",
            "transaction_type": "BUY",
            "quantity": 50.0,
            "value_per_share": 75.00,
            "total_value": 3750.00,
            "account": "Taxable",
        },
    ]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


# --- Test Cases ---


def test_validator_perfect_match(
    transactions_perfect_match,
    sample_market_data,
):
    # pylint: disable=redefined-outer-name
    """
    Tests the scenario where user dividends perfectly match market data.
    The validator should return an empty list of messages.
    """
    validator = DividendValidator(
        transactions_df=transactions_perfect_match,
        market_data_df=sample_market_data,
    )
    messages, dividend_differences = validator.validate()
    assert not messages
    assert not dividend_differences


def test_validator_with_discrepancies(
    transactions_with_discrepancies,
    sample_market_data,
):
    # pylint: disable=redefined-outer-name
    """
    Tests the scenario with a mix of mismatch, extra, and missing dividends.
    The validator should return three specific error messages.
    """
    transactions_df = transactions_with_discrepancies

    validator = DividendValidator(
        transactions_df=transactions_df,
        market_data_df=sample_market_data,
    )
    messages, _ = validator.validate()

    assert len(messages) == 3
    # Check for the "Mismatch" message
    assert any("Mismatch on 2023-03-25 for VTI" in msg for msg in messages)
    assert any("User recorded $9.00" in msg for msg in messages)
    assert any("market data suggests $10.00" in msg for msg in messages)

    # Check for the "Extra dividend" message
    assert any(
        "Extra user dividend on 2023-03-26 for VTI" in msg for msg in messages
    )
    assert any("no market dividend was found" in msg for msg in messages)

    # Check for the "Missing dividend" message
    assert any(
        "Missing dividend on 2023-03-20 for BND" in msg for msg in messages
    )
    assert any(
        "Market data shows a dividend of $0.2000/share" in msg
        for msg in messages
    )


def test_validator_user_has_no_dividends(
    transactions_no_dividends,
    sample_market_data,  # pylint: disable=redefined-outer-name
):
    # pylint: disable=redefined-outer-name
    """
    Tests the scenario where the user has no DIVIDEND type transactions.
    The validator should find the missing BND and VTI dividends.
    """
    validator = DividendValidator(
        transactions_df=transactions_no_dividends,
        market_data_df=sample_market_data,
    )
    messages, _ = validator.validate()

    assert len(messages) == 2
    assert any(
        "Missing dividend on 2023-03-25 for VTI" in msg for msg in messages
    )
    assert any(
        "Missing dividend on 2023-03-20 for BND" in msg for msg in messages
    )


def test_validator_no_market_data_for_ticker(
    transactions_perfect_match,
):  # pylint: disable=redefined-outer-name
    """
    Tests that validation is gracefully skipped for a ticker if no market
    data is available.
    """
    # Market data is missing for 'BND'
    market_data_missing_bnd = {
        "VTI": pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-03-25"], utc=True),
                "close": [102],
                "dividend": [0.5],
            },
        )
    }

    validator = DividendValidator(
        transactions_df=transactions_perfect_match,
        market_data_df=market_data_missing_bnd,
    )
    messages, _ = validator.validate()

    # Should only find a discrepancy for VTI (perfect match = no message)
    # and should not raise an error for the missing BND.
    # The BND dividend recorded by the user will be flagged as "Extra" because
    # there's no market data to compare it against.
    assert len(messages) == 2
    assert "No market dividends for BND" in messages[0]
    assert "Extra user dividend on 2023-03-20 for BND" in messages[1]


class TestDividendProcessorIntegration:
    """Test suite for processing and updating dividend calculations."""

    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            (config_dir / "transactions").mkdir()
            (config_dir / "market_data").mkdir()
            (config_dir / "output").mkdir()

            config = ConfigManager()
            config.config["data"]["base_path"] = str(config_dir)

            # Updating default settings
            config.config["settings"]["cache_market_data"] = True
            config.config["settings"]["force_refresh_market_data"] = False

            config.config["analysis"]["start_date"] = "2023-03-20"
            config.config["analysis"]["end_date"] = "2023-03-26"

            yield config

    @pytest.fixture(
        params=[
            scenario_transactions_with_discrepancies(),
        ],
        ids=lambda x: x[0],  # Use scenario name for test ID
    )
    # pylint: disable=redefined-outer-name
    def scenario_analyzer(
        self, request, temp_config, sample_market_data, monkeypatch
    ):
        """Fixture to set up BogleBenchAnalyzer for a given dividend
        scenario."""
        scenario_name, transactions_df = request.param

        # Save transactions to csv
        temp_data_path = temp_config.get_data_path()
        transactions_file_path = (
            temp_data_path / "transactions" / "transactions.csv"
        )
        transactions_df.to_csv(transactions_file_path, index=False)

        market_data_path = temp_config.get_market_data_path()
        for ticker, df in sample_market_data.items():
            df.to_parquet(market_data_path / f"{ticker}.parquet", index=False)

        output_path = temp_config.get_output_path()

        monkeypatch.setattr(
            ConfigManager, "get_data_path", lambda self, config: temp_data_path
        )

        monkeypatch.setattr(
            ConfigManager,
            "get_transactions_file_path",
            lambda self: transactions_file_path,
        )

        monkeypatch.setattr(
            ConfigManager,
            "get_market_data_path",
            lambda self: market_data_path,
        )

        monkeypatch.setattr(
            ConfigManager,
            "get_output_path",
            lambda self: output_path,
        )

        analyzer = BogleBenchAnalyzer()
        analyzer.config = temp_config

        yield analyzer, scenario_name, transactions_file_path

    def test_dividends_processing_scenarios(self, scenario_analyzer):
        """Test various dividend processing scenarios end-to-end."""
        analyzer, scenario_name, transactions_file = scenario_analyzer

        # --- Main Workflow ---
        analyzer.load_transactions(transactions_file)
        results = analyzer.calculate_performance()

        accuracy = 0.001 / 100  # 0.001% accuracy)

        # --- Assertions ---
        assert results is not None

        transactions_df = analyzer.transactions
        mar_25_vti_dividends = transactions_df[
            (transactions_df["date"] == pd.Timestamp("2023-03-25", tz="UTC"))
            & (transactions_df["ticker"] == "VTI")
            & (transactions_df["transaction_type"] == "DIVIDEND")
        ]
        mar_26_vti_dividends = transactions_df[
            (transactions_df["date"] == pd.Timestamp("2023-03-26", tz="UTC"))
            & (transactions_df["ticker"] == "VTI")
            & (transactions_df["transaction_type"] == "DIVIDEND")
        ]
        if scenario_name == "discrepancies":
            assert not mar_25_vti_dividends.empty
            # After processing, the VTI dividend should be corrected to $0.50
            # per share for 20 shares = $10.00 total
            # By convention, dividends are negative values in transactions
            assert (
                abs(mar_25_vti_dividends.iloc[0]["value_per_share"] - 0.50)
                < accuracy
            )
            assert (
                abs(mar_25_vti_dividends.iloc[0]["total_value"] + 10.00)
                < accuracy
            )

            assert not mar_26_vti_dividends.empty
            # The extra dividend on 2023-03-26 should be removed
            assert mar_26_vti_dividends.iloc[0]["total_value"] == 0.0
