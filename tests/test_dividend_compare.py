"""Tests for comparing user-reported dividends to market data.

This suite checks that the system correctly identifies matches and mismatches
between user transactions labeled as dividends and the dividends reported
in the market data.
"""

import unittest.mock

import pandas as pd
import pytest

from boglebench.core.dividend_validator import DividendValidator
from boglebench.core.portfolio import BogleBenchAnalyzer

START_DATE = "2023-01-01"
END_DATE = "2023-12-31"


class DummyConfig:
    """A dummy config object for testing purposes."""

    # pylint: disable-next=unused-argument
    def get(self, *args, **kwargs):
        """Dummy get method."""
        return None


@pytest.fixture
def analyzer():
    """Fixture to create a BogleBenchAnalyzer with dummy config."""
    # pylint: disable-next=redefined-outer-name
    analyzer = BogleBenchAnalyzer()

    with unittest.mock.patch.object(
        analyzer.__class__, "config", new_callable=unittest.mock.PropertyMock
    ) as mock_config:
        mock_config.return_value = DummyConfig()
        yield analyzer


def make_market_data():
    """Create simulated market data with dividends for testing."""
    # Simulate AlphaVantage market data with dividends
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03"], utc=True
            ),
            "open": [100, 102, 101],
            "close": [101, 103, 104],
            "dividend": [0.0, 1.23, 0.0],
        }
    )
    return {"VTI": df}


def make_transactions_div_and_reinvest(match=True):
    """Create simulated user transactions with dividends and reinvestments."""
    # Simulate user transactions for VTI, both DIVIDEND and DIVIDEND_REINVEST
    # on same date
    # DIVIDENDS recorded as negative values when running the analyzer
    if match:
        amount_div = -1.23
        amount_reinvest = -1.00
    else:
        amount_div = -0.23
        amount_reinvest = -2.00
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-02"], utc=True
            ),
            "ticker": ["VTI", "VTI", "VTI"],
            "transaction_type": ["BUY", "DIVIDEND", "DIVIDEND_REINVEST"],
            "quantity": [1, 0, 0.01],
            "value_per_share": [100, amount_div, amount_reinvest],
            "total_value": [0, amount_div, amount_reinvest],
            "account": ["Test", "Test", "Test"],
        }
    )
    return df


# pylint: disable-next=redefined-outer-name
def test_dividend_and_reinvest_match(analyzer):
    """Test matching user dividends and reinvestments."""
    analyzer.market_data = make_market_data()
    analyzer.transactions = make_transactions_div_and_reinvest(match=True)
    validator = DividendValidator(
        analyzer.transactions,
        analyzer.market_data,
        start_date=pd.to_datetime(START_DATE).tz_localize("UTC").normalize(),
        end_date=pd.to_datetime(END_DATE).tz_localize("UTC").normalize(),
    )
    messages, dividend_differences = validator.validate()
    messages_str = "\n".join(messages)

    assert "No user dividends" not in messages_str
    assert "Missing dividend" not in messages_str
    assert "Mismatch" not in messages_str
    assert "Extra user dividend" not in messages_str

    assert not dividend_differences


# pylint: disable-next=redefined-outer-name
def test_dividend_and_reinvest_mismatch(analyzer):
    """Test mismatching user dividends and reinvestments."""
    analyzer.market_data = make_market_data()
    analyzer.transactions = make_transactions_div_and_reinvest(match=False)
    validator = DividendValidator(
        analyzer.transactions,
        analyzer.market_data,
        start_date=pd.to_datetime(START_DATE).tz_localize("UTC").normalize(),
        end_date=pd.to_datetime(END_DATE).tz_localize("UTC").normalize(),
    )
    messages, dividend_differences = validator.validate()
    messages_str = "\n".join(messages)

    assert "Mismatch" in messages_str
    assert (
        "User recorded $0.23, but market data suggests $1.23" in messages_str
    )

    assert dividend_differences["VTI"].shape[0] == 1
    assert dividend_differences["VTI"].iloc[0]["total_value_market"] == -1.23
