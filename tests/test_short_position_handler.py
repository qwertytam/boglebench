"""
Tests for short position detection and handling.

This test suite validates that the short position handler correctly:
1. Detects transactions that would result in short positions
2. Rejects them in REJECT mode
3. Caps them appropriately in CAP mode
4. Leaves normal transactions unchanged
"""

import pandas as pd
import pytest

from boglebench.core.constants import ShortPositionHandling
from boglebench.core.short_position_handler import (
    ShortPositionError,
    ShortPositionHandler,
    process_transactions_with_short_check,
)


@pytest.fixture
def simple_transactions():
    """Create a simple set of transactions for testing."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2023-01-15",
                    "2023-02-15",
                    "2023-03-15",
                ],
                utc=True,
            ),
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "transaction_type": ["BUY", "BUY", "SELL"],
            "quantity": [100, 50, 25],
            "value_per_share": [150.0, 155.0, 165.0],
            "total_value": [15000.0, 7750.0, -4125.0],
            "account": ["Test", "Test", "Test"],
        }
    )


@pytest.fixture
def short_position_transactions():
    """Create transactions that result in a short position."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2023-01-15",
                    "2023-02-15",
                    "2023-03-15",
                ],
                utc=True,
            ),
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "transaction_type": ["BUY", "BUY", "SELL"],
            "quantity": [100, 50, 200],  # Selling more than owned
            "value_per_share": [150.0, 155.0, 165.0],
            "total_value": [15000.0, 7750.0, -33000.0],
            "account": ["Test", "Test", "Test"],
        }
    )


@pytest.fixture
def multi_account_transactions():
    """Create transactions across multiple accounts."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2023-01-15",
                    "2023-01-15",
                    "2023-02-15",
                    "2023-02-15",
                ],
                utc=True,
            ),
            "symbol": ["AAPL", "AAPL", "AAPL", "AAPL"],
            "transaction_type": ["BUY", "BUY", "SELL", "SELL"],
            "quantity": [100, 50, 60, 40],
            "value_per_share": [150.0, 150.0, 165.0, 165.0],
            "total_value": [15000.0, 7500.0, -9900.0, -6600.0],
            "account": ["Account1", "Account2", "Account1", "Account2"],
        }
    )


class TestShortPositionHandler:
    """Test the ShortPositionHandler class."""

    def test_init_valid_mode(self):
        """Test initialization with valid handling mode."""
        handler = ShortPositionHandler(ShortPositionHandling.REJECT)
        assert handler.handling_mode == ShortPositionHandling.REJECT

        handler = ShortPositionHandler(ShortPositionHandling.CAP)
        assert handler.handling_mode == ShortPositionHandling.CAP

    def test_init_invalid_mode(self):
        """Test initialization with invalid handling mode."""
        with pytest.raises(ValueError, match="Invalid short position"):
            ShortPositionHandler("invalid_mode")

    def test_no_short_position(self, simple_transactions):
        """Test that normal transactions pass through unchanged."""
        handler = ShortPositionHandler(ShortPositionHandling.REJECT)
        holdings = {}

        for _, trans in simple_transactions.iterrows():
            adjusted, was_adjusted = handler.check_and_adjust_transaction(
                trans, holdings
            )
            assert not was_adjusted
            assert adjusted["quantity"] == trans["quantity"]

            # Update holdings for next iteration
            account = trans["account"]
            symbol = trans["symbol"]
            if account not in holdings:
                holdings[account] = {}
            holdings[account][symbol] = (
                holdings[account].get(symbol, 0.0) + trans["quantity"]
            )

    def test_reject_mode_raises_error(self, short_position_transactions):
        """Test that REJECT mode raises ShortPositionError."""
        handler = ShortPositionHandler(ShortPositionHandling.REJECT)
        holdings = {}

        # Process first two transactions (BUY)
        for _, trans in short_position_transactions.iloc[:2].iterrows():
            adjusted, _ = handler.check_and_adjust_transaction(trans, holdings)
            account = trans["account"]
            symbol = trans["symbol"]
            if account not in holdings:
                holdings[account] = {}
            holdings[account][symbol] = (
                holdings[account].get(symbol, 0.0) + trans["quantity"]
            )

        # Third transaction should raise error (SELL too many)
        with pytest.raises(ShortPositionError) as exc_info:
            handler.check_and_adjust_transaction(
                short_position_transactions.iloc[2], holdings
            )

        error = exc_info.value
        assert error.symbol == "AAPL"
        assert error.account == "Test"
        assert error.current_position == 150.0
        assert error.transaction_quantity == -200.0
        assert error.resulting_position == -50.0

    def test_cap_mode_adjusts_transaction(self, short_position_transactions):
        """Test that CAP mode adjusts the transaction to avoid short position."""
        handler = ShortPositionHandler(ShortPositionHandling.CAP)
        holdings = {}

        # Process first two transactions (BUY)
        for _, trans in short_position_transactions.iloc[:2].iterrows():
            adjusted, _ = handler.check_and_adjust_transaction(trans, holdings)
            account = trans["account"]
            symbol = trans["symbol"]
            if account not in holdings:
                holdings[account] = {}
            holdings[account][symbol] = (
                holdings[account].get(symbol, 0.0) + trans["quantity"]
            )

        # Third transaction should be capped
        adjusted, was_adjusted = handler.check_and_adjust_transaction(
            short_position_transactions.iloc[2], holdings
        )

        assert was_adjusted
        # Should cap to -150 (selling all 150 shares owned)
        assert adjusted["quantity"] == -150.0
        # Total value should be adjusted proportionally
        assert adjusted["total_value"] == -150.0 * 165.0

    def test_cap_mode_exact_zero(self):
        """Test CAP mode when transaction would result in exactly zero."""
        handler = ShortPositionHandler(ShortPositionHandling.CAP)
        holdings = {"Test": {"AAPL": 100.0}}

        trans = pd.Series(
            {
                "date": pd.Timestamp("2023-01-15", tz="UTC"),
                "symbol": "AAPL",
                "transaction_type": "SELL",
                "quantity": -100.0,
                "value_per_share": 150.0,
                "total_value": -15000.0,
                "account": "Test",
            }
        )

        adjusted, was_adjusted = handler.check_and_adjust_transaction(
            trans, holdings
        )

        # Should not be adjusted (exactly zero is fine)
        assert not was_adjusted
        assert adjusted["quantity"] == -100.0

    def test_multi_account_independence(self, multi_account_transactions):
        """Test that accounts are tracked independently."""
        handler = ShortPositionHandler(ShortPositionHandling.REJECT)
        holdings = {}

        # All transactions should pass - each account stays positive
        for _, trans in multi_account_transactions.iterrows():
            adjusted, was_adjusted = handler.check_and_adjust_transaction(
                trans, holdings
            )
            assert not was_adjusted

            account = trans["account"]
            symbol = trans["symbol"]
            if account not in holdings:
                holdings[account] = {}
            holdings[account][symbol] = (
                holdings[account].get(symbol, 0.0) + trans["quantity"]
            )

        # Verify final holdings
        assert holdings["Account1"]["AAPL"] == 40.0  # 100 - 60
        assert holdings["Account2"]["AAPL"] == 10.0  # 50 - 40


class TestProcessTransactionsWithShortCheck:
    """Test the process_transactions_with_short_check function."""

    def test_process_normal_transactions(self, simple_transactions):
        """Test processing transactions without short positions."""
        result = process_transactions_with_short_check(
            simple_transactions, ShortPositionHandling.REJECT
        )

        assert len(result) == len(simple_transactions)
        pd.testing.assert_frame_equal(result, simple_transactions)

    def test_process_with_reject_mode(self, short_position_transactions):
        """Test processing with REJECT mode raises error."""
        with pytest.raises(ShortPositionError):
            process_transactions_with_short_check(
                short_position_transactions, ShortPositionHandling.REJECT
            )

    def test_process_with_cap_mode(self, short_position_transactions):
        """Test processing with CAP mode adjusts transactions."""
        result = process_transactions_with_short_check(
            short_position_transactions, ShortPositionHandling.CAP
        )

        assert len(result) == len(short_position_transactions)
        # First two transactions should be unchanged
        assert result.iloc[0]["quantity"] == 100.0
        assert result.iloc[1]["quantity"] == 50.0
        # Third transaction should be capped to -150
        assert result.iloc[2]["quantity"] == -150.0

    def test_process_unsorted_transactions(self):
        """Test that transactions are sorted by date before processing."""
        transactions = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2023-03-15", "2023-01-15", "2023-02-15"], utc=True
                ),
                "symbol": ["AAPL", "AAPL", "AAPL"],
                "transaction_type": ["SELL", "BUY", "BUY"],
                "quantity": [25, 100, 50],
                "value_per_share": [165.0, 150.0, 155.0],
                "total_value": [-4125.0, 15000.0, 7750.0],
                "account": ["Test", "Test", "Test"],
            }
        )

        # Should process in date order, not raise error
        result = process_transactions_with_short_check(
            transactions, ShortPositionHandling.REJECT
        )

        # Result should be sorted by date
        assert result.iloc[0]["date"] == pd.Timestamp("2023-01-15", tz="UTC")
        assert result.iloc[1]["date"] == pd.Timestamp("2023-02-15", tz="UTC")
        assert result.iloc[2]["date"] == pd.Timestamp("2023-03-15", tz="UTC")

    def test_dividend_transactions_ignored(self):
        """Test that dividend transactions don't affect holdings tracking."""
        transactions = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2023-01-15", "2023-02-15", "2023-03-15"], utc=True
                ),
                "symbol": ["AAPL", "AAPL", "AAPL"],
                "transaction_type": ["BUY", "DIVIDEND", "SELL"],
                "quantity": [100, 0, 50],
                "value_per_share": [150.0, 0.0, 165.0],
                "total_value": [15000.0, -50.0, -8250.0],
                "account": ["Test", "Test", "Test"],
            }
        )

        result = process_transactions_with_short_check(
            transactions, ShortPositionHandling.REJECT
        )

        # Should process without error
        assert len(result) == 3

    def test_multiple_symbols(self):
        """Test handling multiple symbols independently."""
        transactions = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2023-01-15",
                        "2023-01-15",
                        "2023-02-15",
                        "2023-02-15",
                    ],
                    utc=True,
                ),
                "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
                "transaction_type": ["BUY", "BUY", "SELL", "SELL"],
                "quantity": [100, 50, 50, 25],
                "value_per_share": [150.0, 240.0, 165.0, 245.0],
                "total_value": [15000.0, 12000.0, -8250.0, -6125.0],
                "account": ["Test", "Test", "Test", "Test"],
            }
        )

        result = process_transactions_with_short_check(
            transactions, ShortPositionHandling.REJECT
        )

        # Should process without error - each symbol tracked independently
        assert len(result) == 4

    def test_short_then_buy_same_day(self):
        """Test handling short position that's resolved same day."""
        # This tests if we try to sell before buying on the same day
        transactions = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2023-01-15", "2023-01-15"], utc=True
                ),
                "symbol": ["AAPL", "AAPL"],
                "transaction_type": ["SELL", "BUY"],
                "quantity": [-100, 100],
                "value_per_share": [150.0, 150.0],
                "total_value": [-15000.0, 15000.0],
                "account": ["Test", "Test"],
            }
        )

        # Should raise error in REJECT mode (sell before buy)
        with pytest.raises(ShortPositionError):
            process_transactions_with_short_check(
                transactions, ShortPositionHandling.REJECT
            )

        # Should cap in CAP mode
        result = process_transactions_with_short_check(
            transactions, ShortPositionHandling.CAP
        )
        # First transaction should be capped to 0 (can't sell what you don't have)
        assert result.iloc[0]["quantity"] == 0.0
