"""
Short position detection and handling.

This module provides functionality to detect and handle transactions that would
result in short positions (negative holdings). BogleBench does not support 
short positions, so this module implements three strategies:
1. REJECT: Reject transactions that would result in short positions
2. CAP: Cap the transaction quantity to the available long position
3. IGNORE: Log warning but allow short positions to occur
"""

from typing import Dict, Optional, Tuple

import pandas as pd

from ..utils.logging_config import get_logger
from .constants import ShortPositionHandling, TransactionTypes

logger = get_logger()


class ShortPositionError(ValueError):
    """Exception raised when a transaction would result in a short position."""

    def __init__(
        self,
        date: pd.Timestamp,
        account: str,
        symbol: str,
        current_position: float,
        transaction_quantity: float,
        resulting_position: float,
    ):
        self.date = date
        self.account = account
        self.symbol = symbol
        self.current_position = current_position
        self.transaction_quantity = transaction_quantity
        self.resulting_position = resulting_position

        message = (
            f"Transaction would result in short position:\n"
            f"  Date: {date.date()}\n"
            f"  Account: {account}\n"
            f"  Symbol: {symbol}\n"
            f"  Current position: {current_position:.2f} shares\n"
            f"  Transaction quantity: {transaction_quantity:.2f} shares\n"
            f"  Would result in: {resulting_position:.2f} shares\n"
            f"Short positions are not supported. "
            f"Consider setting 'short_position_handling' to 'cap' in config "
            f"to automatically adjust the transaction."
        )
        super().__init__(message)


class ShortPositionHandler:
    """Handles detection and resolution of short position scenarios."""

    def __init__(self, handling_mode: str = ShortPositionHandling.REJECT):
        """
        Initialize the short position handler.

        Args:
            handling_mode: How to handle short positions ("reject" or "cap")
        """
        if not ShortPositionHandling.is_valid(handling_mode):
            raise ValueError(
                f"Invalid short position handling mode: {handling_mode}. "
                f"Must be one of: {ShortPositionHandling.all_modes()}"
            )
        self.handling_mode = handling_mode
        self.logger = get_logger()

    def check_and_adjust_transaction(
        self,
        transaction: pd.Series,
        current_holdings: Dict[str, Dict[str, float]],
    ) -> Tuple[pd.Series, bool]:
        """
        Check if a transaction would result in a short position and handle it.

        Args:
            transaction: Transaction data as a pandas Series
            current_holdings: Dict of account -> symbol -> quantity

        Returns:
            Tuple of (adjusted_transaction, was_adjusted)

        Raises:
            ShortPositionError: If handling_mode is REJECT and short position detected
        """
        # Only check transactions that change quantity
        if not TransactionTypes.is_quantity_changing(
            transaction["transaction_type"]
        ):
            return transaction, False

        account = transaction["account"]
        symbol = transaction["symbol"]
        trans_qty = transaction["quantity"]

        # Get current position (default to 0 if not exists)
        if account not in current_holdings:
            current_holdings[account] = {}
        current_position = current_holdings[account].get(symbol, 0.0)

        # Calculate resulting position
        resulting_position = current_position + trans_qty

        # Check if this would result in a short position
        # Use a small epsilon to account for floating point errors
        epsilon = 1e-10
        if resulting_position < -epsilon:
            return self._handle_short_position(
                transaction, current_position, resulting_position
            )

        return transaction, False

    def _handle_short_position(
        self,
        transaction: pd.Series,
        current_position: float,
        resulting_position: float,
    ) -> Tuple[pd.Series, bool]:
        """
        Handle a detected short position based on the configured mode.

        Args:
            transaction: Transaction data
            current_position: Current holdings before transaction
            resulting_position: Resulting holdings after transaction

        Returns:
            Tuple of (adjusted_transaction, was_adjusted)
        """
        if self.handling_mode == ShortPositionHandling.REJECT:
            raise ShortPositionError(
                date=transaction["date"],
                account=transaction["account"],
                symbol=transaction["symbol"],
                current_position=current_position,
                transaction_quantity=transaction["quantity"],
                resulting_position=resulting_position,
            )

        elif self.handling_mode == ShortPositionHandling.CAP:
            return self._cap_transaction(transaction, current_position)

        elif self.handling_mode == ShortPositionHandling.IGNORE:
            return self._ignore_short_position(
                transaction, current_position, resulting_position
            )

        # Should never reach here due to validation in __init__
        raise ValueError(f"Unknown handling mode: {self.handling_mode}")

    def _cap_transaction(
        self, transaction: pd.Series, current_position: float
    ) -> Tuple[pd.Series, bool]:
        """
        Cap a transaction to the available long position.

        Args:
            transaction: Transaction data
            current_position: Current holdings before transaction

        Returns:
            Tuple of (adjusted_transaction, was_adjusted)
        """
        # Create a copy to avoid modifying the original
        adjusted = transaction.copy()

        # For a SELL transaction (negative quantity), cap to current position
        # This results in zero holdings after the transaction
        original_quantity = adjusted["quantity"]
        adjusted["quantity"] = -current_position

        # Also adjust the total_value to reflect the capped quantity
        if adjusted["value_per_share"] != 0:
            adjusted["total_value"] = (
                adjusted["quantity"] * adjusted["value_per_share"]
            )

        self.logger.warning(
            "⚠️  Short position detected and capped:\n"
            "   Date: %s\n"
            "   Account: %s\n"
            "   Symbol: %s\n"
            "   Original transaction: %.2f shares\n"
            "   Current position: %.2f shares\n"
            "   Adjusted transaction: %.2f shares (resulting in 0 shares)",
            adjusted["date"].date(),
            adjusted["account"],
            adjusted["symbol"],
            original_quantity,
            current_position,
            adjusted["quantity"],
        )

        return adjusted, True

    def _ignore_short_position(
        self,
        transaction: pd.Series,
        current_position: float,
        resulting_position: float,
    ) -> Tuple[pd.Series, bool]:
        """
        Log a warning but allow the short position to occur.

        Args:
            transaction: Transaction data
            current_position: Current holdings before transaction
            resulting_position: Resulting holdings after transaction

        Returns:
            Tuple of (original_transaction, was_adjusted=False)
        """
        self.logger.warning(
            "⚠️  Short position detected but allowed (IGNORE mode):\n"
            "   Date: %s\n"
            "   Account: %s\n"
            "   Symbol: %s\n"
            "   Current position: %.2f shares\n"
            "   Transaction quantity: %.2f shares\n"
            "   Will result in: %.2f shares (short position)\n"
            "   Note: Portfolio metrics may be incorrect for short positions",
            transaction["date"].date(),
            transaction["account"],
            transaction["symbol"],
            current_position,
            transaction["quantity"],
            resulting_position,
        )

        return transaction, False


def process_transactions_with_short_check(
    transactions: pd.DataFrame,
    handling_mode: str = ShortPositionHandling.REJECT,
) -> pd.DataFrame:
    """
    Process all transactions and check for short positions.

    This function simulates processing transactions chronologically and
    checks/adjusts any that would result in short positions.

    Args:
        transactions: DataFrame with transaction data (must be sorted by date)
        handling_mode: How to handle short positions ("reject" or "cap")

    Returns:
        DataFrame with potentially adjusted transactions

    Raises:
        ShortPositionError: If handling_mode is REJECT and short position detected
    """
    handler = ShortPositionHandler(handling_mode)
    logger = get_logger()

    # Ensure transactions are sorted by date
    transactions = transactions.sort_values("date").reset_index(drop=True)

    # Track holdings per account per symbol
    holdings: Dict[str, Dict[str, float]] = {}

    # Track which transactions were adjusted
    adjusted_transactions = []
    any_adjusted = False

    logger.info("🔍 Checking transactions for short positions...")

    for idx, transaction in transactions.iterrows():
        # Check and potentially adjust this transaction
        adjusted_trans, was_adjusted = handler.check_and_adjust_transaction(
            transaction, holdings
        )

        if was_adjusted:
            any_adjusted = True

        adjusted_transactions.append(adjusted_trans)

        # Update holdings if this transaction changes quantity
        if TransactionTypes.is_quantity_changing(
            adjusted_trans["transaction_type"]
        ):
            account = adjusted_trans["account"]
            symbol = adjusted_trans["symbol"]

            if account not in holdings:
                holdings[account] = {}

            holdings[account][symbol] = (
                holdings[account].get(symbol, 0.0) + adjusted_trans["quantity"]
            )

    if any_adjusted:
        logger.info(
            "✅ Transaction processing complete with adjustments applied"
        )
    else:
        logger.info("✅ Transaction processing complete - no adjustments needed")

    return pd.DataFrame(adjusted_transactions)
