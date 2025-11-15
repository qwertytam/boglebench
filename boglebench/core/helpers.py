"""
Utility functions for portfolio analysis.

This module provides helper functions for identifying transaction types,
calculating share balances, and other common operations used throughout
the portfolio analysis workflow.
"""

from typing import Optional

import pandas as pd

from ..core.constants import TransactionTypes


def identify_dividend_transactions(series: pd.Series) -> pd.Series:
    """
    Identifies transactions in a Series with type 'DIVIDEND'.

    Args:
        series: A pandas Series containing transaction type strings.

    Returns:
        A Series of booleans indicating which transactions are 'DIVIDEND'.
    """
    # Get the set of all valid transaction type strings from our Enum
    valid_types = [TransactionTypes.DIVIDEND.value]

    # Return the unique values from the original series that were invalid
    return series.isin(valid_types)


def identify_any_dividend_transactions(series: pd.Series) -> pd.Series:
    """
    Identifies transactions in a Series with type 'DIVIDEND' or
    'DIVIDEND_REINVEST'.

    Args:
        series: A pandas Series containing transaction type strings.

    Returns:
        A Series of booleans indicating which transactions are 'DIVIDEND' or
        'DIVIDEND_REINVEST'.
    """
    # Get the set of all valid transaction type strings from our Enum
    valid_types = [
        TransactionTypes.DIVIDEND.value,
        TransactionTypes.DIVIDEND_REINVEST.value,
    ]

    # Return the unique values from the original series that were invalid
    return series.isin(valid_types)


def identify_quantity_change_transactions(series: pd.Series) -> pd.Series:
    """
    Identifies transactions in a Series with a type that will change the
    quantity.

    Args:
        series: A pandas Series containing transaction type strings.

    Returns:
        A Series of booleans indicating which transactions will change the
        quantity.
    """
    # Get the set of all valid transaction type strings from our Enum
    valid_types = [
        ttype for ttype in TransactionTypes.all_quantity_changing_types()
    ]

    # Return the unique values from the original series that were invalid
    return series.isin(valid_types)


def get_shares_held_on_date(
    symbol: str,
    date: pd.Timestamp,
    transactions_df: pd.DataFrame,
    account: Optional[str] = None,
    start_date: Optional[pd.Timestamp] = None,
) -> float:
    """
    Retrieves the number of shares held for a specific symbol on a given date.

    Args:
        symbol: The stock symbol symbol.
        date: The date for which to retrieve the share quantity.
        account: Optional account identifier to filter by account.

    Returns:
        The number of shares held on the specified date.
    """

    mask = (
        (transactions_df["symbol"] == symbol)
        & (transactions_df["date"].dt.date < date.date())
        & (
            identify_quantity_change_transactions(
                transactions_df["transaction_type"]
            )
        )
    )

    if account:
        mask &= transactions_df["account"] == account

    if start_date is not None:
        mask &= transactions_df["date"].dt.date >= start_date.date()

    shares_held = transactions_df.loc[mask, "quantity"].sum()

    return shares_held
