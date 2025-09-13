"""
Provides functionality to validate user-provided dividends against market data.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.constants import (
    DateAndTimeConstants,
    NumericalConstants,
    TransactionTypes,
)
from ..utils.logging_config import get_logger
from ..utils.tools import aggregate_dividends

logger = get_logger()


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


class DividendValidator:
    """
    Compares and validates user-recorded dividends against market data.
    """

    def __init__(
        self,
        transactions_df: pd.DataFrame,
        market_data_df: Dict[str, pd.DataFrame],
        dividend_tolerance: float = NumericalConstants.ONE_CENT.value,
    ):
        """
        Initializes the DividendValidator.

        Args:
            transactions_df: DataFrame of user's transactions.
            market_data_df: DataFrame of market data for all relevant tickers.
            dividend_tolerance: The tolerance for dividend amount comparison.
        """
        self.transactions_df = transactions_df
        self.market_data_df = market_data_df
        self.dividend_tolerance = dividend_tolerance

    def _get_market_dividends_for_ticker(self, ticker: str) -> pd.DataFrame:
        """Extracts market dividends for a specific ticker."""

        logger.debug("Starting market dividend extraction for %s.", ticker)

        data_df = self.market_data_df.get(ticker)
        if data_df is None or data_df.empty:
            logger.warning(
                "⚠️  No market dividend data found for %s, skipping comparison.",
                ticker,
            )
            return pd.DataFrame()

        try:
            logger.debug("data_df column dtypes:\n%s", data_df.dtypes)
            date_tz = data_df["date"].dt.tz
            logger.debug("Market data 'date' column tz info: %s", date_tz)
        except TypeError as e:
            logger.debug(
                "Market data 'date' column tz info could not be determined: %s",
                e,
            )
            date_tz = None

        if date_tz is None:
            data_df["date"] = data_df["date"].dt.tz_localize(
                DateAndTimeConstants.TZ_UTC.value
            )
            logger.debug(
                "Localized 'date' column to %s.", data_df["date"].dt.tz
            )

        market_dividends = data_df[np.abs(data_df["dividend"]) > 0][
            ["date", "dividend"]
        ].copy()

        market_dividends.rename(
            columns={"dividend": "value_per_share"},
            inplace=True,
        )

        return market_dividends

    def _get_shares_held_on_date(
        self, ticker: str, date: pd.Timestamp, account: Optional[str] = None
    ) -> float:
        """
        Retrieves the number of shares held for a specific ticker on a given date.

        Args:
            ticker: The stock ticker symbol.
            date: The date for which to retrieve the share quantity.
            account: Optional account identifier to filter by account.

        Returns:
            The number of shares held on the specified date.
        """
        logger.debug(
            "Retrieving shares held for %s on %s in account %s.",
            ticker,
            date,
            account if account else "ALL",
        )

        # Filter transactions for the specific ticker and date
        mask = (
            (self.transactions_df["ticker"] == ticker)
            & (self.transactions_df["date"].dt.date < date.date())
            & (
                identify_quantity_change_transactions(
                    self.transactions_df["transaction_type"]
                )
            )
        )

        if account:
            mask &= self.transactions_df["account"] == account

        shares_held = self.transactions_df.loc[mask, "quantity"].sum()
        logger.debug(
            "Shares held for %s on %s in account %s: %d",
            ticker,
            date,
            account if account else "ALL",
            shares_held,
        )

        return shares_held

    def validate(self) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
        """
        Performs the dividend comparison and returns a list of messages.

        Returns:
            A list of formatted strings describing dividend discrepancies.
        """
        messages: list[str] = []
        dividend_differences: Dict[str, pd.DataFrame] = {}
        user_dividends = self.transactions_df[
            (
                identify_dividend_transactions(
                    self.transactions_df["transaction_type"]
                )
            )
        ].copy()

        all_tickers = self.transactions_df["ticker"].unique()
        logger.info(
            "Validating dividends for %d unique tickers: %s",
            len(all_tickers),
            ", ".join(all_tickers),
        )

        for ticker in all_tickers:
            user_ticker_dividends = user_dividends[
                user_dividends["ticker"] == ticker
            ]

            user_divs_grouped = user_ticker_dividends.groupby(
                ["date", "account"],
                as_index=False,
            ).apply(aggregate_dividends)

            user_divs_grouped.reset_index(drop=True, inplace=True)

            market_ticker_dividends = self._get_market_dividends_for_ticker(
                ticker
            )

            if market_ticker_dividends.empty:
                logger.warning(
                    "⚠️  No market dividend data found for %s",
                    ticker,
                )
                market_ticker_dividends = pd.DataFrame(
                    {
                        "date": pd.to_datetime([], utc=True),
                        "value_per_share_market": [],
                    }
                )

            comparison_df = pd.merge(
                user_divs_grouped,
                market_ticker_dividends,
                on="date",
                how="outer",
                suffixes=("_user", "_market"),
            )

            # Check for discrepancies
            dividend_differences_df = pd.DataFrame()
            for _, row in comparison_df.iterrows():
                total_value_user = row.get("total_value")
                value_per_share_market = row.get("value_per_share_market")
                date = pd.to_datetime(row["date"])
                account = row.get("account", None)
                shares = self._get_shares_held_on_date(ticker, date, account)

                if pd.notna(total_value_user) and pd.notna(
                    value_per_share_market
                ):
                    # Case 1: Both user and market have a dividend on this day
                    # Dividends are treated as cash outflows (hence negative)
                    expected_total = -shares * value_per_share_market
                    if not np.isclose(
                        total_value_user,
                        expected_total,
                        atol=self.dividend_tolerance,
                    ):
                        msg = (
                            f"   Mismatch on {row['date'].date()} "
                            f"for {ticker} "
                            f"in account {account if account else 'ALL'}: "
                            f"User recorded ${np.abs(total_value_user):.2f}, "
                            f"but market data suggests "
                            f"${np.abs(expected_total):.2f} "
                            f"({shares:.4f} shares * "
                            f"${value_per_share_market:.4f})."
                        )
                        messages.append(msg)

                        dividend_differences_df = pd.concat(
                            [
                                pd.DataFrame(
                                    {
                                        "date": [row["date"]],
                                        "account": [account],
                                        "total_value_user": [total_value_user],
                                        "value_per_share_market": [
                                            value_per_share_market
                                        ],
                                        "quantity": [shares],
                                        "total_value_market": [expected_total],
                                    }
                                ),
                            ],
                            ignore_index=True,
                        )
                elif pd.notna(total_value_user) and pd.isna(
                    value_per_share_market
                ):
                    # Case 2: User recorded a dividend, but market has none
                    msg = (
                        f"   Extra dividend on {row['date'].date()} "
                        f"for {ticker} "
                        f"in account {account if account else 'ALL'}: "
                        f"User recorded ${np.abs(total_value_user):.2f}, "
                        f"but no market dividend was found."
                    )
                    messages.append(msg)
                    dividend_differences_df = pd.concat(
                        [
                            dividend_differences_df,
                            pd.DataFrame(
                                {
                                    "date": [row["date"]],
                                    "account": [account],
                                    "total_value_user": [total_value_user],
                                    "value_per_share_market": [np.nan],
                                    "quantity": [shares],
                                    "total_value_market": [np.nan],
                                }
                            ),
                        ],
                        ignore_index=True,
                    )
                elif pd.isna(total_value_user) and pd.notna(
                    value_per_share_market
                ):
                    # Case 3: Market has a dividend, but user did not record one
                    msg = (
                        f"   Missing dividend on {row['date'].date()} for "
                        f"{ticker} "
                        f"in account {account if account else 'ALL'}: "
                        f"Market data shows a dividend of "
                        f"${value_per_share_market:.4f}/share, "
                        f"but none was recorded."
                    )
                    messages.append(msg)
                    dividend_differences_df = pd.concat(
                        [
                            dividend_differences_df,
                            pd.DataFrame(
                                {
                                    "date": [row["date"]],
                                    "account": [account],
                                    "total_value_user": [np.nan],
                                    "value_per_share_market": [
                                        value_per_share_market
                                    ],
                                    "quantity": [shares],
                                    "total_value_market": [
                                        shares * value_per_share_market
                                    ],
                                }
                            ),
                        ],
                        ignore_index=True,
                    )

                if not dividend_differences_df.empty:
                    dividend_differences[ticker] = dividend_differences_df

        if messages:
            logger.warning("Found %d dividend discrepancies.", len(messages))
        else:
            logger.info("✅ User dividends align with market data.")

        return messages, dividend_differences
