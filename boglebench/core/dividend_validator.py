"""
Dividend data validation against market sources.

This module provides detailed validation of user-provided dividends against
official dividend data from Alpha Vantage. Compares dividend amounts, per-share
values, and dates to identify discrepancies and ensure accurate portfolio
performance calculations.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.constants import DateAndTimeConstants, NumericalConstants
from ..core.helpers import (
    get_shares_held_on_date,
    identify_dividend_transactions,
)
from ..utils.logging_config import get_logger
from ..utils.tools import aggregate_dividends

logger = get_logger()


class DividendValidator:
    """
    Compares and validates user-recorded dividends against market data.
    """

    def __init__(
        self,
        transactions_df: pd.DataFrame,
        market_data_df: Dict[str, pd.DataFrame],
        dividend_tolerance: float = NumericalConstants.ONE_CENT.value,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ):
        """
        Initializes the DividendValidator.

        Args:
            transactions_df: DataFrame of user's transactions.
            market_data_df: DataFrame of market data for all relevant symbols.
            dividend_tolerance: The tolerance for dividend amount comparison.
        """
        self.transactions_df = transactions_df
        self.market_data_df = market_data_df
        self.dividend_tolerance = dividend_tolerance
        self.start_date = start_date
        self.end_date = end_date

    def _get_market_dividends_for_symbol(self, symbol: str) -> pd.DataFrame:
        """Extracts market dividends for a specific symbol."""

        logger.debug("Starting market dividend extraction for %s.", symbol)

        data_df = self.market_data_df.get(symbol)
        if data_df is None or data_df.empty:
            logger.warning(
                "⚠️  No market dividend data found for %s, skipping comparison.",
                symbol,
            )
            return pd.DataFrame()

        try:
            date_tz = data_df["date"].dt.tz

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
        self, symbol: str, date: pd.Timestamp, account: Optional[str] = None
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
        shares_held = get_shares_held_on_date(
            symbol,
            date,
            self.transactions_df,
            account=account,
            start_date=self.start_date,
        )

        return shares_held

    def validate(
        self, future_div_mode: str = "ignore"
    ) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
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

        all_symbols = self.transactions_df["symbol"].unique()
        logger.info(
            "Validating dividends for %d unique symbols: %s",
            len(all_symbols),
            ", ".join(all_symbols),
        )

        for symbol in all_symbols:
            user_symbol_dividends = user_dividends[
                user_dividends["symbol"] == symbol
            ]

            user_divs_grouped = user_symbol_dividends.groupby(
                ["date", "account"],
                as_index=False,
            ).apply(aggregate_dividends)

            user_divs_grouped.reset_index(drop=True, inplace=True)

            market_symbol_dividends = self._get_market_dividends_for_symbol(
                symbol
            )

            if self.start_date is not None and self.end_date is not None:
                logger.debug(
                    "Filtering user dividends for %s between %s and %s.",
                    symbol,
                    self.start_date,
                    self.end_date,
                )
                user_dividends_in_range = user_divs_grouped[
                    (user_divs_grouped["date"] >= self.start_date)
                    & (user_divs_grouped["date"] <= self.end_date)
                ]

            else:
                user_dividends_in_range = user_divs_grouped

            user_dividends_has_data = (
                user_dividends_in_range.empty
                or user_dividends_in_range["total_value"].abs().sum()
                <= self.dividend_tolerance
            )
            if user_dividends_has_data:
                msg = (
                    f"No user dividends for {symbol} "
                    f"in the specified date range."
                )
                # messages.append(msg)
                logger.debug(msg)

            if market_symbol_dividends.empty:
                if self.start_date is not None and self.end_date is not None:
                    if (
                        user_dividends_in_range.empty
                        or user_dividends_in_range["total_value"].abs().sum()
                        <= self.dividend_tolerance
                    ):
                        msg = (
                            f"No user or market dividends for {symbol} "
                            f"in the specified date range; skipping comparison."
                        )
                        messages.append(msg)
                        logger.debug(msg)
                else:
                    msg = (
                        f"⚠️  No market dividends for {symbol}; "
                        f"skipping comparison."
                    )
                    messages.append(msg)
                    logger.warning(msg)

                market_symbol_dividends = pd.DataFrame(
                    {
                        "date": pd.to_datetime([], utc=True),
                        "value_per_share_market": [],
                    }
                )

            elif self.start_date is not None and self.end_date is not None:
                last_transaction_date = self.transactions_df[
                    self.transactions_df["symbol"] == symbol
                ]["date"].max()

                adj_end_date = self.end_date
                if future_div_mode == "ignore" and (
                    self.end_date > last_transaction_date
                ):
                    logger.debug(
                        "Ignoring market dividends for %s after last "
                        "transaction date %s.",
                        symbol,
                        last_transaction_date.date(),
                    )
                    adj_end_date = min(self.end_date, last_transaction_date)

                market_symbol_dividends = market_symbol_dividends[
                    (market_symbol_dividends["date"] >= self.start_date)
                    & (market_symbol_dividends["date"] <= adj_end_date)
                ]

            comparison_df = pd.merge(
                user_dividends_in_range,
                market_symbol_dividends,
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
                shares = self._get_shares_held_on_date(symbol, date, account)

                if pd.notna(total_value_user) and pd.notna(
                    value_per_share_market
                ):
                    # Case 1: Both user and market have a dividend on this day
                    # Dividends are treated as cash outflows (hence negative)
                    # Calculate expected total dividend based on shares held
                    # and market rate
                    # For a short position, short holder pays the dividend
                    # For a long position, long holder receives the dividend
                    expected_total = np.round(
                        -shares * value_per_share_market, 2
                    )  # Rounded to nearest whole cents
                    if not np.isclose(
                        total_value_user,
                        expected_total,
                        atol=self.dividend_tolerance,
                    ):
                        msg = (
                            f"Mismatch on {row['date'].date()} "
                            f"for {symbol} "
                            f"in account {account if account else 'ALL'}: "
                            f"User recorded ${-total_value_user:.2f}, "
                            f"but market data suggests "
                            f"${-expected_total:.2f} "
                            f"({shares:.4f} shares * "
                            f"${value_per_share_market:.4f})."
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
                        f"   Extra user dividend on {row['date'].date()} "
                        f"for {symbol} "
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
                                    "value_per_share_market": [0.0],
                                    "quantity": [shares],
                                    "total_value_market": [0.0],
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
                        f"Missing dividend on {row['date'].date()} for "
                        f"{symbol} "
                        f"in account {account if account else 'ALL'}: "
                        f"Market data shows a dividend of "
                        f"${value_per_share_market:.4f}/share, "
                        f"but none were by recorded user."
                    )
                    messages.append(msg)
                    dividend_differences_df = pd.concat(
                        [
                            dividend_differences_df,
                            pd.DataFrame(
                                {
                                    "date": [row["date"]],
                                    "account": [account],
                                    "total_value_user": [0],
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
                dividend_differences[symbol] = dividend_differences_df

        if messages:
            logger.warning("Found %d dividend discrepancies.", len(messages))
        else:
            logger.info("✅ User dividends align with market data.")

        return messages, dividend_differences
