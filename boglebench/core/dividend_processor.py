"""
Dividend validation and processing.

This module validates user-provided dividend transactions against market data
from Alpha Vantage and corrects any discrepancies. Handles both cash dividends
and dividend reinvestments, ensuring accuracy in portfolio return calculations.
"""

from typing import Optional

import pandas as pd

from ..core.dividend_validator import DividendValidator
from ..core.helpers import (
    get_shares_held_on_date,
    identify_any_dividend_transactions,
)
from ..utils.config import ConfigManager
from ..utils.logging_config import get_logger


class DividendProcessor:
    """Validates and corrects dividend data in a transactions DataFrame."""

    def __init__(
        self,
        config: ConfigManager,
        transactions_df: pd.DataFrame,
        market_data: dict,
        start_date=None,
        end_date=None,
    ):
        """
        Initialize the DividendProcessor.

        Args:
            config: ConfigManager with dividend validation settings
            transactions_df: DataFrame containing transaction data
            market_data: Dictionary mapping symbols to their market data
            start_date: Optional start date for dividend validation
            end_date: Optional end date for dividend validation
        """
        self.config = config
        self.transactions_df = transactions_df.copy()
        self.market_data = market_data
        self.start_date = start_date
        self.end_date = end_date
        self.logger = get_logger()

    def run(self) -> pd.DataFrame:
        """
        Validates dividends and applies corrections if configured to do so.
        Returns the (potentially modified) transactions DataFrame.
        """
        if not self.config.get("dividend.auto_validate", True):
            return self.transactions_df

        self.logger.debug("🔍 Validating dividend data against market data...")
        if not self.market_data:
            self.logger.warning(
                "⚠️ No market data available for dividend validation."
            )
            return self.transactions_df

        validator = DividendValidator(
            self.transactions_df,
            self.market_data,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        messages, diffs = validator.validate()

        self._log_validation_messages(messages)

        if (
            self.config.get("dividend.auto_calculate_div_per_share", True)
            and diffs
        ):
            self._apply_corrections(diffs)

        # Handle future dividends based on configuration
        future_div_mode = self.config.get(
            "dividend.handle_future_dividends", "ignore"
        )
        if future_div_mode == "add_to_all_accounts":
            self._add_future_dividends()

        return self.transactions_df

    def _log_validation_messages(self, messages: list[str]):
        """Logs messages from the DividendValidator."""
        warn_missing = self.config.get("dividend.warn_missing_dividends", True)
        if not warn_missing:
            return

        for msg in messages:
            if any(
                keyword in msg.lower()
                for keyword in ["mismatch", "warning", "missing", "extra"]
            ):
                self.logger.warning("⚠️ %s", msg)
            else:
                self.logger.info("ℹ️ %s", msg)

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

    def _apply_corrections(self, differences: dict):
        """Applies dividend corrections to the transactions DataFrame."""
        self.logger.info(
            "Automatically adjusting user dividends based on market data..."
        )
        for symbol, df in differences.items():
            for _, row in df.iterrows():
                date = row["date"]
                new_div_per_share = row["value_per_share_market"]
                account = row.get("account", None)

                shares = self._get_shares_held_on_date(symbol, date, account)

                mask = (
                    (self.transactions_df["symbol"] == symbol)
                    & (self.transactions_df["date"].dt.date == date.date())
                    & (
                        identify_any_dividend_transactions(
                            self.transactions_df["transaction_type"]
                        )
                    )
                )
                if account:
                    mask &= self.transactions_df["account"] == account

                if not self.transactions_df.loc[mask].empty:
                    old_total = self.transactions_df.loc[mask, "total_value"]
                    if isinstance(old_total, pd.Series):
                        old_total_value = old_total.sum()
                    else:
                        old_total_value = old_total

                    # Negative because dividends are cash outflows from the
                    # portfolio perspective
                    new_total_value = -shares * new_div_per_share

                    self.logger.info(
                        "Updating %s dividend on %s: total from $%.2f to $%.2f",
                        symbol,
                        date.date(),
                        old_total_value,
                        new_total_value,
                    )
                    # This logic assumes one dividend event per day per symbol.
                    # More complex scenarios might need more granular updates.
                    self.transactions_df.loc[mask, "value_per_share"] = (
                        new_div_per_share
                    )
                    self.transactions_df.loc[mask, "total_value"] = (
                        new_total_value
                    )

    def _get_last_transaction_date(self, symbol: Optional[str] = None) -> pd.Timestamp:
        """
        Get the last transaction date for a symbol, or overall if symbol not specified.
        
        Args:
            symbol: Optional symbol to filter by. If None, returns last date across all symbols.
            
        Returns:
            Last transaction date as Timestamp
        """
        if symbol:
            symbol_txns = self.transactions_df[self.transactions_df["symbol"] == symbol]
            if symbol_txns.empty:
                return self.start_date if self.start_date else pd.Timestamp.min
            return symbol_txns["date"].max()
        else:
            return self.transactions_df["date"].max()

    def _get_accounts_holding_symbol(self, symbol: str, date: pd.Timestamp) -> list:
        """
        Get list of accounts that have holdings of a symbol on a given date.
        
        Args:
            symbol: The symbol to check
            date: The date to check holdings
            
        Returns:
            List of account names with positive holdings
        """
        accounts = self.transactions_df["account"].unique()
        holding_accounts = []
        
        for account in accounts:
            shares = self._get_shares_held_on_date(symbol, date, account)
            if shares > 0:
                holding_accounts.append(account)
        
        return holding_accounts

    def _add_future_dividends(self):
        """
        Add dividend transactions for market dividends that occur after the last
        transaction date but before end_date. Only adds to accounts holding the symbol.
        """
        if not self.end_date:
            return

        last_txn_date = self._get_last_transaction_date()
        self.logger.info(
            "🔍 Checking for dividends after last transaction date (%s) through end date (%s)",
            last_txn_date.date(),
            self.end_date.date(),
        )

        all_symbols = self.transactions_df["symbol"].unique()
        new_transactions = []

        for symbol in all_symbols:
            # Get market dividends for this symbol
            market_data = self.market_data.get(symbol)
            if market_data is None or market_data.empty:
                continue

            # Filter for dividends after last transaction but before end_date
            market_dividends = market_data[
                (market_data["dividend"].abs() > 0)
                & (market_data["date"] > last_txn_date)
                & (market_data["date"] <= self.end_date)
            ]

            for _, div_row in market_dividends.iterrows():
                div_date = div_row["date"]
                div_per_share = div_row["dividend"]

                # Check if user already recorded this dividend
                existing_div = self.transactions_df[
                    (self.transactions_df["symbol"] == symbol)
                    & (self.transactions_df["date"].dt.date == div_date.date())
                    & (
                        identify_any_dividend_transactions(
                            self.transactions_df["transaction_type"]
                        )
                    )
                ]

                if not existing_div.empty:
                    # User already has this dividend recorded
                    continue

                # Find accounts holding this symbol at dividend date
                holding_accounts = self._get_accounts_holding_symbol(symbol, div_date)

                for account in holding_accounts:
                    shares = self._get_shares_held_on_date(symbol, div_date, account)
                    if shares <= 0:
                        continue

                    # Negative because dividends are cash inflows to investor
                    # (outflows from portfolio perspective)
                    total_value = -shares * div_per_share

                    new_txn = {
                        "date": div_date,
                        "symbol": symbol,
                        "transaction_type": "DIVIDEND",
                        "quantity": 0,
                        "value_per_share": div_per_share,
                        "total_value": total_value,
                        "account": account,
                        "div_type": self.config.get(
                            "dividend.default_div_type", "CASH"
                        ),
                        "div_pay_date": div_date,
                        "div_record_date": pd.NaT,
                        "div_ex_date": pd.NaT,
                        "split_ratio": 0,
                        "notes": "Auto-added future dividend",
                    }
                    new_transactions.append(new_txn)
                    
                    self.logger.info(
                        "➕ Adding future dividend: %s on %s for account %s: $%.2f (%s shares × $%.4f)",
                        symbol,
                        div_date.date(),
                        account,
                        -total_value,
                        shares,
                        div_per_share,
                    )

        if new_transactions:
            # Add new transactions to the DataFrame
            new_df = pd.DataFrame(new_transactions)
            self.transactions_df = pd.concat(
                [self.transactions_df, new_df], ignore_index=True
            )
            # Re-sort by date
            self.transactions_df = self.transactions_df.sort_values("date").reset_index(drop=True)
            self.logger.info(
                "✅ Added %d future dividend transactions", len(new_transactions)
            )
        else:
            self.logger.info(
                "ℹ️  No future dividends to add (all already recorded or no holdings)"
            )
