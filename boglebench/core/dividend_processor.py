"""
Handles dividend validation and correction.
"""

import pandas as pd

from ..core.dividend_validator import (
    DividendValidator,
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
    ):
        self.config = config
        self.transactions_df = transactions_df.copy()
        self.market_data = market_data
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

        validator = DividendValidator(self.transactions_df, self.market_data)
        messages, diffs = validator.validate()

        self._log_validation_messages(messages)

        if (
            self.config.get("dividend.auto_calculate_div_per_share", True)
            and diffs
        ):
            self._apply_corrections(diffs)

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

    def _apply_corrections(self, differences: dict):
        """Applies dividend corrections to the transactions DataFrame."""
        self.logger.info(
            "Automatically adjusting user dividends based on market data..."
        )
        for ticker, df in differences.items():
            for _, row in df.iterrows():
                date = row["date"]
                new_div_per_share = row["value_per_share_market"]
                mask = (
                    (self.transactions_df["ticker"] == ticker)
                    & (self.transactions_df["date"].dt.date == date.date())
                    & (
                        identify_any_dividend_transactions(
                            self.transactions_df["transaction_type"]
                        )
                    )
                )

                if not self.transactions_df.loc[mask].empty:
                    quantity = self.transactions_df.loc[mask, "quantity"]
                    if isinstance(quantity, pd.Series):
                        shares = quantity.sum()
                    else:
                        shares = quantity

                    old_total = self.transactions_df.loc[mask, "total_value"]
                    if isinstance(old_total, pd.Series):
                        old_total_value = old_total.sum()
                    else:
                        old_total_value = old_total

                    new_total_value = shares * new_div_per_share

                    self.logger.info(
                        "Updating %s dividend on %s: total from $%.2f to $%.2f",
                        ticker,
                        date.date(),
                        old_total_value,
                        new_total_value,
                    )
                    # This logic assumes one dividend event per day per ticker.
                    # More complex scenarios might need more granular updates.
                    self.transactions_df.loc[mask, "value_per_share"] = (
                        new_div_per_share
                    )
                    self.transactions_df.loc[mask, "total_value"] = (
                        new_total_value
                    )
