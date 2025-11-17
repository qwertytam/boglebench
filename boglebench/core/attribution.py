"""
Performance attribution analysis for portfolio holdings.

This module calculates how different holdings, accounts, or attribute groups
contributed to overall portfolio performance. Supports both database and
DataFrame data sources for backward compatibility.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from ..core.portfolio_db import PortfolioDatabase
from ..utils.logging_config import get_logger

logger = get_logger()


class AttributionCalculator:
    """
    Calculates performance attribution based on detailed portfolio history.

    Supports both normalized database and legacy DataFrame sources.
    """

    def __init__(
        self,
        transactions: Optional[pd.DataFrame] = None,
        attrib_group_cols: Optional[List[str]] = None,
        portfolio_db: Optional[PortfolioDatabase] = None,  # NEW
    ):
        """
        Initialize the AttributionCalculator.

        Args:
            transactions: DataFrame containing transaction data (optional)
            attrib_group_cols: List of column names for grouping attributes (optional)
            portfolio_db: PortfolioDatabase for normalized data access (preferred)
        """
        self.transactions = transactions
        self.attrib_group_cols = attrib_group_cols or []
        self.portfolio_db = portfolio_db  # NEW
        self.symbol_to_groups_map = self._create_symbol_map()

    def _create_symbol_map(self) -> dict:
        """Creates a map from symbol to its associated factor groups."""
        if not self.attrib_group_cols or self.transactions is None:
            return {}

        factor_cols = sorted(list(set(self.attrib_group_cols)))
        if not factor_cols:
            return {}
        return (
            self.transactions.drop_duplicates(subset=["symbol"], keep="last")
            .set_index("symbol")[factor_cols]
            .to_dict("index")
        )

    def calculate(self, group_by: str) -> pd.DataFrame:
        """
        Calculate performance attribution for a given grouping.

        Args:
            group_by: Grouping dimension ('symbol', 'account', or custom attribute)

        Returns:
            DataFrame with attribution analysis
        """
        if self.portfolio_db is not None:
            try:
                return self._calculate_from_database(group_by)
            except Exception as e:  # pylint: disable=broad-except
                logger.warning(
                    "Database calculation failed: %s",
                    e,
                )

        logger.error("No data source available for attribution calculation")
        return pd.DataFrame()

    def _calculate_from_database(self, group_by: str) -> pd.DataFrame:
        """Calculate attribution using database queries."""

        if group_by == "symbol":
            df = self._calculate_symbol_attribution_from_db()

        elif group_by == "account":
            df = self._calculate_account_attribution_from_db()

        elif group_by in [
            "asset_class",
            "geography",
            "region",
            "sector",
            "style",
            "market_cap",
            "fund_type",
        ]:
            df = self._calculate_attribute_attribution_from_db(group_by)

        else:
            logger.error("Unsupported group_by value: %s", group_by)
            df = pd.DataFrame()

        return df

    def _calculate_cumulative_return(self, returns: pd.Series) -> float:
        """Calculate cumulative return from a series of periodic returns."""
        prod_result = (1 + returns.fillna(0)).prod()
        if isinstance(prod_result, (int, float, np.number)):
            cumulative_return = float(prod_result) - 1.0
        else:
            logger.error(
                "AttributionCalculator: Unexpected product result type: %s",
                type(prod_result),
            )
            cumulative_return = 0.0

        return cumulative_return

    def _calculate_symbol_attribution_from_db(self) -> pd.DataFrame:
        """Calculate attribution by symbol using database."""
        if self.portfolio_db is None:
            return pd.DataFrame()

        symbol_df = self.portfolio_db.get_symbol_data()

        if symbol_df.empty:
            return pd.DataFrame()

        # Calculate metrics by symbol
        summary_data = []
        for symbol in symbol_df["symbol"].unique():
            sym_data = symbol_df[symbol_df["symbol"] == symbol]

            # Average weight
            avg_weight = sym_data["weight"].mean()

            # Time-weighted return (compound returns)
            twr = (1 + sym_data["twr_return"].fillna(0)).prod() - 1

            # Contribution to portfolio return
            contribution = avg_weight * twr

            summary_data.append(
                {
                    "Symbol": symbol,
                    "Avg. Weight": avg_weight,
                    "Return (TWR)": twr,
                    "Contribution to Portfolio Return": contribution,
                }
            )

        result_df = pd.DataFrame(summary_data)

        # Add benchmark comparison if available
        portfolio_summary = self.portfolio_db.get_portfolio_summary()
        if (
            not portfolio_summary.empty
            and "benchmark_return" in portfolio_summary.columns
        ):
            benchmark_twr = self._calculate_cumulative_return(
                portfolio_summary["benchmark_return"]
            )
            result_df["Excess Return vs. Benchmark"] = (
                result_df["Return (TWR)"] - benchmark_twr
            )
            result_df["Contribution to Excess Return"] = (
                result_df["Avg. Weight"]
                * result_df["Excess Return vs. Benchmark"]
            )

        result_df = result_df.set_index("Symbol")
        return result_df.sort_values("Avg. Weight", ascending=False)

    def _calculate_account_attribution_from_db(self) -> pd.DataFrame:
        """Calculate attribution by account using database."""
        if self.portfolio_db is None:
            return pd.DataFrame()

        account_df = self.portfolio_db.get_account_data()

        if account_df.empty:
            return pd.DataFrame()

        # Calculate metrics by account
        summary_data = []
        for account in account_df["account"].unique():
            acc_data = account_df[account_df["account"] == account]

            # Average weight
            avg_weight = acc_data["weight"].mean()

            # Time-weighted return
            twr = (1 + acc_data["twr_return"].fillna(0)).prod() - 1

            # Contribution to portfolio return
            contribution = avg_weight * twr

            summary_data.append(
                {
                    "Account": account,
                    "Avg. Weight": avg_weight,
                    "Return (TWR)": twr,
                    "Contribution to Portfolio Return": contribution,
                }
            )

        result_df = pd.DataFrame(summary_data)

        # Add benchmark comparison if available
        portfolio_summary = self.portfolio_db.get_portfolio_summary()
        if (
            not portfolio_summary.empty
            and "benchmark_return" in portfolio_summary.columns
        ):
            benchmark_twr = self._calculate_cumulative_return(
                portfolio_summary["benchmark_return"]
            )
            result_df["Excess Return vs. Benchmark"] = (
                result_df["Return (TWR)"] - benchmark_twr
            )
            result_df["Contribution to Excess Return"] = (
                result_df["Avg. Weight"]
                * result_df["Excess Return vs. Benchmark"]
            )

        result_df = result_df.set_index("Account")
        return result_df.sort_values("Avg. Weight", ascending=False)

    def _calculate_attribute_attribution_from_db(
        self, attribute: str
    ) -> pd.DataFrame:
        """
        Calculate attribution by symbol attribute (geography, sector, etc.).

        Uses temporal symbol attributes to correctly attribute performance
        even when attributes change over time.
        """
        if self.portfolio_db is None:
            return pd.DataFrame()

        # Get symbol data with temporal attributes
        symbol_df = self.portfolio_db.get_symbol_data()

        if symbol_df.empty:
            return pd.DataFrame()

        # For each date and symbol, get the attribute value that was effective
        attribution_data = []

        for date in symbol_df["date"].unique():
            date_symbols = symbol_df[symbol_df["date"] == date]

            # Get attributes as of this date
            attributes_at_date = (
                self.portfolio_db.get_symbol_attributes_at_date(date)
            )

            if attributes_at_date.empty:
                continue

            # Merge symbol data with attributes
            merged = date_symbols.merge(
                attributes_at_date[["symbol", attribute]],
                on="symbol",
                how="left",
            )

            # Group by attribute category
            for category in merged[attribute].dropna().unique():
                category_data = merged[merged[attribute] == category]

                if not category_data.empty:
                    attribution_data.append(
                        {
                            "date": date,
                            "category": category,
                            "weight": category_data["weight"].sum(),
                            "value": category_data["total_value"].sum(),
                            "weighted_return": (
                                (
                                    category_data["twr_return"]
                                    * category_data["weight"]
                                ).sum()
                                / category_data["weight"].sum()
                                if category_data["weight"].sum() > 0
                                else 0
                            ),
                        }
                    )

        if not attribution_data:
            return pd.DataFrame()

        attr_df = pd.DataFrame(attribution_data)

        # Calculate summary metrics by category
        summary_data = []
        for category in attr_df["category"].unique():
            cat_data = attr_df[attr_df["category"] == category]

            # Average weight over time
            avg_weight = cat_data["weight"].mean()

            # Compound return
            twr = (1 + cat_data["weighted_return"]).prod() - 1

            # Contribution
            contribution = avg_weight * twr

            summary_data.append(
                {
                    "Category": category,
                    "Avg. Weight": avg_weight,
                    "Return (TWR)": twr,
                    "Contribution to Portfolio Return": contribution,
                }
            )

        result_df = pd.DataFrame(summary_data)

        # Add benchmark comparison if available
        portfolio_summary = self.portfolio_db.get_portfolio_summary()
        if (
            not portfolio_summary.empty
            and "benchmark_return" in portfolio_summary.columns
        ):
            benchmark_twr = self._calculate_cumulative_return(
                portfolio_summary["benchmark_return"]
            )
            result_df["Excess Return vs. Benchmark"] = (
                result_df["Return (TWR)"] - benchmark_twr
            )
            result_df["Contribution to Excess Return"] = (
                result_df["Avg. Weight"]
                * result_df["Excess Return vs. Benchmark"]
            )

        result_df = result_df.set_index("Category")
        return result_df.sort_values("Avg. Weight", ascending=False)

    def _calculate_daily_twr(
        self,
        group_values: pd.DataFrame,
        group_cash_flows: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate time-weighted returns for each group (legacy DataFrame method)."""
        group_returns = pd.DataFrame(index=group_values.index)

        for group in group_values.columns:
            values = group_values[group]
            cash_flows = (
                group_cash_flows[group]
                if group in group_cash_flows.columns
                else pd.Series(0, index=values.index)
            )

            # Calculate TWR
            beginning_values = values.shift(1).fillna(0)
            returns = (
                (values - cash_flows - beginning_values) / beginning_values
            ).fillna(0)
            returns = returns.replace([np.inf, -np.inf], 0)

            group_returns[group] = returns

        return group_returns
