"""
Performance attribution analysis for portfolio holdings.

This module calculates how different holdings, accounts, or attribute groups
contributed to overall portfolio performance using database as the single source of truth.
"""

import numpy as np
import pandas as pd

from ..core.portfolio_db import PortfolioDatabase
from ..utils.logging_config import get_logger

logger = get_logger()


class AttributionCalculator:
    """
    Calculates performance attribution based on detailed portfolio history.

    """

    def __init__(
        self,
        portfolio_db: PortfolioDatabase,
    ):
        """
        Initialize the AttributionCalculator.

        Args:
            portfolio_db: PortfolioDatabase for normalized data access (required)

        Raises:
            ValueError: If portfolio_db is None
        """
        if portfolio_db is None:
            raise ValueError(
                "portfolio_db is required for attribution calculation"
            )
        self.portfolio_db = portfolio_db

    def calculate(self, group_by: str) -> pd.DataFrame:
        """
        Calculate performance attribution for a given grouping.

        Args:
            group_by: Grouping dimension ('symbol', 'account', or custom attribute)

        Returns:
            DataFrame with attribution analysis
        """
        return self._calculate_from_database(group_by)

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
        even when attributes change over time. This is a vectorized implementation
        that eliminates per-date database queries for better performance.
        """
        if self.portfolio_db is None:
            return pd.DataFrame()

        # Get symbol data with temporal attributes
        symbol_df = self.portfolio_db.get_symbol_data()

        if symbol_df.empty:
            return pd.DataFrame()

        # ✅ GET ALL ATTRIBUTE HISTORY ONCE instead of per-date queries
        all_attributes = self.portfolio_db.get_symbol_attributes(
            include_history=True
        )

        if all_attributes.empty:
            return pd.DataFrame()

        # Check if attribute exists
        if attribute not in all_attributes.columns:
            logger.warning(
                "Attribute '%s' not found in symbol attributes", attribute
            )
            return pd.DataFrame()

        # Ensure dates are timezone-aware for comparison
        if "effective_date" in all_attributes.columns:
            all_attributes["effective_date"] = pd.to_datetime(
                all_attributes["effective_date"], utc=True
            )
        if "end_date" in all_attributes.columns:
            all_attributes["end_date"] = pd.to_datetime(
                all_attributes["end_date"], utc=True
            )

        symbol_df["date"] = pd.to_datetime(symbol_df["date"], utc=True)

        # ✅ VECTORIZED: Build attribute mapping using efficient lookups
        # Group attributes by symbol for efficient lookup
        attr_by_symbol = {
            symbol: group for symbol, group in all_attributes.groupby("symbol")
        }

        # Build results using optimized lookups
        results = []
        for symbol, symbol_group in symbol_df.groupby("symbol"):
            if symbol not in attr_by_symbol:
                continue

            symbol_attrs = attr_by_symbol[symbol]

            # Use itertuples() for better performance than iterrows()
            for row in symbol_group.itertuples(index=False):
                date = row.date

                # Find attributes valid at this date
                valid = symbol_attrs[
                    (symbol_attrs["effective_date"] <= date)
                    & (
                        (symbol_attrs["end_date"].isna())
                        | (symbol_attrs["end_date"] >= date)
                    )
                ]

                if not valid.empty:
                    # Get most recent attribute value
                    attr_value = valid.sort_values(
                        "effective_date", ascending=False
                    ).iloc[0][attribute]
                    if pd.notna(attr_value):
                        results.append(
                            {
                                "date": date,
                                "category": attr_value,
                                "weight": row.weight,
                                "total_value": row.total_value,
                                "twr_return": row.twr_return,
                            }
                        )

        if not results:
            return pd.DataFrame()

        merged = pd.DataFrame(results)

        # ✅ VECTORIZED: Calculate weighted returns for all rows at once
        merged["weighted_return"] = merged["twr_return"] * merged["weight"]

        # ✅ VECTORIZED: Group by date and category in one operation
        attr_df = (
            merged.groupby(["date", "category"])
            .agg(
                {
                    "weight": "sum",
                    "total_value": "sum",
                    "weighted_return": "sum",  # Sum of weighted returns
                }
            )
            .reset_index()
        )

        # ✅ VECTORIZED: Calculate weighted average return (avoiding division by zero)
        attr_df["twr_return"] = np.where(
            attr_df["weight"] > 0,
            attr_df["weighted_return"] / attr_df["weight"],
            0,
        )

        # Drop the intermediate weighted_return column
        attr_df = attr_df.drop(columns=["weighted_return"])

        # ✅ VECTORIZED: Calculate summary metrics by category
        # Use a more efficient compound return calculation
        def compound_return(returns):
            """Calculate compound return from a series of period returns."""
            return np.prod(1 + returns.values) - 1

        summary = (
            attr_df.groupby("category")
            .agg(
                {
                    "weight": "mean",  # Average weight over time
                    "twr_return": compound_return,  # Compound return
                }
            )
            .reset_index()
        )

        # Calculate contribution
        summary["contribution"] = summary["weight"] * summary["twr_return"]

        # Format result
        result_df = pd.DataFrame(
            {
                "Category": summary["category"],
                "Avg. Weight": summary["weight"],
                "Return (TWR)": summary["twr_return"],
                "Contribution to Portfolio Return": summary["contribution"],
            }
        )

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
