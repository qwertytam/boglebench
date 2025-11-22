"""
Performance attribution analysis for portfolio holdings.

This module calculates how different holdings, accounts, or attribute groups
contributed to overall portfolio performance using database as the single source of truth.
"""

import threading

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

        # Add caching for repeated data access
        self._cache_lock = threading.Lock()
        self._symbol_data_cache = None
        self._account_data_cache = None
        self._attributes_cache = None

        # Pre-cache common data
        self._precache_data()

    def _precache_data(self):
        """Pre-fetch data to speed up calculations."""
        logger.debug("🔄 Pre-caching data for attribution calculations...")
        self._get_symbol_data()
        self._get_account_data()
        self._get_attributes()
        logger.debug("✅ Attribution pre-caching complete")

    def _get_symbol_data(self) -> pd.DataFrame:
        """Get symbol data with caching."""
        if self._symbol_data_cache is None:
            with self._cache_lock:
                if self._symbol_data_cache is None:
                    self._symbol_data_cache = (
                        self.portfolio_db.get_symbol_data()
                    )
        return self._symbol_data_cache

    def _get_account_data(self) -> pd.DataFrame:
        """Get account data with caching."""
        if self._account_data_cache is None:
            with self._cache_lock:
                if self._account_data_cache is None:
                    self._account_data_cache = (
                        self.portfolio_db.get_account_data()
                    )
        return self._account_data_cache

    def _get_attributes(self) -> pd.DataFrame:
        """Get attributes with caching."""
        if self._attributes_cache is None:
            with self._cache_lock:
                if self._attributes_cache is None:
                    self._attributes_cache = (
                        self.portfolio_db.get_symbol_attributes()
                    )
        return self._attributes_cache

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
        """Calculate attribution by symbol (VECTORIZED)."""
        if self.portfolio_db is None:
            return pd.DataFrame()

        symbol_df = self._get_symbol_data()

        if symbol_df.empty:
            return pd.DataFrame()

        # VECTORIZED: Calculate all metrics at once
        result_df = symbol_df.groupby("symbol", as_index=False).agg(
            {
                "weight": "mean",
                "twr_return": lambda x: (1 + x.fillna(0)).prod() - 1,
            }
        )

        # Calculate contribution
        result_df["contribution"] = (
            result_df["weight"] * result_df["twr_return"]
        )

        # Rename columns
        result_df = result_df.rename(
            columns={
                "symbol": "Symbol",
                "weight": "Avg. Weight",
                "twr_return": "Return (TWR)",
                "contribution": "Contribution to Portfolio Return",
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

        result_df = result_df.set_index("Symbol")
        return result_df.sort_values("Avg. Weight", ascending=False)

    def _calculate_account_attribution_from_db(self) -> pd.DataFrame:
        """
        Calculate attribution by account (FULLY VECTORIZED).

        Performance improvement: Eliminates account-by-account iteration by using
        pandas groupby operations to process all accounts at once.

        Returns:
            DataFrame with attribution analysis indexed by account name
        """
        if self.portfolio_db is None:
            return pd.DataFrame()

        account_df = self._get_account_data()

        if account_df.empty:
            return pd.DataFrame()

        # ✅ VECTORIZED: Calculate all metrics at once using groupby
        result_df = account_df.groupby("account", as_index=False).agg(
            {
                "weight": "mean",  # Average weight over time
                "twr_return": lambda x: (1 + x.fillna(0)).prod()
                - 1,  # Compound return
            }
        )

        # Calculate contribution
        result_df["contribution"] = (
            result_df["weight"] * result_df["twr_return"]
        )

        # Rename columns to match expected output format
        result_df = result_df.rename(
            columns={
                "account": "Account",
                "weight": "Avg. Weight",
                "twr_return": "Return (TWR)",
                "contribution": "Contribution to Portfolio Return",
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

        # Get ALL symbol data at once (using cache)
        symbol_df = self._get_symbol_data()

        if symbol_df.empty:
            return pd.DataFrame()

        # Get ALL attributes at once (using cache)
        all_attributes = self._get_attributes()

        if all_attributes.empty:
            return pd.DataFrame()

        if attribute not in all_attributes.columns:
            logger.warning(
                "Attribute '%s' not found in symbol attributes", attribute
            )
            return pd.DataFrame()

        # VECTORIZED: Single merge for all dates
        merged = symbol_df.merge(
            all_attributes[["symbol", attribute]],
            on="symbol",
            how="left",
        )

        # Drop rows where attribute is null
        merged = merged.dropna(subset=[attribute])

        if merged.empty:
            return pd.DataFrame()

        # VECTORIZED: Calculate weighted returns for all rows at once
        merged["weighted_value"] = merged["twr_return"] * merged["weight"]

        # VECTORIZED: Group by date and category in a single operation
        daily_attribution = merged.groupby(
            ["date", attribute], as_index=False
        ).agg(
            {
                "weight": "sum",
                "total_value": "sum",
                "weighted_value": "sum",
            }
        )

        # VECTORIZED: Calculate weighted return for each group
        daily_attribution["weighted_return"] = (
            daily_attribution["weighted_value"] / daily_attribution["weight"]
        ).fillna(0)

        daily_attribution = daily_attribution.drop(columns=["weighted_value"])
        daily_attribution = daily_attribution.rename(
            columns={attribute: "category"}
        )

        # VECTORIZED: Calculate summary metrics by category
        summary = daily_attribution.groupby("category", as_index=False).agg(
            {
                "weight": "mean",  # Average weight over time
                "weighted_return": lambda x: (1 + x).prod()
                - 1,  # Compound return
            }
        )

        # Calculate contribution
        summary["contribution"] = (
            summary["weight"] * summary["weighted_return"]
        )

        # Format result DataFrame
        result_df = pd.DataFrame(
            {
                "Category": summary["category"],
                "Avg. Weight": summary["weight"],
                "Return (TWR)": summary["weighted_return"],
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
