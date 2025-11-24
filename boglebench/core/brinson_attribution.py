"""
Brinson-Fachler attribution analysis implementation.

This module implements the Brinson-Fachler attribution methodology to decompose
portfolio returns into allocation and selection effects. Helps identify whether
outperformance came from sector allocation decisions or individual security
selection using database as the single source of truth.
"""

import threading
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..core.portfolio_db import PortfolioDatabase
from ..utils.logging_config import get_logger
from ..utils.performance import timeit

logger = get_logger()


class BrinsonAttributionCalculator:
    """
    Implements Brinson-Fachler attribution analysis.

    """

    def __init__(
        self,
        benchmark_history: pd.DataFrame,
        portfolio_db: PortfolioDatabase,
    ):
        """
        Initialize the BrinsonAttributionCalculator.

        Args:
            benchmark_history: DataFrame with benchmark performance data (required)
            portfolio_db: PortfolioDatabase for normalized data access (required)

        Raises:
            ValueError: If either parameter is None
        """
        if benchmark_history is None or benchmark_history.empty:
            raise ValueError("benchmark_history is required")
        if portfolio_db is None:
            raise ValueError("portfolio_db is required")

        self.benchmark_history = benchmark_history
        self.portfolio_db = portfolio_db

        # Add caching for database queries (optimization)
        # Thread-safe caching using lock for parallel execution
        self._cache_lock = threading.Lock()
        self._symbol_data_cache: pd.DataFrame | None = None
        self._attributes_cache: Dict[str, pd.DataFrame] = {}

        # Pre-cache all data to avoid delays in parallel execution
        self._precache_data()

    def _precache_data(self):
        """Pre-fetch and cache all data needed for calculations."""
        logger.debug("🔄 Pre-caching data for Brinson calculations...")
        start = time.time()

        # Pre-load all data that will be needed
        self._get_symbol_data()
        self._get_all_attributes(include_history=False)
        self._get_all_attributes(include_history=True)

        elapsed = time.time() - start
        logger.debug("✅ Pre-caching complete in %.2fs", elapsed)

    def _get_symbol_data(self) -> pd.DataFrame:
        """Get symbol data with thread-safe caching to avoid repeated database queries."""
        cache_hit = self._symbol_data_cache is not None
        logger.debug(
            "Symbol data cache %s", "HIT ✅" if cache_hit else "MISS ❌"
        )

        if self._symbol_data_cache is None:
            with self._cache_lock:
                # Double-check pattern to avoid race conditions
                if self._symbol_data_cache is None:
                    logger.debug("Fetching symbol data from database...")
                    self._symbol_data_cache = (
                        self.portfolio_db.get_symbol_data()
                    )
                # Return from within lock to avoid race condition
                if self._symbol_data_cache is None:
                    return pd.DataFrame()
                return self._symbol_data_cache
        return self._symbol_data_cache

    def _get_all_attributes(
        self, include_history: bool = False
    ) -> pd.DataFrame:
        """Get attributes with thread-safe caching to avoid repeated database queries."""
        cache_key = f"history_{include_history}"
        cache_hit = cache_key in self._attributes_cache
        logger.debug(
            "Attributes cache (history=%s) %s",
            include_history,
            "HIT ✅" if cache_hit else "MISS ❌",
        )

        if cache_key not in self._attributes_cache:
            with self._cache_lock:
                # Double-check pattern to avoid race conditions
                if cache_key not in self._attributes_cache:
                    logger.debug("Fetching attributes from database...")
                    self._attributes_cache[cache_key] = (
                        self.portfolio_db.get_symbol_attributes(
                            include_history=include_history
                        )
                    )
                # Return from within lock to avoid race condition
                return self._attributes_cache[cache_key]
        return self._attributes_cache[cache_key]

    @timeit
    def calculate(
        self, group_by: str
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Perform Brinson attribution analysis.

        Args:
            group_by: Attribute to group by (e.g., 'geography', 'sector')

        Returns:
            Tuple of (summary_df, selection_drilldown)
        """
        return self._calculate_from_database(group_by)

    def _calculate_from_database(
        self, group_by: str
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Calculate Brinson attribution using database queries."""

        # Validate attribute
        valid_attributes = [
            "asset_class",
            "geography",
            "region",
            "sector",
            "style",
            "market_cap",
            "fund_type",
        ]
        if group_by not in valid_attributes:
            raise ValueError(
                f"Invalid group_by: {group_by}. "
                f"Must be one of: {', '.join(valid_attributes)}"
            )

        # Get portfolio data with attributes
        portfolio_data = self._get_portfolio_data_with_attributes(group_by)

        # Get benchmark data with attributes
        benchmark_data = self._get_benchmark_data_with_attributes(group_by)

        if portfolio_data.empty or benchmark_data.empty:
            logger.warning("Insufficient data for Brinson attribution")
            return pd.DataFrame(), {}

        # Calculate Brinson components
        summary_df = self._calculate_brinson_components(
            portfolio_data, benchmark_data
        )

        # Calculate selection drilldown
        selection_drilldown = self._calculate_selection_drilldown_from_db(
            group_by, portfolio_data, benchmark_data
        )

        return summary_df, selection_drilldown

    @timeit
    def _get_portfolio_data_with_attributes(
        self, attribute: str
    ) -> pd.DataFrame:
        """Get portfolio data grouped by attribute (FULLY VECTORIZED)."""

        if self.portfolio_db is None:
            return pd.DataFrame()

        # Get all symbol data (using cache)
        symbol_data = self._get_symbol_data()

        if symbol_data.empty:
            return pd.DataFrame()

        # Get ALL attributes once (using cache)
        all_attributes = self._get_all_attributes(include_history=False)

        if all_attributes.empty or attribute not in all_attributes.columns:
            logger.warning(
                "Attribute '%s' not found or no attributes available",
                attribute,
            )
            return pd.DataFrame()

        # VECTORIZED: Single merge for all dates
        merged = symbol_data.merge(
            all_attributes[["symbol", attribute]],
            on="symbol",
            how="left",
        )

        # Drop rows with null attributes
        merged = merged.dropna(subset=[attribute])

        if merged.empty:
            return pd.DataFrame()

        # VECTORIZED: Calculate weighted return for all rows
        merged["weighted_return"] = merged["twr_return"] * merged["weight"]

        # VECTORIZED: Group by date and category in one operation
        result_df = merged.groupby(["date", attribute], as_index=False).agg(
            {
                "weight": "sum",
                "weighted_return": "sum",
            }
        )

        # VECTORIZED: Use numpy for better performance (no apply!)
        result_df["twr_return"] = np.where(
            result_df["weight"] > 0,
            result_df["weighted_return"] / result_df["weight"],
            0,
        )

        result_df = result_df.drop(columns=["weighted_return"])
        result_df = result_df.rename(columns={attribute: "category"})

        return result_df

    @timeit
    def _get_benchmark_data_with_attributes(
        self, attribute: str
    ) -> pd.DataFrame:
        """Get benchmark data grouped by attribute (FULLY VECTORIZED with merge_asof)."""

        if self.portfolio_db is None:
            return pd.DataFrame()

        if self.benchmark_history is None or self.benchmark_history.empty:
            logger.warning("No benchmark history available")
            return pd.DataFrame()

        # Extract benchmark component symbols from benchmark_history columns
        benchmark_symbols = [
            col.replace("_twr_return", "")
            for col in self.benchmark_history.columns
            if col.endswith("_twr_return") and not col.startswith("benchmark")
        ]

        if not benchmark_symbols:
            logger.warning("No benchmark symbols found in benchmark_history")
            return pd.DataFrame()

        # ✅ Get all attribute history for benchmark symbols at once (using cache)
        bench_attrs = self._get_all_attributes(include_history=True)
        # Filter and copy to avoid modifying cached data when we set timezone-aware dates below
        bench_attrs = bench_attrs.loc[
            bench_attrs["symbol"].isin(benchmark_symbols)
        ].copy()

        if bench_attrs.empty:
            logger.warning("No attributes found for benchmark symbols")
            return pd.DataFrame()

        # Check if attribute exists
        if attribute not in bench_attrs.columns:
            logger.warning(
                "Attribute '%s' not found for benchmark symbols", attribute
            )
            return pd.DataFrame()

        # Ensure dates are timezone-aware for comparison
        if "effective_date" in bench_attrs.columns:
            bench_attrs["effective_date"] = pd.to_datetime(
                bench_attrs["effective_date"], utc=True
            )
        if "end_date" in bench_attrs.columns:
            bench_attrs["end_date"] = pd.to_datetime(
                bench_attrs["end_date"], utc=True
            )

        # Convert benchmark_history to long format for easier processing
        # Get dates from index or column
        if "date" in self.benchmark_history.columns:
            bench_df = self.benchmark_history.copy()
            bench_df["date"] = pd.to_datetime(bench_df["date"], utc=True)
        else:
            # Date is in the index
            bench_df = self.benchmark_history.reset_index()
            bench_df.rename(columns={"index": "date"}, inplace=True)
            bench_df["date"] = pd.to_datetime(bench_df["date"], utc=True)

        # ✅ VECTORIZED: Convert wide format to long format using pd.melt
        # Extract weight and return columns for each symbol
        results_list = []

        for symbol in benchmark_symbols:
            weight_col = f"{symbol}_weight"
            return_col = f"{symbol}_twr_return"

            # Check if columns exist
            if (
                weight_col not in bench_df.columns
                or return_col not in bench_df.columns
            ):
                continue

            # Create long format dataframe for this symbol
            symbol_df = bench_df[["date", weight_col, return_col]].copy()
            symbol_df["symbol"] = symbol
            symbol_df.rename(
                columns={weight_col: "weight", return_col: "twr_return"},
                inplace=True,
            )

            # Filter out rows with NaN values
            symbol_df = symbol_df.dropna(subset=["weight", "twr_return"])

            if symbol_df.empty:
                continue

            # ✅ VECTORIZED: Use merge_asof for temporal join with attributes
            # Get attributes for this symbol
            symbol_attrs = bench_attrs[bench_attrs["symbol"] == symbol].copy()

            if symbol_attrs.empty:
                continue

            # Sort both dataframes for merge_asof
            symbol_df = symbol_df.sort_values("date")
            symbol_attrs = symbol_attrs.sort_values("effective_date")

            # Use merge_asof to find the most recent attribute for each date
            merged = pd.merge_asof(
                symbol_df,
                symbol_attrs[["effective_date", "end_date", attribute]],
                left_on="date",
                right_on="effective_date",
                direction="backward",
            )

            # Filter out rows where the attribute has expired (end_date < date)
            # Keep rows where end_date is NaN or end_date >= date
            merged = merged[
                merged["end_date"].isna()
                | (merged["end_date"] >= merged["date"])
            ]

            # Drop rows with null attributes
            merged = merged.dropna(subset=[attribute])

            if not merged.empty:
                # Keep only needed columns
                merged = merged[["date", attribute, "weight", "twr_return"]]
                merged.rename(columns={attribute: "category"}, inplace=True)
                results_list.append(merged)

        if not results_list:
            return pd.DataFrame()

        # ✅ VECTORIZED: Concatenate all results at once
        df = pd.concat(results_list, ignore_index=True)

        # ✅ VECTORIZED: Aggregate by date and category using vectorized operations
        df["weighted_return"] = df["twr_return"] * df["weight"]

        grouped = (
            df.groupby(["date", "category"])
            .agg(
                {
                    "weight": "sum",
                    "weighted_return": "sum",
                }
            )
            .reset_index()
        )

        # Calculate weighted average return (vectorized, avoiding division by zero)
        grouped["twr_return"] = np.where(
            grouped["weight"] != 0,
            grouped["weighted_return"] / grouped["weight"],
            0,
        )

        grouped = grouped.drop(columns=["weighted_return"])

        return grouped

    def _calculate_brinson_components(
        self,
        portfolio_data: pd.DataFrame,
        benchmark_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate Brinson allocation, selection, and interaction effects."""

        # Calculate average weights and returns for portfolio
        portfolio_summary = (
            portfolio_data.groupby("category")
            .agg(
                {
                    "weight": "mean",
                    "twr_return": lambda x: (1 + x).prod()
                    - 1,  # Compound return
                }
            )
            .rename(
                columns={
                    "weight": "portfolio_weight",
                    "twr_return": "portfolio_return",
                }
            )
        )

        # Calculate average weights and returns for benchmark
        benchmark_summary = (
            benchmark_data.groupby("category")
            .agg(
                {
                    "weight": "mean",
                    "twr_return": lambda x: (1 + x).prod()
                    - 1,  # Compound return
                }
            )
            .rename(
                columns={
                    "weight": "benchmark_weight",
                    "twr_return": "benchmark_return",
                }
            )
        )

        # Merge portfolio and benchmark
        combined = portfolio_summary.join(
            benchmark_summary, how="outer"
        ).fillna(0)

        # Calculate Brinson components
        attribution_results = {}

        for category in combined.index:
            p_w = float(
                pd.to_numeric(
                    combined.loc[category, "portfolio_weight"], errors="coerce"
                )
            )
            b_w = float(
                pd.to_numeric(
                    combined.loc[category, "benchmark_weight"], errors="coerce"
                )
            )
            p_r = float(
                pd.to_numeric(
                    combined.loc[category, "portfolio_return"], errors="coerce"
                )
            )
            b_r = float(
                pd.to_numeric(
                    combined.loc[category, "benchmark_return"], errors="coerce"
                )
            )

            # Brinson-Fachler formulation
            allocation = (p_w - b_w) * b_r
            selection = b_w * (p_r - b_r)
            interaction = (p_w - b_w) * (p_r - b_r)

            attribution_results[category] = {
                "Portfolio Weight": p_w,
                "Benchmark Weight": b_w,
                "Portfolio Return": p_r,
                "Benchmark Return": b_r,
                "Allocation Effect": allocation,
                "Selection Effect": selection,
                "Interaction Effect": interaction,
                "Combined Selection Effect": selection + interaction,
            }

        summary_df = pd.DataFrame.from_dict(
            attribution_results, orient="index"
        )
        summary_df["Total Effect"] = (
            summary_df["Allocation Effect"] + summary_df["Selection Effect"]
        )

        return summary_df

    def _calculate_selection_drilldown_from_db(
        self,
        group_by: str,
        portfolio_data: pd.DataFrame,
        benchmark_data: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """Calculate selection drilldown showing contribution of each symbol (vectorized)."""

        if self.portfolio_db is None:
            return {}

        drilldown = {}

        # Get unique categories
        categories = portfolio_data["category"].dropna().unique()

        # ✅ GET ALL DATA ONCE instead of per-date/per-category queries (using cache)
        portfolio_symbols = self._get_symbol_data()
        all_attributes = self._get_all_attributes(include_history=True)

        if portfolio_symbols.empty or all_attributes.empty:
            return {}

        # Check if attribute exists
        if group_by not in all_attributes.columns:
            logger.warning(
                "Attribute '%s' not found in symbol attributes", group_by
            )
            return {}

        # Ensure dates are timezone-aware for comparison
        if "effective_date" in all_attributes.columns:
            all_attributes["effective_date"] = pd.to_datetime(
                all_attributes["effective_date"], utc=True
            )
        if "end_date" in all_attributes.columns:
            all_attributes["end_date"] = pd.to_datetime(
                all_attributes["end_date"], utc=True
            )

        portfolio_symbols["date"] = pd.to_datetime(
            portfolio_symbols["date"], utc=True
        )

        # ✅ VECTORIZED: Build (symbol, date) -> attribute mapping once
        attr_by_symbol = {
            symbol: group for symbol, group in all_attributes.groupby("symbol")
        }

        # Build results for all symbols and dates at once
        results = []
        for symbol, symbol_group in portfolio_symbols.groupby("symbol"):
            if symbol not in attr_by_symbol:
                continue

            symbol_attrs = attr_by_symbol[symbol]

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
                    attr_value = valid.sort_values(
                        "effective_date", ascending=False
                    ).iloc[0][group_by]
                    if pd.notna(attr_value):
                        results.append(
                            {
                                "symbol": symbol,
                                "date": date,
                                "category": attr_value,
                                "weight": row.weight,
                                "twr_return": row.twr_return,
                            }
                        )

        if not results:
            return {}

        symbol_df = pd.DataFrame(results)

        # ✅ VECTORIZED: Process all categories at once using groupby
        for category in categories:
            category_data = symbol_df[symbol_df["category"] == category]

            if category_data.empty:
                continue

            # Calculate summary by symbol for this category
            symbol_summary = category_data.groupby("symbol").agg(
                {
                    "weight": "mean",
                    "twr_return": lambda x: (1 + x).prod() - 1,
                }
            )

            # Get benchmark return for this category
            bench_cat = benchmark_data[benchmark_data["category"] == category]
            if not bench_cat.empty:
                benchmark_return = (1 + bench_cat["twr_return"]).prod() - 1

                # Calculate selection effect for each symbol
                symbol_summary["Selection Effect"] = symbol_summary[
                    "weight"
                ] * (symbol_summary["twr_return"] - benchmark_return)

                drilldown[category] = symbol_summary.sort_values(
                    "Selection Effect", ascending=False
                )

        return drilldown
