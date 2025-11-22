"""
Brinson-Fachler attribution analysis implementation.

This module implements the Brinson-Fachler attribution methodology to decompose
portfolio returns into allocation and selection effects. Helps identify whether
outperformance came from sector allocation decisions or individual security
selection using database as the single source of truth.
"""

from typing import Dict, Tuple

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
        """Get portfolio data grouped by attribute (optimized)."""

        if self.portfolio_db is None:
            return pd.DataFrame()

        # Get all symbol data
        symbol_data = self.portfolio_db.get_symbol_data()

        if symbol_data.empty:
            return pd.DataFrame()

        # ✅ GET ALL ATTRIBUTES ONCE instead of per-date
        all_attributes = self.portfolio_db.get_symbol_attributes()

        if all_attributes.empty:
            return pd.DataFrame()

        # Filter to only the attribute we need
        if attribute not in all_attributes.columns:
            logger.warning("Attribute '%s' not found in symbol attributes", attribute)
            return pd.DataFrame()

        # Merge symbol data with attributes once
        merged = symbol_data.merge(
            all_attributes[["symbol", attribute]],
            on="symbol",
            how="left",
        )

        # Calculate weighted return
        merged["weighted_return"] = merged["twr_return"] * merged["weight"]

        # Group by date and attribute category
        result_df = (
            merged.groupby(["date", attribute])
            .agg({
                "weight": "sum",
                "weighted_return": "sum",
            })
            .reset_index()
        )

        # Calculate weighted average return for each group
        result_df["twr_return"] = result_df.apply(
            lambda row: (
                row["weighted_return"] / row["weight"]
                if row["weight"] > 0
                else 0
            ),
            axis=1,
        )

        # Clean up
        result_df = result_df.drop(columns=["weighted_return"])
        result_df = result_df.rename(columns={attribute: "category"})

        return result_df

    @timeit
    def _get_benchmark_data_with_attributes(
        self, attribute: str
    ) -> pd.DataFrame:
        """Get benchmark data grouped by attribute (optimized)."""

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

        # ✅ Get attributes for all benchmark symbols at once
        bench_attrs = self.portfolio_db.get_symbol_attributes()
        bench_attrs = bench_attrs[bench_attrs["symbol"].isin(benchmark_symbols)]

        if bench_attrs.empty:
            logger.warning("No attributes found for benchmark symbols")
            return pd.DataFrame()

        # Check if attribute exists
        if attribute not in bench_attrs.columns:
            logger.warning("Attribute '%s' not found for benchmark symbols", attribute)
            return pd.DataFrame()

        # Create symbol to category mapping
        symbol_to_category = bench_attrs.set_index("symbol")[attribute].to_dict()

        # ✅ VECTORIZED: Process all dates at once
        dates = (
            self.benchmark_history.index
            if "date" not in self.benchmark_history.columns
            else self.benchmark_history["date"]
        )

        result_list = []

        # Iterate through symbols (few) instead of dates (many)
        for symbol in benchmark_symbols:
            weight_col = f"{symbol}_weight"
            return_col = f"{symbol}_twr_return"

            # Skip if columns don't exist
            if weight_col not in self.benchmark_history.columns:
                continue
            if return_col not in self.benchmark_history.columns:
                continue

            # Get category for this symbol
            category = symbol_to_category.get(symbol)
            if pd.isna(category):
                continue

            # ✅ Extract all data for this symbol at once
            for idx, row in self.benchmark_history.iterrows():
                result_list.append({
                    "date": idx if "date" not in self.benchmark_history.columns else row["date"],
                    "symbol": symbol,
                    "category": category,
                    "weight": row[weight_col],
                    "twr_return": row[return_col],
                })

        if not result_list:
            return pd.DataFrame()

        df = pd.DataFrame(result_list)

        # ✅ Aggregate by date and category using vectorized operations
        df["weighted_return"] = df["twr_return"] * df["weight"]

        grouped = (
            df.groupby(["date", "category"])
            .agg({
                "weight": "sum",
                "weighted_return": "sum",
            })
            .reset_index()
        )

        # Calculate weighted average return
        grouped["twr_return"] = grouped.apply(
            lambda row: (
                row["weighted_return"] / row["weight"]
                if row["weight"] > 0
                else 0
            ),
            axis=1,
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
        """Calculate selection drilldown showing contribution of each symbol."""

        if self.portfolio_db is None:
            return {}

        drilldown = {}

        # Get unique categories
        categories = portfolio_data["category"].dropna().unique()

        for category in categories:
            # Get portfolio symbols in this category
            portfolio_symbols = self.portfolio_db.get_symbol_data()

            # Filter by dates and get attributes
            symbol_data = []

            for date in portfolio_data["date"].unique():
                attributes = self.portfolio_db.get_symbol_attributes_at_date(
                    date
                )

                if attributes.empty:
                    continue

                category_symbols = attributes[
                    attributes[group_by] == category
                ]["symbol"].tolist()

                for symbol in category_symbols:
                    sym_data = portfolio_symbols[
                        (portfolio_symbols["symbol"] == symbol)
                        & (portfolio_symbols["date"] == date)
                    ]

                    if not sym_data.empty:
                        symbol_data.append(
                            {
                                "symbol": symbol,
                                "date": date,
                                "weight": sym_data.iloc[0]["weight"],
                                "twr_return": sym_data.iloc[0]["twr_return"],
                            }
                        )

            if not symbol_data:
                continue

            symbol_df = pd.DataFrame(symbol_data)

            # Calculate summary by symbol
            symbol_summary = symbol_df.groupby("symbol").agg(
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
