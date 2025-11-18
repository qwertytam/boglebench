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

    def _get_portfolio_data_with_attributes(
        self, attribute: str
    ) -> pd.DataFrame:
        """Get portfolio data grouped by attribute."""

        if self.portfolio_db is None:
            return pd.DataFrame()

        # Get all symbol data
        symbol_data = self.portfolio_db.get_symbol_data()

        if symbol_data.empty:
            return pd.DataFrame()

        # Get attributes for all dates
        result_data = []

        for date in symbol_data["date"].unique():
            date_data = symbol_data[symbol_data["date"] == date]

            # Get attributes as of this date
            attributes_at_date = (
                self.portfolio_db.get_symbol_attributes_at_date(date)
            )

            if attributes_at_date.empty:
                continue

            # Merge with attributes
            merged = date_data.merge(
                attributes_at_date[["symbol", attribute]],
                on="symbol",
                how="left",
            )

            # Group by attribute category
            # Calculate weighted return manually to avoid lambda capturing
            # loop variable
            merged["weighted_return"] = merged["twr_return"] * merged["weight"]

            grouped = (
                merged.groupby([attribute])
                .agg(
                    {
                        "weight": "sum",
                        "weighted_return": "sum",
                    }
                )
                .reset_index()
            )

            # Calculate the final weighted average return
            grouped["twr_return"] = grouped.apply(
                lambda row: (
                    row["weighted_return"] / row["weight"]
                    if row["weight"] > 0
                    else 0
                ),
                axis=1,
            )
            grouped = grouped.drop(columns=["weighted_return"])

            grouped["date"] = date
            result_data.append(grouped)

        if not result_data:
            return pd.DataFrame()

        result_df = pd.concat(result_data, ignore_index=True)
        result_df = result_df.rename(columns={attribute: "category"})

        return result_df

    def _get_benchmark_data_with_attributes(
        self, attribute: str
    ) -> pd.DataFrame:
        """Get benchmark data grouped by attribute."""

        if self.portfolio_db is None:
            return pd.DataFrame()

        if self.benchmark_history is None or self.benchmark_history.empty:
            logger.warning("No benchmark history available")
            return pd.DataFrame()

        # Extract benchmark component symbols from benchmark_history columns
        benchmark_symbols = []
        for col in self.benchmark_history.columns:
            if col.endswith("_twr_return") and not col.startswith("benchmark"):
                symbol = col.replace("_twr_return", "")
                benchmark_symbols.append(symbol)

        if not benchmark_symbols:
            logger.warning("No benchmark symbols found in benchmark_history")
            return pd.DataFrame()

        # Get attributes for benchmark symbols
        result_data = []

        # Get dates from index or column
        if "date" in self.benchmark_history.columns:
            dates = pd.to_datetime(self.benchmark_history["date"], utc=True)
        else:
            # Date is in the index
            index_dates = self.benchmark_history.index
            if not isinstance(index_dates, pd.DatetimeIndex):
                dates = pd.Series(pd.to_datetime(index_dates, utc=True))
            else:
                dates = pd.Series(index_dates)

        for date in dates:
            # Get attributes as of this date
            attributes_at_date = (
                self.portfolio_db.get_symbol_attributes_at_date(
                    date,
                )
            )

            if attributes_at_date.empty:
                continue

            # Filter to benchmark symbols
            bench_attrs = attributes_at_date[
                attributes_at_date["symbol"].isin(benchmark_symbols)
            ]

            if bench_attrs.empty:
                continue

            # Get weights and returns from benchmark_history
            if "date" in self.benchmark_history.columns:
                date_row = self.benchmark_history[
                    self.benchmark_history["date"].dt.date == date.date()
                ]
            else:
                # Date is in the index
                try:
                    date_row = self.benchmark_history.loc[[date]]
                except KeyError:
                    # Try with date() if it's a timestamp
                    try:
                        matching_idx = [
                            idx
                            for idx in self.benchmark_history.index
                            if idx.date() == date.date()
                        ]
                        if matching_idx:
                            date_row = self.benchmark_history.loc[matching_idx]
                        else:
                            date_row = pd.DataFrame()
                    except Exception:  # pylint: disable=broad-except
                        date_row = pd.DataFrame()
                        logger.warning("No benchmark data for date %s", date)

            if date_row.empty:
                continue

            date_series = date_row.iloc[0]

            # Aggregate by attribute
            category_data = []
            for category in bench_attrs[attribute].dropna().unique():
                cat_symbols = bench_attrs[bench_attrs[attribute] == category][
                    "symbol"
                ].tolist()

                # Sum weights and calculate weighted return
                total_weight = 0
                weighted_return = 0

                for symbol in cat_symbols:
                    weight_col = f"{symbol}_weight"
                    return_col = f"{symbol}_twr_return"

                    if (
                        weight_col in date_series.index
                        and return_col in date_series.index
                    ):
                        weight = date_series[weight_col]
                        ret = date_series[return_col]

                        total_weight += weight
                        weighted_return += weight * ret

                if total_weight > 0:
                    category_data.append(
                        {
                            "date": date,
                            "category": category,
                            "weight": total_weight,
                            "twr_return": weighted_return / total_weight,
                        }
                    )

            if category_data:
                result_data.extend(category_data)

        if not result_data:
            return pd.DataFrame()

        return pd.DataFrame(result_data)

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
