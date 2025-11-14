"""
Brinson-Fachler attribution analysis implementation.

This module implements the Brinson-Fachler attribution methodology to decompose
portfolio returns into allocation and selection effects. Helps identify whether
outperformance came from sector allocation decisions or individual security
selection. Supports both database and DataFrame data sources.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.portfolio_db import PortfolioDatabase
from ..utils.logging_config import get_logger

logger = get_logger()


class BrinsonAttributionCalculator:
    """
    Implements Brinson-Fachler attribution analysis.

    Supports both normalized database and legacy DataFrame sources.
    """

    def __init__(
        self,
        portfolio_history: Optional[pd.DataFrame] = None,
        benchmark_history: Optional[pd.DataFrame] = None,
        transactions: Optional[pd.DataFrame] = None,
        benchmark_components: Optional[List[Dict]] = None,
        portfolio_db: Optional[PortfolioDatabase] = None,  # NEW
    ):
        """
        Initialize the BrinsonAttributionCalculator.

        Args:
            portfolio_history: Legacy DataFrame with portfolio history (optional)
            benchmark_history: DataFrame with benchmark performance data
            transactions: DataFrame containing transaction data (optional)
            benchmark_components: List of benchmark components (optional)
            portfolio_db: PortfolioDatabase for normalized data access (preferred)
        """
        self.portfolio_history = portfolio_history
        self.benchmark_history = benchmark_history
        self.transactions = transactions
        self.benchmark_components = benchmark_components or []
        self.portfolio_db = portfolio_db  # NEW

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
        # Try database first if available
        if self.portfolio_db is not None:
            try:
                return self._calculate_from_database(group_by)
            except Exception as e:  # pylint: disable=broad-except
                logger.warning(
                    "Database calculation failed, falling back to DataFrame: %s",
                    e,
                )

        # Fallback to DataFrame
        if (
            self.portfolio_history is not None
            and self.benchmark_history is not None
        ):
            return self._calculate_from_dataframe(group_by)

        logger.error("No data source available for Brinson attribution")
        return pd.DataFrame(), {}

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

        for date in pd.to_datetime(self.benchmark_history["date"], utc=True):
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
            date_row = self.benchmark_history[
                self.benchmark_history["date"].dt.date == date.date()
            ]

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

    def _calculate_from_dataframe(
        self, group_by: str
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Legacy: calculate Brinson attribution using DataFrame."""
        if self.portfolio_history is None or self.benchmark_history is None:
            logger.error("Portfolio or benchmark history is not available")
            return pd.DataFrame(), {}

        # Get grouped data for portfolio
        port_returns, port_weights = self._get_grouped_data(
            self.portfolio_history, group_by, "portfolio"
        )

        # Get grouped data for benchmark
        bench_returns, bench_weights = self._get_grouped_data(
            self.benchmark_history, group_by, "benchmark"
        )

        # Calculate average weights and compound returns
        port_avg_weights = port_weights.mean()
        bench_avg_weights = bench_weights.mean()

        port_twr = (1 + port_returns).prod() - 1
        bench_twr = (1 + bench_returns).prod() - 1

        # Calculate Brinson components
        attribution_results = {}

        all_categories = set(port_avg_weights.index).union(
            set(bench_avg_weights.index)
        )

        for category in all_categories:
            p_w = port_avg_weights.get(category, 0)
            b_w = bench_avg_weights.get(category, 0)
            p_r = port_twr.get(category, 0)
            b_r = bench_twr.get(category, 0)

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

        # Calculate selection drill-down
        bench_returns_df = pd.DataFrame(bench_returns)
        selection_drilldown = self._calculate_selection_drilldown(
            group_by, bench_returns_df
        )

        return summary_df, selection_drilldown

    def _get_grouped_data(
        self, history_df: pd.DataFrame, group_by: str, source: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aggregates daily returns and weights by category (legacy DataFrame method).
        """
        if source == "portfolio":
            trans_df = self.transactions
        elif source == "benchmark":
            trans_df = pd.DataFrame(self.benchmark_components)
        else:
            raise ValueError(f"Invalid source '{source}' specified.")

        if trans_df is None or trans_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        grouped_returns = pd.DataFrame(index=history_df.index)
        grouped_weights = pd.DataFrame(index=history_df.index)

        categories = trans_df[group_by].unique()

        for category in categories:
            symbols_in_cat = trans_df[trans_df[group_by] == category][
                "symbol"
            ].unique()

            # Aggregate returns (weighted average of symbol returns)
            cat_return = pd.Series(0.0, index=history_df.index)
            cat_total_value = pd.Series(0.0, index=history_df.index)

            for symbol in symbols_in_cat:
                return_col = f"{symbol}_twr_return"
                value_col = f"{symbol}_total_value"
                if (
                    return_col in history_df.columns
                    and value_col in history_df.columns
                ):
                    symbol_start_value = (
                        history_df[value_col].shift(1).fillna(0)
                    )
                    cat_return += symbol_start_value * history_df[return_col]
                    cat_total_value += symbol_start_value

            grouped_returns[category] = (
                (cat_return / cat_total_value)
                .replace([np.inf, -np.inf], 0)
                .fillna(0)
            )

            # Aggregate weights
            weight_cols = [
                f"{symbol}_weight"
                for symbol in symbols_in_cat
                if f"{symbol}_weight" in history_df.columns
            ]
            if weight_cols:
                grouped_weights[category] = history_df[weight_cols].sum(axis=1)

        return grouped_returns, grouped_weights

    def _calculate_selection_drilldown(
        self,
        group_by: str,
        bench_returns: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculates the contribution of each individual symbol to the Selection Effect
        (legacy DataFrame method).
        """
        if self.portfolio_history is None or self.transactions is None:
            return {}

        drilldown = {}
        port_symbols_by_cat = self.transactions.groupby(group_by)[
            "symbol"
        ].unique()

        for category, symbols in port_symbols_by_cat.items():
            category_str = str(category)
            symbol_data = []

            for symbol in symbols:
                return_col = f"{symbol}_twr_return"
                value_col = f"{symbol}_total_value"

                if (
                    return_col in self.portfolio_history.columns
                    and value_col in self.portfolio_history.columns
                ):
                    # Symbol return
                    symbol_returns = self.portfolio_history[return_col]
                    symbol_twr = (1 + symbol_returns).prod()
                    if isinstance(symbol_twr, (int, float, np.number)):
                        symbol_twr -= 1
                    else:
                        logger.error(
                            "Invalid TWR calculation for symbol %s", symbol
                        )
                        symbol_twr = 0

                    # Symbol weight
                    total_value = self.portfolio_history["total_value"]
                    symbol_value = self.portfolio_history[value_col]
                    symbol_weights = symbol_value / total_value
                    avg_weight = symbol_weights.mean()
                    # Benchmark return for this category
                    if category_str in bench_returns.columns:
                        bench_cat_ret = (
                            1 + bench_returns[category_str]
                        ).prod()
                        bench_cat_ret = (1 + bench_returns[category]).prod()
                        if isinstance(bench_cat_ret, (int, float, np.number)):
                            bench_cat_ret -= 1
                        else:
                            logger.error(
                                "Invalid TWR calculation for symbol %s", symbol
                            )
                            bench_cat_ret = 0
                    else:
                        bench_cat_ret = 0

                    # Selection effect
                    selection_effect = avg_weight * (
                        symbol_twr - bench_cat_ret
                    )

                    symbol_data.append(
                        {
                            "Symbol": symbol,
                            "Avg. Weight": avg_weight,
                            "Return (TWR)": symbol_twr,
                            "Selection Effect": selection_effect,
                        }
                    )
            if symbol_data:
                drilldown[category_str] = pd.DataFrame(symbol_data).set_index(
                    "Symbol"
                )

        return drilldown
