"""
Performs Brinson-Fachler performance attribution analysis.

This module breaks down the active return (portfolio return vs. benchmark return)
into Allocation and Selection effects.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..utils.config import ConfigManager
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class BrinsonAttributionCalculator:
    """
    Calculates Brinson-Fachler attribution effects (Allocation and Selection).
    """

    def __init__(
        self,
        config: ConfigManager,
        portfolio_history: pd.DataFrame,
        benchmark_history: pd.DataFrame,
        transactions: pd.DataFrame,
    ):
        self.config = config
        self.portfolio_history = portfolio_history.set_index("date")
        self.benchmark_history = benchmark_history.set_index("date")
        self.transactions = transactions
        self.benchmark_components = self.config.get("benchmark.components", [])

    def calculate(self, group_by: str) -> tuple[pd.DataFrame, Dict]:
        """
        The main method to perform the full attribution analysis.

        Args:
            group_by (str): The category to group by (e.g., 'group_asset_class').

        Returns:
            A tuple containing:
            - A DataFrame with the high-level attribution effects.
            - A dictionary with the detailed selection drill-down for each
            category.
        """
        if not self.benchmark_components or self.benchmark_history.empty:
            return pd.DataFrame(), {}

        if group_by not in self.transactions.columns:
            raise ValueError(
                f"Grouping column '{group_by}' not found in transactions."
            )

        # Get portfolio and benchmark returns and weights, grouped by the
        # category
        port_returns, port_weights = self._get_grouped_data(
            self.portfolio_history, group_by, source="portfolio"
        )
        bench_returns, bench_weights = self._get_grouped_data(
            self.benchmark_history, group_by, source="benchmark"
        )

        # Get total returns for the portfolio and benchmark
        prod_result = (
            1 + self.portfolio_history["portfolio_daily_return_twr"]
        ).prod()
        if isinstance(prod_result, (int, float, np.number)):
            total_port_return = float(prod_result) - 1.0
        else:
            total_port_return = 0.0

        logger.debug(
            "Total Portfolio Return (TWR): %.4f%%", total_port_return * 100
        )

        prod_result = (1 + self.benchmark_history["benchmark_return"]).prod()
        if isinstance(prod_result, (int, float, np.number)):
            total_bench_return = float(prod_result) - 1.0
        else:
            total_bench_return = 0.0

        # Calculate high-level attribution effects
        attribution_results = {}
        all_categories = sorted(
            list(set(port_weights.columns) | set(bench_weights.columns))
        )

        for category in all_categories:
            p_w = (
                port_weights[category].mean()
                if category in port_weights
                else 0
            )
            b_w = (
                bench_weights[category].mean()
                if category in bench_weights
                else 0
            )
            if category in port_weights:
                prod_result = (1 + port_returns[category]).prod()
                if isinstance(prod_result, (int, float, np.number)):
                    p_r = float(prod_result) - 1.0
                else:
                    p_r = 0.0
            else:
                p_r = 0

            if category in bench_returns:
                prod_result = (1 + bench_returns[category]).prod()
                if isinstance(prod_result, (int, float, np.number)):
                    b_r = float(prod_result) - 1.0
                else:
                    b_r = 0.0
            else:
                b_r = 0

            # Brinson-Fachler Formulas
            allocation = (p_w - b_w) * (b_r - total_bench_return)
            selection = p_w * (p_r - b_r)
            interaction = (p_w - b_w) * (
                p_r - b_r
            )  # Optional, often combined with selection

            attribution_results[category] = {
                "Portfolio Weight": p_w,
                "Benchmark Weight": b_w,
                "Portfolio Return": p_r,
                "Benchmark Return": b_r,
                "Allocation Effect": allocation,
                "Selection Effect": selection,
                "Interaction Effect": interaction,
                "Combined Selection Effect": selection
                + interaction,  # Combine for clarity
            }

        summary_df = pd.DataFrame.from_dict(
            attribution_results, orient="index"
        )
        summary_df["Total Effect"] = (
            summary_df["Allocation Effect"] + summary_df["Selection Effect"]
        )

        # 5. Calculate the selection drill-down for each category
        selection_drilldown = self._calculate_selection_drilldown(
            group_by, bench_returns
        )

        return summary_df, selection_drilldown

    def _get_grouped_data(
        self, history_df: pd.DataFrame, group_by: str, source: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aggregates daily returns and weights for a given history DataFrame by
        the specified category.

        Args:
            history_df: The daily history DataFrame (portfolio or benchmark).
            group_by: The category to group by (e.g., 'group_asset_class').
            source: A string ('portfolio' or 'benchmark') to determine which
                    transaction list to use for grouping.
        """
        if source == "portfolio":
            trans_df = self.transactions
        elif source == "benchmark":
            trans_df = pd.DataFrame(self.benchmark_components)
        else:
            raise ValueError(f"Invalid source '{source}' specified.")

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
        """Calculates the contribution of each individual symbol to the
        Selection Effect."""
        drilldown = {}
        port_symbols_by_cat = self.transactions.groupby(group_by)[
            "symbol"
        ].unique()

        for category, symbols in port_symbols_by_cat.items():
            cat_results = {}
            if category in bench_returns:
                prod_result = (1 + bench_returns[category]).prod()
                if isinstance(prod_result, (int, float, np.number)):
                    bench_cat_return = float(prod_result) - 1.0
                else:
                    bench_cat_return = 0.0
            else:
                bench_cat_return = 0

            for symbol in symbols:
                symbol_return_col = f"{symbol}_twr_return"
                if symbol_return_col in self.portfolio_history.columns:
                    # Calculate TWR for this specific symbol
                    symbol_values = self.portfolio_history[
                        f"{symbol}_total_value"
                    ]
                    start_values = symbol_values.shift(1).fillna(0)

                    # Ensure daily returns are only calculated when the position is held
                    symbol_returns_daily = np.where(
                        start_values > 0,
                        self.portfolio_history[symbol_return_col],
                        0,
                    )
                    prod_result = (
                        1
                        + pd.Series(
                            symbol_returns_daily,
                            index=self.portfolio_history.index,
                        )
                    ).prod()
                    if isinstance(prod_result, (int, float, np.number)):
                        symbol_twr = float(prod_result) - 1.0
                    else:
                        symbol_twr = 0.0

                    # Calculate contribution to selection
                    avg_weight_in_cat = self.portfolio_history[
                        f"{symbol}_weight"
                    ].mean()
                    contribution = avg_weight_in_cat * (
                        symbol_twr - bench_cat_return
                    )

                    cat_results[symbol] = {
                        "Avg. Weight": avg_weight_in_cat,
                        "Return (TWR)": symbol_twr,
                        "Contribution to Selection": contribution,
                    }

            if cat_results:
                drilldown[str(category)] = pd.DataFrame.from_dict(
                    cat_results, orient="index"
                ).sort_values(by="Contribution to Selection", ascending=False)

        return drilldown
