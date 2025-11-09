"""
Calculates performance attribution by various groupings (symbol, account, factor).
"""

import numpy as np
import pandas as pd

from ..utils.logging_config import get_logger

logger = get_logger()


class AttributionCalculator:
    """
    Calculates performance attribution based on detailed portfolio history.
    """

    def __init__(
        self,
        portfolio_history: pd.DataFrame,
        transactions: pd.DataFrame,
        attrib_group_cols: list[str],
    ):
        self.portfolio_history = portfolio_history
        self.transactions = transactions
        self.attrib_group_cols = attrib_group_cols
        self.symbol_to_groups_map = self._create_symbol_map()

    def _create_symbol_map(self) -> dict:
        """Creates a map from symbol to its associated factor groups."""
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
        Calculates performance attribution for a given grouping.
        """
        group_values, group_cash_flows = self._get_daily_group_data(group_by)

        if group_values.empty:
            return pd.DataFrame()

        total_portfolio_value = self.portfolio_history.set_index("date")[
            "total_value"
        ]
        daily_weights = group_values.div(total_portfolio_value, axis=0).fillna(
            0
        )
        avg_weight = daily_weights.mean()

        return_cols = [f"{group}_twr_return" for group in group_values.columns]
        if not all(
            col in self.portfolio_history.columns for col in return_cols
        ):
            group_returns = self._calculate_daily_twr(
                group_values,
                group_cash_flows,
                self.portfolio_history,
                group_by,
            )
        else:
            group_returns = self.portfolio_history[return_cols].rename(
                columns=lambda c: c.replace("_twr_return", "")
            )

        twr = (1 + group_returns).prod() - 1
        contribution = avg_weight * twr

        summary = pd.DataFrame(
            {
                "Avg. Weight": avg_weight,
                "Return (TWR)": twr,
                "Contribution to Portfolio Return": contribution,
            }
        )

        if "benchmark_return" in self.portfolio_history.columns:
            benchmark_return_series = self.portfolio_history.set_index("date")[
                "benchmark_return"
            ]
            prod_result = (1 + benchmark_return_series).prod()
            if isinstance(prod_result, (int, float, np.number)):
                benchmark_twr = float(prod_result) - 1.0
            else:
                logger.error(
                    "AttributionCalculator: Unexpected product result type: %s",
                    type(prod_result),
                )
                benchmark_twr = 0.0

            summary["Excess Return vs. Benchmark"] = twr - benchmark_twr
            summary["Contribution to Excess Return"] = avg_weight * (
                twr - benchmark_twr
            )

        return summary.sort_values(by="Avg. Weight", ascending=False)

    def _get_daily_group_data(
        self, group_by: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Aggregates daily market values and cash flows."""
        history = self.portfolio_history.set_index("date")
        value_cols = [col for col in history.columns if col.endswith("_value")]
        cash_flow_cols = [
            col for col in history.columns if col.endswith("_cash_flow")
        ]

        group_values = pd.DataFrame(index=history.index)
        group_cash_flows = pd.DataFrame(index=history.index)

        if group_by == "symbol":
            groups = self.transactions["symbol"].unique()

            def __find_symbol_value_col(g, col):
                # Returning total value of symbol across all accounts
                return f"{g}_total_value" in col

            value_identifier = __find_symbol_value_col
        elif group_by == "account":
            groups = self.transactions["account"].unique()

            def __find_account_value_col(g, col):
                return col.startswith(f"{g}_total_value")

            value_identifier = __find_account_value_col
        else:  # Factor
            if group_by not in self.transactions.columns:
                return pd.DataFrame(), pd.DataFrame()
            groups = self.transactions[group_by].unique()

            def factor_val_identifier(group_item, col):
                symbols_in_group = [
                    t
                    for t, groups_map in self.symbol_to_groups_map.items()
                    if groups_map.get(group_by) == group_item
                ]
                # Returning total value of all symbols in this factor group
                # across all accounts
                return any(f"{t}_total_value" in col for t in symbols_in_group)

            value_identifier = factor_val_identifier

        for group in groups:
            cols_for_group_val = [
                c for c in value_cols if value_identifier(group, c)
            ]
            if cols_for_group_val:
                group_values[group] = history[cols_for_group_val].sum(axis=1)

            def __find_cash_flow_col(g, col):
                return col.endswith(f"{g}_cash_flow")

            cf_identifier = __find_cash_flow_col
            cols_for_group_cf = [
                c for c in cash_flow_cols if cf_identifier(group, c)
            ]
            if cols_for_group_cf:
                group_cash_flows[group] = history[cols_for_group_cf].sum(
                    axis=1
                )

        return group_values.fillna(0), group_cash_flows.fillna(0)

    def _calculate_daily_twr(
        self,
        values: pd.DataFrame,
        cash_flows: pd.DataFrame,
        history_df: pd.DataFrame,
        group_by: str,
    ) -> pd.DataFrame:
        """Robustly calculates daily Time-Weighted Returns for any group."""
        daily_returns = pd.DataFrame(
            index=values.index, columns=values.columns, data=0.0
        )
        history_df = history_df.set_index("date")

        for group_name in values.columns:
            group_start_values = values[group_name].shift(1).fillna(0)

            # Use explicit cash flows if they exist for the group (i.e., for accounts)
            if (
                group_name in cash_flows.columns
                and cash_flows[group_name].sum() != 0
            ):
                group_cash_flow = cash_flows[group_name]
            else:
                # Otherwise, derive the cash flow from market effects
                # This will apply to symbols and factors

                # Find all symbols within this group
                if group_by == "symbol":
                    symbols_in_group = [group_name]
                else:  # Factor
                    symbols_in_group = [
                        t
                        for t, groups_map in self.symbol_to_groups_map.items()
                        if groups_map.get(group_by) == group_name
                    ]

                if not symbols_in_group:
                    continue

                # Calculate the weighted average market return for all symbols in the group
                market_effect = pd.Series(0.0, index=history_df.index)
                for symbol in symbols_in_group:
                    return_col = f"{symbol}_market_return"
                    # Find all value columns for this symbol across all accounts
                    symbol_value_cols = [
                        c
                        for c in history_df.columns
                        if f"{symbol}_total_value" in c
                    ]
                    if return_col in history_df.columns and symbol_value_cols:
                        symbol_start_value = (
                            history_df[symbol_value_cols]
                            .sum(axis=1)
                            .shift(1)
                            .fillna(0)
                        )
                        market_effect += (
                            symbol_start_value * history_df[return_col]
                        )

                group_cash_flow = (
                    values[group_name] - group_start_values - market_effect
                )

            # Calculate TWR for the group for the day
            denominator = group_start_values + group_cash_flow
            numerator = (
                values[group_name] - group_start_values - group_cash_flow
            )

            daily_returns[group_name] = (
                (numerator / denominator)
                .replace([np.inf, -np.inf], 0)
                .fillna(0)
            )

        return daily_returns
