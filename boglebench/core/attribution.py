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
        portfolio_history: Optional[pd.DataFrame] = None,
        transactions: Optional[pd.DataFrame] = None,
        attrib_group_cols: Optional[List[str]] = None,
        portfolio_db: Optional[PortfolioDatabase] = None,  # NEW
    ):
        self.portfolio_history = portfolio_history
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
        if self.portfolio_history is not None:
            return self._calculate_from_dataframe(group_by)

        logger.error("No data source available for attribution calculation")
        return pd.DataFrame()

    def _calculate_from_database(self, group_by: str) -> pd.DataFrame:
        """Calculate attribution using database queries."""

        if group_by == "symbol":
            return self._calculate_symbol_attribution_from_db()

        elif group_by == "account":
            return self._calculate_account_attribution_from_db()

        elif group_by in [
            "asset_class",
            "geography",
            "region",
            "sector",
            "style",
            "market_cap",
            "fund_type",
        ]:
            return self._calculate_attribute_attribution_from_db(group_by)

        else:
            # For custom groupings from transactions, use DataFrame approach
            logger.info(
                "Custom grouping '%s' using DataFrame calculation", group_by
            )
            return self._calculate_from_dataframe(group_by)

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

    def _calculate_from_dataframe(self, group_by: str) -> pd.DataFrame:
        """Legacy: calculate attribution using DataFrame."""
        if self.portfolio_history is None:
            return pd.DataFrame()

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
        """Aggregates daily market values and cash flows (legacy DataFrame method)."""
        if self.portfolio_history is None:
            return pd.DataFrame(), pd.DataFrame()

        history = self.portfolio_history.set_index("date")
        value_cols = [col for col in history.columns if col.endswith("_value")]

        group_values = pd.DataFrame(index=history.index)
        group_cash_flows = pd.DataFrame(index=history.index)

        if group_by == "symbol":
            # Extract symbols from column names
            symbol_set = set()
            for col in value_cols:
                parts = col.split("_")
                if len(parts) >= 3 and col.endswith("_total_value"):
                    symbol = col.replace("_total_value", "")
                    symbol_set.add(symbol)

            for symbol in symbol_set:
                value_col = f"{symbol}_total_value"
                cf_col = f"{symbol}_cash_flow"
                if value_col in history.columns:
                    group_values[symbol] = history[value_col]
                if cf_col in history.columns:
                    group_cash_flows[symbol] = history[cf_col]

        elif group_by == "account":
            # Extract accounts
            accounts = set()
            for col in value_cols:
                if col.endswith("_total_value") and not col.startswith(
                    "total"
                ):
                    account = col.replace("_total_value", "")
                    # Make sure it's an account, not a symbol
                    if "_" not in account:  # Simple heuristic
                        accounts.add(account)

            for account in accounts:
                value_col = f"{account}_total_value"
                cf_col = f"{account}_cash_flow"
                if value_col in history.columns:
                    group_values[account] = history[value_col]
                if cf_col in history.columns:
                    group_cash_flows[account] = history[cf_col]

        else:
            # Custom grouping from symbol_to_groups_map
            if (
                not self.symbol_to_groups_map
                or group_by not in list(self.symbol_to_groups_map.values())[0]
            ):
                logger.warning(
                    "Group '%s' not found in symbol attributes", group_by
                )
                return pd.DataFrame(), pd.DataFrame()

            # Group symbols by the attribute
            groups: dict[str, List[str]] = {}
            for symbol, attrs in self.symbol_to_groups_map.items():
                group_val = attrs.get(group_by)
                if group_val:
                    if group_val not in groups:
                        groups[group_val] = []
                    groups[group_val].append(symbol)

            # Aggregate by group
            for group_val, symbols in groups.items():
                group_value = pd.Series(0.0, index=history.index)
                group_cf = pd.Series(0.0, index=history.index)

                for symbol in symbols:
                    value_col = f"{symbol}_total_value"
                    cf_col = f"{symbol}_cash_flow"
                    if value_col in history.columns:
                        group_value += history[value_col]
                    if cf_col in history.columns:
                        group_cf += history[cf_col]

                group_values[group_val] = group_value
                group_cash_flows[group_val] = group_cf

        return group_values, group_cash_flows

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
