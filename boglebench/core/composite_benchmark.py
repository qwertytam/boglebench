"""
Builds a daily performance history for a composite benchmark.
"""

import pandas as pd

from ..utils.config import ConfigManager
from ..utils.logging_config import get_logger
from ..utils.tools import to_tzts_scaler

logger = get_logger()


class CompositeBenchmarkBuilder:
    """
    Calculates the daily history of a composite benchmark based on its
    components and rebalancing frequency.
    """

    def __init__(
        self,
        config: ConfigManager,
        market_data: dict[str, pd.DataFrame],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ):
        self.config = config
        self.market_data = market_data
        self.start_date = start_date
        self.end_date = end_date
        self.components = self.config.get_benchmark_components()
        self.rebalancing_freq = (
            self.config.get_benchmark_rebalancing_frequency()
        )
        self.symbols = [comp["symbol"] for comp in self.components]

    def build(self) -> pd.DataFrame:
        """
        Generates the daily history of the composite benchmark.

        Returns a DataFrame with 'date', 'adj_close', 'dividend', and
        'split_coefficient' columns,
        mimicking the structure of a single symbol's market data.
        """
        # Ensure all component data is available
        if not all(symbol in self.market_data for symbol in self.symbols):
            logger.warning(
                "Missing market data for one or more benchmark components."
            )
            return pd.DataFrame()

        # Prepare a combined DataFrame with adjusted close prices for all components
        adj_close_df = self._prepare_price_data()
        if adj_close_df.empty:
            return pd.DataFrame()

        # Simulate the benchmark portfolio
        initial_investment = 10000.0  # Starting with $10,000
        shares = self._calculate_initial_shares(
            adj_close_df, initial_investment
        )

        daily_values = []
        for date, row in adj_close_df.iterrows():
            market_values = row * shares
            market_value = market_values.sum()
            weights = (
                market_values / market_value
                if market_value > 0
                else [0] * len(self.symbols)
            )
            daily_values.append({"date": date, "value": market_value})

            for symbol, weight in zip(self.symbols, weights):
                daily_values[-1][f"{symbol}_weight"] = weight
                daily_values[-1][f"{symbol}_total_value"] = market_values[
                    symbol
                ]
            daily_values[-1]["total_value"] = market_value

            # Check for rebalancing
            if not isinstance(date, pd.Timestamp):
                try:
                    date = to_tzts_scaler(date)  # type: ignore
                    if date is None:
                        msg = f"Date '{date}' could not be converted to Timestamp."
                        logger.warning(msg)
                        raise ValueError(msg)

                except ValueError:
                    logger.warning(
                        "Could not convert date '%s' to Timestamp.", date
                    )
                    continue
            if self._is_rebalancing_day(date):
                shares = self._rebalance(row, market_value)

        # Convert daily values to a DataFrame that mimics market data
        benchmark_df = pd.DataFrame(daily_values)
        benchmark_df = benchmark_df.rename(columns={"value": "adj_close"})
        benchmark_df["dividend"] = (
            0.0  # Simplification: dividends are handled by adj_close
        )
        benchmark_df["split_coefficient"] = 0.0  # No splits in composite

        benchmark_df = benchmark_df.sort_values("date").reset_index(drop=True)

        benchmark_df["benchmark_return"] = (
            benchmark_df["adj_close"].pct_change().fillna(0.0)
        )

        for symbol in self.symbols:
            benchmark_df[f"{symbol}_twr_return"] = (
                benchmark_df[f"{symbol}_total_value"].pct_change().fillna(0.0)
            )

        return benchmark_df

    def _prepare_price_data(self) -> pd.DataFrame:
        """Creates a merged DataFrame of adjusted close prices for all components."""
        all_dfs = []
        for symbol in self.symbols:
            df = self.market_data[symbol][["date", "adj_close"]].set_index(
                "date"
            )
            df = df.rename(columns={"adj_close": symbol})
            all_dfs.append(df)

        # Merge all dataframes and forward-fill missing values
        merged_df = pd.concat(all_dfs, axis=1)
        merged_df = merged_df.ffill().bfill()

        # Filter to the analysis date range
        return merged_df.loc[self.start_date : self.end_date]

    def _calculate_initial_shares(
        self, adj_close_df: pd.DataFrame, investment: float
    ) -> pd.Series:
        """Calculates the initial number of shares for each component."""
        first_day_prices = adj_close_df.iloc[0]
        shares = pd.Series(0.0, index=self.symbols)
        for comp in self.components:
            symbol, weight = comp["symbol"], comp["weight"]
            allocated_amount = investment * weight
            shares[symbol] = allocated_amount / first_day_prices[symbol]
        return shares

    def _is_rebalancing_day(self, date: pd.Timestamp) -> bool:
        """Checks if the given date is a rebalancing day."""
        if self.rebalancing_freq == "none":
            return False

        # Using pandas offsets for robust date checks
        if self.rebalancing_freq == "daily":
            return True
        if self.rebalancing_freq == "weekly":
            return date.dayofweek == 0  # Monday
        if self.rebalancing_freq == "monthly":
            rebal_date = date + pd.offsets.MonthBegin(0)  # First day of month
            is_rebal_day = date == rebal_date
            return is_rebal_day
        if self.rebalancing_freq == "quarterly":
            return date == date + pd.offsets.QuarterBegin(-1, startingMonth=1)
        if self.rebalancing_freq == "yearly":
            return date == date + pd.offsets.YearBegin(-1)

        return False

    def _rebalance(self, prices: pd.Series, total_value: float) -> pd.Series:
        """Calculates new share counts based on target weights."""
        new_shares = pd.Series(0.0, index=self.symbols)
        for comp in self.components:
            symbol, weight = comp["symbol"], comp["weight"]
            allocated_amount = total_value * weight
            new_shares[symbol] = allocated_amount / prices[symbol]
        return new_shares
