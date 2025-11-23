"""
Portfolio history construction and database persistence.

This module builds the daily portfolio history from transactions and market data,
calculating holdings, values, cash flows, and returns for each day. Writes the
normalized portfolio history to a SQLite database for efficient querying and
analysis.
"""

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal  # type: ignore

from ..core.constants import DateAndTimeConstants, TransactionTypes
from ..core.metrics import (
    calculate_account_modified_dietz_returns,
    calculate_account_twr_daily_returns,
    calculate_market_change_and_returns,
    calculate_modified_dietz_returns,
    calculate_twr_daily_returns,
)
from ..core.portfolio_db import PortfolioDatabase
from ..utils.config import ConfigManager
from ..utils.logging_config import get_logger
from ..utils.tools import to_tzts_scaler


class PortfolioHistoryBuilder:
    """Builds the daily portfolio history and stores in normalized database."""

    def __init__(
        self,
        config: ConfigManager,
        transactions: pd.DataFrame,
        market_data: dict,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        db_path: Optional[str] = None,
    ):
        """
        Initialize the PortfolioHistoryBuilder.

        Args:
            config: ConfigManager with portfolio settings
            transactions: DataFrame containing transaction data
            market_data: Dictionary mapping symbols to their market data
            start_date: Start date for portfolio history
            end_date: End date for portfolio history
            db_path: Optional custom database path (defaults to config path)
        """
        self.config = config
        self.transactions = transactions
        self.market_data = market_data
        self.start_date = start_date
        self.end_date = end_date
        self.logger = get_logger()
        self.symbols = self.transactions["symbol"].unique().tolist()
        self.accounts = self.transactions["account"].unique().tolist()

        # Initialize database
        if db_path is None:
            db_path = str(self.config.get_database_path())
            db_path = db_path if db_path is not None else None

        if db_path is None:
            self.logger.warning(
                "No database path provided, using in-memory database"
            )
            db_path = ":memory:"

        self.db = PortfolioDatabase(db_path=db_path, config=self.config)
        # Clear existing data for this date range
        self.db.clear_date_range(start_date, end_date)

        # Pre-build price lookup structures for fast O(log n) access
        self.price_lookup = self._build_price_lookup()

    def build(self) -> PortfolioDatabase:
        """
        Build and return the portfolio database.

        Returns:
            PortfolioDatabase: Database with normalized portfolio history
        """
        self.logger.info("🏗️ Building portfolio history...")

        date_range = self._get_processing_date_range()
        if date_range.empty:
            self.logger.warning(
                "No dates to process for building portfolio history."
            )
            return self.db

        holdings = {
            acc: {tck: 0.0 for tck in self.symbols} for acc in self.accounts
        }

        # Build to database
        self._build_to_database(date_range, holdings)

        self.logger.info(
            "✅ Portfolio history built: %d days", len(date_range)
        )
        return self.db

    def _build_to_database(self, date_range: pd.DatetimeIndex, holdings: dict):
        """Build portfolio history and write to database using bulk inserts."""

        # First pass: collect all daily data
        daily_data_list = []
        for date in date_range:
            day_data, holdings = self._process_one_day(date, holdings)
            daily_data_list.append(day_data)

        # Create temporary DataFrame for returns calculation
        temp_df = pd.DataFrame(daily_data_list)
        temp_df = self._add_returns_and_metrics(temp_df)

        # Second pass: collect data for bulk insert
        self.logger.info(
            "💾 Preparing to write %d days to database...", len(temp_df)
        )

        all_portfolio_summaries = []
        all_account_data = []
        all_holdings = []
        all_symbol_data = []

        for _, row in temp_df.iterrows():
            date = row["date"]

            # Collect portfolio summary
            all_portfolio_summaries.append(
                {
                    "date": date,
                    "total_value": row["total_value"],
                    "net_cash_flow": row.get("net_cash_flow", 0),
                    "investment_cash_flow": row.get("investment_cash_flow", 0),
                    "income_cash_flow": row.get("income_cash_flow", 0),
                    "portfolio_mod_dietz_return": row.get(
                        "portfolio_daily_return_mod_dietz"
                    ),
                    "portfolio_twr_return": row.get(
                        "portfolio_daily_return_twr"
                    ),
                    "market_value_change": row.get("market_value_change"),
                }
            )

            # Collect account data
            for account in self.accounts:
                all_account_data.append(
                    {
                        "date": date,
                        "account": account,
                        "total_value": row.get(f"{account}_total_value", 0),
                        "cash_flow": row.get(f"{account}_cash_flow", 0),
                        "weight": row.get(f"{account}_weight", 0),
                        "mod_dietz_return": row.get(
                            f"{account}_mod_dietz_return"
                        ),
                        "twr_return": row.get(f"{account}_twr_return"),
                    }
                )

            # Collect holdings (only non-zero)
            for account in self.accounts:
                for symbol in self.symbols:
                    quantity = row.get(f"{account}_{symbol}_quantity", 0)
                    value = row.get(f"{account}_{symbol}_value", 0)
                    if quantity != 0 or value != 0:
                        all_holdings.append(
                            {
                                "date": date,
                                "account": account,
                                "symbol": symbol,
                                "quantity": quantity,
                                "value": value,
                                "weight": row.get(
                                    f"{account}_{symbol}_weight", 0
                                ),
                            }
                        )

            # Collect symbol data
            for symbol in self.symbols:
                all_symbol_data.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "price": row.get(f"{symbol}_price"),
                        "adj_price": row.get(f"{symbol}_adj_price"),
                        "total_quantity": row.get(
                            f"{symbol}_total_quantity", 0
                        ),
                        "total_value": row.get(f"{symbol}_total_value", 0),
                        "weight": row.get(f"{symbol}_weight", 0),
                        "cash_flow": row.get(f"{symbol}_cash_flow", 0),
                        "market_return": row.get(f"{symbol}_market_return"),
                        "twr_return": row.get(f"{symbol}_twr_return"),
                    }
                )

        # Bulk insert all data in a single transaction
        with self.db.transaction():
            self.logger.debug(
                "Bulk inserting %d portfolio summaries...",
                len(all_portfolio_summaries),
            )
            self.db.bulk_insert_portfolio_summaries(all_portfolio_summaries)

            self.logger.debug(
                "Bulk inserting %d account records...", len(all_account_data)
            )
            self.db.bulk_insert_account_data(all_account_data)

            self.logger.debug(
                "Bulk inserting %d holdings...", len(all_holdings)
            )
            self.db.bulk_insert_holdings(all_holdings)

            self.logger.debug(
                "Bulk inserting %d symbol records...", len(all_symbol_data)
            )
            self.db.bulk_insert_symbol_data(all_symbol_data)

        self.logger.info("✅ Wrote %d days to database", len(temp_df))

    def _get_processing_date_range(self) -> pd.DatetimeIndex:
        """Determines the exact dates to process."""
        nyse = mcal.get_calendar("NYSE")
        trading_days = nyse.schedule(
            start_date=self.start_date, end_date=self.end_date
        )
        trading_dates = pd.to_datetime(
            trading_days.index, utc=True
        ).normalize()

        trans_dates_mask = (self.transactions["date"] >= self.start_date) & (
            self.transactions["date"] <= self.end_date
        )
        transaction_dates = pd.to_datetime(
            self.transactions.loc[trans_dates_mask, "date"], utc=True
        ).dt.normalize()
        transaction_dates_idx = pd.DatetimeIndex(transaction_dates.unique())

        all_dates = trading_dates.union(transaction_dates_idx).sort_values()
        return all_dates

    def _build_price_lookup(self) -> Dict[str, Dict]:
        """
        Pre-build efficient price lookup structure for fast O(log n) access.

        Creates sorted arrays of dates and prices for each symbol, enabling
        binary search via numpy.searchsorted instead of repeated DataFrame filtering.

        Returns:
            Dictionary mapping symbol -> {dates, close_prices, adj_close_prices}
        """
        price_lookup = {}

        for symbol, df in self.market_data.items():
            if df.empty:
                price_lookup[symbol] = {
                    "dates": np.array([], dtype="datetime64[ns]"),
                    "close": np.array([]),
                    "adj_close": np.array([]),
                }
                continue

            # Sort by date and extract as numpy arrays for fast lookup
            df_sorted = df.sort_values("date").reset_index(drop=True)

            # Convert to numpy arrays for O(log n) searchsorted lookups
            # Normalize to UTC and remove timezone to avoid numpy warnings
            date_series = pd.to_datetime(df_sorted["date"])
            if date_series.dt.tz is not None:
                # If timezone-aware, convert to UTC first
                dates = (
                    date_series.dt.tz_convert("UTC")
                    .dt.tz_localize(None)
                    .dt.normalize()
                    .values
                )
            else:
                # If already tz-naive, just normalize
                dates = date_series.dt.normalize().values

            close_prices = df_sorted["close"].values
            adj_close_prices = (
                df_sorted["adj_close"].values
                if "adj_close" in df_sorted.columns
                else df_sorted["close"].values
            )

            price_lookup[symbol] = {
                "dates": dates,
                "close": close_prices,
                "adj_close": adj_close_prices,
            }

        return price_lookup

    def _get_price_fast(
        self, symbol: str, price_date: pd.Timestamp, adjusted: bool = False
    ) -> float:
        """
        Fast price lookup using pre-built structure with O(log n) binary search.

        This replaces the O(n) DataFrame filtering in _get_price_for_date with
        numpy's binary search (searchsorted), providing 50-100x speedup per call.

        Args:
            symbol: Stock/ETF symbol
            price_date: Date to get price for
            adjusted: Whether to use adjusted close price

        Returns:
            Price for the given date, with forward/backward fill if exact date not found
        """
        if symbol not in self.price_lookup:
            return 0.0

        data = self.price_lookup[symbol]

        # Handle empty data
        if len(data["dates"]) == 0:
            return 0.0

        # Convert target date to numpy datetime64 for comparison
        # Normalize to UTC and remove timezone to match lookup structure
        target_ts = pd.Timestamp(price_date)
        if target_ts.tz is not None:
            # If timezone-aware, convert to UTC first
            target_date = (
                target_ts.tz_convert("UTC").tz_localize(None).normalize()
            )
        else:
            # If already tz-naive, just normalize
            target_date = target_ts.normalize()
        target_np = target_date.to_datetime64()

        # Use binary search to find insertion point (O(log n) instead of O(n))
        idx = np.searchsorted(data["dates"], target_np, side="right") - 1

        # Select price array based on adjusted flag
        prices = data["adj_close"] if adjusted else data["close"]

        # Exact match or forward fill
        if 0 <= idx < len(prices) and data["dates"][idx] <= target_np:
            return float(prices[idx])

        # Backward fill (price_date is before all available data)
        if idx < 0 and len(prices) > 0:
            return float(prices[0])

        return 0.0

    def _process_one_day(self, date: pd.Timestamp, current_holdings: dict):
        """Processes all transactions and values for a single day."""
        # 1. Update holdings based on today's transactions
        day_transactions = self.transactions[
            self.transactions["date"].dt.date == date.date()
        ]
        for _, trans in day_transactions.iterrows():
            if TransactionTypes.is_quantity_changing(
                trans["transaction_type"]
            ):
                current_holdings[trans["account"]][trans["symbol"]] += trans[
                    "quantity"
                ]

        # 2. Calculate values based on updated holdings
        day_data: Dict[str, Union[float, pd.Timestamp]] = {"date": date}
        total_portfolio_value = 0.0
        symbol_total_shares: Dict[str, float] = {}
        symbol_total_value: Dict[str, float] = {}
        for account in self.accounts:
            account_value = 0.0
            for symbol in self.symbols:
                prev_shares = symbol_total_shares.get(symbol, 0.0)
                prev_value = symbol_total_value.get(symbol, 0.0)

                shares = current_holdings[account][symbol]
                price = self._get_price_fast(symbol, date, adjusted=False)
                value = shares * price
                value = value if not pd.isna(value) else 0.0

                symbol_total_shares[symbol] = prev_shares + shares
                symbol_total_value[symbol] = prev_value + value

                account_value += float(value)
                day_data[f"{account}_{symbol}_quantity"] = shares
                day_data[f"{account}_{symbol}_value"] = value

            day_data[f"{account}_total_value"] = account_value
            total_portfolio_value += account_value
        day_data["total_value"] = total_portfolio_value

        # 3. Add symbol-level totals and prices
        for symbol in self.symbols:
            day_data[f"{symbol}_total_quantity"] = symbol_total_shares.get(
                symbol, 0.0
            )
            day_data[f"{symbol}_total_value"] = symbol_total_value.get(
                symbol, 0.0
            )
            day_data[f"{symbol}_price"] = self._get_price_fast(
                symbol, date, adjusted=False
            )
            day_data[f"{symbol}_adj_price"] = self._get_price_fast(
                symbol, date, adjusted=True
            )

        # 4. Calculate weights: by account, by account-symbol, overall symbol
        for account in self.accounts:
            value_acct = day_data[f"{account}_total_value"]
            value_acct = float(value_acct) if not pd.isna(value_acct) else 0.0
            weight_acct = (
                value_acct / total_portfolio_value
                if total_portfolio_value != 0
                else 0.0
            )
            day_data[f"{account}_weight"] = weight_acct

            for symbol in self.symbols:
                value_acct_symbol = day_data[f"{account}_{symbol}_value"]
                value_acct_symbol = (
                    float(value_acct_symbol)
                    if not pd.isna(value_acct_symbol)
                    else 0.0
                )
                weight_acct_symbol = (
                    value_acct_symbol / total_portfolio_value
                    if total_portfolio_value != 0
                    else 0.0
                )
                day_data[f"{account}_{symbol}_weight"] = weight_acct_symbol

        for symbol in self.symbols:
            value_symbol = day_data[f"{symbol}_total_value"]
            value_symbol = (
                float(value_symbol) if not pd.isna(value_symbol) else 0.0
            )
            weight_symbol = (
                value_symbol / total_portfolio_value
                if total_portfolio_value != 0
                else 0.0
            )
            day_data[f"{symbol}_weight"] = weight_symbol

        return day_data, current_holdings

    def _get_price_for_date(
        self, symbol: str, price_date: pd.Timestamp, adjusted: bool = False
    ) -> float:
        """Gets price for a symbol on a specific date with forward-fill."""
        price_col = "adj_close" if adjusted else "close"

        if symbol not in self.market_data:
            return 0.0

        symbol_data = self.market_data[symbol]
        target_date = to_tzts_scaler(
            price_date, tz=DateAndTimeConstants.TZ_UTC.value
        )
        if target_date is None:
            return 0.0

        exact_match = symbol_data[
            symbol_data["date"].dt.date == target_date.date()
        ]
        if not exact_match.empty:
            return exact_match[price_col].iloc[0]

        # Forward-fill logic
        available_data = symbol_data[
            symbol_data["date"].dt.date <= target_date.date()
        ]
        if not available_data.empty:
            return available_data[price_col].iloc[-1]

        # Backward-fill logic
        future_data = symbol_data[
            symbol_data["date"].dt.date > target_date.date()
        ]
        if not future_data.empty:
            return future_data[price_col].iloc[0]

        return 0.0

    def _compute_cash_flows_vectorized(
        self, portfolio_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute cash flows for all dates using vectorized operations.

        Much faster than applying _process_daily_transactions to each date.

        Args:
            portfolio_df: DataFrame with portfolio history

        Returns:
            DataFrame with cash flow columns added
        """
        df = portfolio_df.copy()

        # Filter transactions to date range
        trans = self.transactions[
            self.transactions["date"].isin(df["date"])
        ].copy()

        if trans.empty:
            # No transactions - all cash flows are zero
            df["investment_cash_flow"] = 0.0
            df["income_cash_flow"] = 0.0
            df["net_cash_flow"] = 0.0

            for acc in self.accounts:
                df[f"{acc}_cash_flow"] = 0.0
            for symbol in self.symbols:
                df[f"{symbol}_cash_flow"] = 0.0

            return df

        # Classify transactions
        is_buy_sell = trans["transaction_type"].apply(
            lambda x: TransactionTypes.is_buy_or_sell(x)
        )
        is_dividend = trans["transaction_type"].apply(
            lambda x: TransactionTypes.is_any_dividend(x)
        )

        trans["is_investment"] = is_buy_sell
        trans["is_income"] = is_dividend

        # Aggregate investment cash flows by date
        inv_by_date = (
            trans[trans["is_investment"]].groupby("date")["total_value"].sum()
        )
        df["investment_cash_flow"] = df["date"].map(inv_by_date).fillna(0)

        # Aggregate income cash flows by date
        inc_by_date = (
            trans[trans["is_income"]].groupby("date")["total_value"].sum()
        )
        df["income_cash_flow"] = df["date"].map(inc_by_date).fillna(0)

        df["net_cash_flow"] = (
            df["investment_cash_flow"] + df["income_cash_flow"]
        )

        # Account-level cash flows
        for acc in self.accounts:
            acc_trans = trans[trans["account"] == acc]
            acc_cf = acc_trans.groupby("date")["total_value"].sum()
            df[f"{acc}_cash_flow"] = df["date"].map(acc_cf).fillna(0)

        # Symbol-level cash flows
        for symbol in self.symbols:
            sym_trans = trans[trans["symbol"] == symbol]
            sym_cf = sym_trans.groupby("date")["total_value"].sum()
            df[f"{symbol}_cash_flow"] = df["date"].map(sym_cf).fillna(0)

        return df

    def _process_daily_transactions(self, date: pd.Timestamp):
        """Calculates cash flows for a specific date."""
        day_trans = self.transactions[
            self.transactions["date"].dt.date == date.date()
        ]

        zero_cf_dict = {acc: 0.0 for acc in self.accounts}
        zero_cf_dict.update({sym: 0.0 for sym in self.symbols})
        inv_cf = zero_cf_dict.copy()
        inc_cf = zero_cf_dict.copy()
        inv_cf["total"] = 0.0
        inc_cf["total"] = 0.0

        for _, trans in day_trans.iterrows():
            acc = trans["account"]
            sym = trans["symbol"]
            ttype = trans["transaction_type"]
            value = trans["total_value"]
            if TransactionTypes.is_buy_or_sell(ttype):
                inv_cf[acc] += value
                inv_cf[sym] += value
                inv_cf["total"] += value
            elif TransactionTypes.is_any_dividend(ttype):
                inc_cf[acc] += value
                inc_cf[sym] += value
                inc_cf["total"] += value
        return inv_cf, inc_cf

    def _add_returns_and_metrics(
        self, portfolio_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Adds cash flow, returns, and other metrics to the history df."""
        df = portfolio_df.sort_values("date").reset_index(drop=True)

        # Use vectorized cash flow computation
        df = self._compute_cash_flows_vectorized(df)

        # Add Ticker-level returns
        for symbol in self.symbols:
            adj_price_col = f"{symbol}_adj_price"
            qty_col = f"{symbol}_total_quantity"
            value_col = f"{symbol}_total_value"
            cf_col = f"{symbol}_cash_flow"

            symbol_active_days = df[qty_col] != 0

            short_position_multiplier = pd.Series(
                np.where(df[qty_col] <= 0, -1, 1), index=df.index
            )
            # Calculate the daily percentage change of the adjusted price.
            # This is the true market return for the symbol.
            df[f"{symbol}_market_return"] = (
                df[adj_price_col].pct_change().fillna(0)
                * short_position_multiplier
                * symbol_active_days
            )

            df[f"{symbol}_twr_return"] = (
                calculate_twr_daily_returns(df[value_col], df[cf_col])
                * short_position_multiplier
                * symbol_active_days
            )

        # Add returns
        df["portfolio_daily_return_mod_dietz"] = (
            calculate_modified_dietz_returns(df, config=self.config)
        )
        df["portfolio_daily_return_twr"] = calculate_twr_daily_returns(
            df["total_value"], df["net_cash_flow"]
        )

        for acc in self.accounts:
            df[f"{acc}_mod_dietz_return"] = (
                calculate_account_modified_dietz_returns(
                    df, acc, config=self.config
                )
            )
            df[f"{acc}_twr_return"] = calculate_account_twr_daily_returns(
                df, acc
            )

        df = calculate_market_change_and_returns(df)

        return df

    def get_database(self) -> PortfolioDatabase:
        """Get the database instance."""
        return self.db
