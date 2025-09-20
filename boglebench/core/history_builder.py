"""
Builds the daily portfolio history from transactions and market data.
"""

from typing import Dict, Union

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
from ..utils.config import ConfigManager
from ..utils.logging_config import get_logger
from ..utils.tools import to_tzts_scaler


class PortfolioHistoryBuilder:
    """Builds the daily portfolio history DataFrame."""

    def __init__(
        self,
        config: ConfigManager,
        transactions: pd.DataFrame,
        market_data: dict,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ):
        self.config = config
        self.transactions = transactions
        self.market_data = market_data
        self.start_date = start_date
        self.end_date = end_date
        self.logger = get_logger()
        self.tickers = self.transactions["ticker"].unique().tolist()
        self.accounts = self.transactions["account"].unique().tolist()

    def build(self) -> pd.DataFrame:
        """The main method to build and return the portfolio history."""
        self.logger.info("🏗️ Building portfolio history...")

        date_range = self._get_processing_date_range()
        if date_range.empty:
            self.logger.warning(
                "No dates to process for building portfolio history."
            )
            return pd.DataFrame()

        holdings = {
            acc: {tck: 0.0 for tck in self.tickers} for acc in self.accounts
        }
        portfolio_data = []

        for date in date_range:
            day_data, holdings = self._process_one_day(date, holdings)
            portfolio_data.append(day_data)

        portfolio_df = pd.DataFrame(portfolio_data)
        portfolio_df = self._add_returns_and_metrics(portfolio_df)

        self.logger.info(
            "✅ Portfolio history built: %d days", len(portfolio_df)
        )
        return portfolio_df

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
                current_holdings[trans["account"]][trans["ticker"]] += trans[
                    "quantity"
                ]

        # 2. Calculate values based on updated holdings
        day_data: Dict[str, Union[float, pd.Timestamp]] = {"date": date}
        total_portfolio_value = 0.0
        ticker_total_shares: Dict[str, float] = {}
        ticker_total_value: Dict[str, float] = {}
        for account in self.accounts:
            account_value = 0.0
            for ticker in self.tickers:
                prev_shares = ticker_total_shares.get(ticker, 0.0)
                prev_value = ticker_total_value.get(ticker, 0.0)

                shares = current_holdings[account][ticker]
                price = self._get_price_for_date(ticker, date)
                value = shares * price
                value = value if not pd.isna(value) else 0.0

                ticker_total_shares[ticker] = prev_shares + shares
                ticker_total_value[ticker] = prev_value + value

                account_value += float(value)
                day_data[f"{account}_{ticker}_shares"] = shares
                day_data[f"{account}_{ticker}_value"] = value

            day_data[f"{account}_total"] = account_value
            total_portfolio_value += account_value
        day_data["total_value"] = total_portfolio_value

        # 3. Add ticker-level totals and prices
        for ticker in self.tickers:
            day_data[f"{ticker}_total_shares"] = ticker_total_shares.get(
                ticker, 0.0
            )
            day_data[f"{ticker}_total_value"] = ticker_total_value.get(
                ticker, 0.0
            )
            day_data[f"{ticker}_price"] = self._get_price_for_date(
                ticker, date
            )

        # 4. Calculate weights: by account, by account-ticker, overall ticker
        for account in self.accounts:
            value_acct = day_data[f"{account}_total"]
            value_acct = float(value_acct) if not pd.isna(value_acct) else 0.0
            weight_acct = (
                value_acct / total_portfolio_value
                if total_portfolio_value != 0
                else 0.0
            )
            day_data[f"{account}_weight"] = weight_acct

            for ticker in self.tickers:
                value_acct_ticker = day_data[f"{account}_{ticker}_value"]
                value_acct_ticker = (
                    float(value_acct_ticker)
                    if not pd.isna(value_acct_ticker)
                    else 0.0
                )
                weight_acct_ticker = (
                    value_acct_ticker / total_portfolio_value
                    if total_portfolio_value != 0
                    else 0.0
                )
                day_data[f"{account}_{ticker}_weight"] = weight_acct_ticker

        for ticker in self.tickers:
            value_ticker = day_data[f"{ticker}_total_value"]
            value_ticker = (
                float(value_ticker) if not pd.isna(value_ticker) else 0.0
            )
            weight_ticker = (
                value_ticker / total_portfolio_value
                if total_portfolio_value != 0
                else 0.0
            )
            day_data[f"{ticker}_weight"] = weight_ticker

        return day_data, current_holdings

    def _get_price_for_date(
        self, ticker: str, price_date: pd.Timestamp
    ) -> float:
        """Gets price for a ticker on a specific date with forward-fill."""
        if ticker not in self.market_data:
            return 0.0

        ticker_data = self.market_data[ticker]
        target_date = to_tzts_scaler(
            price_date, tz=DateAndTimeConstants.TZ_UTC.value
        )
        if target_date is None:
            return 0.0

        exact_match = ticker_data[
            ticker_data["date"].dt.date == target_date.date()
        ]
        if not exact_match.empty:
            return exact_match["close"].iloc[0]

        # Forward-fill logic
        available_data = ticker_data[
            ticker_data["date"].dt.date <= target_date.date()
        ]
        if not available_data.empty:
            return available_data["close"].iloc[-1]

        # Backward-fill logic
        future_data = ticker_data[
            ticker_data["date"].dt.date > target_date.date()
        ]
        if not future_data.empty:
            return future_data["close"].iloc[0]

        return 0.0

    def _process_daily_transactions(self, date: pd.Timestamp):
        """Calculates cash flows for a specific date."""
        day_trans = self.transactions[
            self.transactions["date"].dt.date == date.date()
        ]
        inv_cf = {acc: 0.0 for acc in self.accounts}
        inc_cf = {acc: 0.0 for acc in self.accounts}
        inv_cf["total"] = 0.0
        inc_cf["total"] = 0.0

        for _, trans in day_trans.iterrows():
            acc = trans["account"]
            ttype = trans["transaction_type"]
            value = trans["total_value"]
            if TransactionTypes.is_buy_or_sell(ttype):
                inv_cf[acc] += value
                inv_cf["total"] += value
            elif TransactionTypes.is_any_dividend(ttype):
                inc_cf[acc] += value
                inc_cf["total"] += value
        return inv_cf, inc_cf

    def _add_returns_and_metrics(
        self, portfolio_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Adds cash flow, returns, and other metrics to the history df."""
        df = portfolio_df.sort_values("date").reset_index(drop=True)

        # Add cash flows
        cash_flows = df["date"].apply(self._process_daily_transactions)
        inv_cfs, inc_cfs = zip(*cash_flows)
        df["investment_cash_flow"] = [cf["total"] for cf in inv_cfs]
        df["income_cash_flow"] = [cf["total"] for cf in inc_cfs]
        df["net_cash_flow"] = (
            df["investment_cash_flow"] + df["income_cash_flow"]
        )

        # Add account-specific cash flows
        for acc in self.accounts:
            df[f"{acc}_cash_flow"] = [
                inv.get(acc, 0.0) + inc.get(acc, 0.0)
                for inv, inc in zip(inv_cfs, inc_cfs)
            ]

        # Add Ticker-level returns
        for ticker in self.tickers:
            price_col = f"{ticker}_price"
            # Calculate the daily percentage change of the price.
            # This is the true market return for the ticker.
            df[f"{ticker}_return"] = df[price_col].pct_change().fillna(0)

        # Add returns
        df["portfolio_daily_return_mod_dietz"] = (
            calculate_modified_dietz_returns(df, config=self.config)
        )
        df["portfolio_daily_return_twr"] = calculate_twr_daily_returns(df)

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
