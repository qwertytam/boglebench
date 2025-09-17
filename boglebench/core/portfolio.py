"""
Core BogleBench portfolio analyzer for performance analysis and benchmarking.

This module implements the main BogleBenchAnalyzer class that handles:
- Transaction data loading and processing
- Market data acquisition
- Portfolio construction over time
- Performance metrics calculation
- Benchmark comparison

Following John Bogle's investment principles of simplicity, low costs,
and long-term focus.
"""

import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from zoneinfo import ZoneInfo  # pylint: disable=wrong-import-order

import pandas as pd
import pandas_market_calendars as mcal  # type: ignore

from ..core.constants import DateAndTimeConstants, Defaults, TransactionTypes
from ..core.dividend_validator import (
    DividendValidator,
    identify_any_dividend_transactions,
    identify_quantity_change_transactions,
)
from ..core.market_data import MarketDataProvider
from ..core.metrics import (
    calculate_account_modified_dietz_returns,
    calculate_account_twr_daily_returns,
    calculate_irr,
    calculate_metrics,
    calculate_modified_dietz_returns,
    calculate_relative_metrics,
    calculate_twr_daily_returns,
)
from ..core.results import PerformanceResults
from ..core.transaction_loader import load_validate_transactions
from ..core.types import DateLike
from ..utils.config import ConfigManager
from ..utils.logging_config import get_logger, setup_logging
from ..utils.tools import is_tz_aware, to_tzts_scaler
from ..utils.workspace import WorkspaceContext


class BogleBenchAnalyzer:
    """
    Main analyzer class for portfolio performance analysis and benchmarking.

    This class embodies John Bogle's investment philosophy:
    - Focus on long-term performance over short-term fluctuations
    - Emphasis on low-cost, broad market exposure
    - Simple, transparent analysis without unnecessary complexity
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the BogleBench analyzer.

        Args:
            config_path: Path to configuration file.
                        If None, uses default locations.
        """

        # Set workspace context if config provided
        if config_path:
            config_file = Path(config_path).expanduser()
            if config_file.exists():
                WorkspaceContext.discover_workspace(config_file.parent)

        self.config = ConfigManager(config_path)
        # print("DEBUG: Config loaded from:", self.config.config_path)
        # print("DEBUG: BogleBench Setting up logging...")
        setup_logging()  # Initialize after workspace context is set
        self.logger = get_logger("core.portfolio")

        self._start_date: Optional[pd.Timestamp] = None
        initial_start_date = self.config.get("analysis.start_date", None)
        self.start_date = initial_start_date

        self._end_date: Optional[pd.Timestamp] = None
        initial_end_date = self.config.get("analysis.end_date", None)
        self.end_date = initial_end_date

        initial_local_tz = self.config.get(
            "advanced.market_data.local_tz", DateAndTimeConstants.TZ_UTC.value
        )
        self.local_tz = initial_local_tz

        self.transactions = pd.DataFrame()
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.portfolio_history = pd.DataFrame()
        self.benchmark_data = pd.DataFrame()
        self.performance_results = PerformanceResults()

        # Set up market data provider
        initial_api_key = self.config.get(
            "api.alpha_vantage_key", Defaults.DEFAULT_API_KEY
        )
        self.api_key = initial_api_key

        cache_enabled = self.config.get("settings.cache_market_data", True)
        if isinstance(cache_enabled, dict):
            cache_enabled = cache_enabled.get("value", True)
        if cache_enabled is None:
            cache_enabled = True

        cache_dir = self.config.get_market_data_path()

        initial_market_data_provider = MarketDataProvider(
            api_key=self.api_key,
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
            force_cache_refresh=False,
        )
        self.market_data_provider = initial_market_data_provider

        # Suppress warnings for cleaner output
        warnings.filterwarnings("ignore", category=FutureWarning)
        self.logger.info("BogleBench analyzer initialized")

    @property
    def start_date(self) -> Optional[pd.Timestamp]:
        """Get the analysis start date."""
        return self._start_date

    @start_date.setter
    def start_date(self, value: Union[DateLike, dict, None]) -> None:
        """Set the analysis start date.

        Args:
            value: Date-like object (str, pd.Timestamp, datetime) or None
                   to set the start date.

        Raises:
            ValueError: If the provided date is invalid or not timezone-aware.

        Returns:
            None

        """

        if isinstance(value, dict):
            value = value.get("value", None)

        if value is not None and not isinstance(value, dict):
            value = to_tzts_scaler(value, tz=DateAndTimeConstants.TZ_UTC.value)
            if value is None:
                raise ValueError(
                    "Unable to set start date: Invalid start date provided"
                )
            if not is_tz_aware(value):
                raise ValueError("Unable to set timezone-aware start date")
            self._start_date = value
            self.logger.debug(
                "Using configured start date: %s", self.start_date
            )
            return
        elif value is None:
            self._start_date = None
            self.logger.debug("'Start date provided as None, clearing it")
            return
        elif isinstance(value, dict):
            raise ValueError(
                f"Unable to set start date: Invalid start date provided {value}"
            )

    @property
    def end_date(self) -> Optional[pd.Timestamp]:
        """Get the analysis end date."""
        return self._end_date

    @end_date.setter
    def end_date(self, value: Union[DateLike, dict, None]) -> None:
        """Set the analysis end date.
        Rules:
        1. If value is provided, use it.
        2. Otherwise, if the market is currently open, use the last closed market day.
        3. Otherwise, use the current datetime as the end date.
        Args:
            value: Date-like object (str, pd.Timestamp, datetime) or None
                   to clear the end date.
        Raises:
            ValueError: If the provided date is invalid or not timezone-aware.
        Returns:
            None
        """

        # Ensure the config value is not a dict before passing to pd.Timestamp
        if isinstance(value, dict):
            value = value.get("value", None)

        if value is not None and not isinstance(value, dict):
            self._end_date = to_tzts_scaler(
                value, tz=DateAndTimeConstants.TZ_UTC.value
            )
            self.logger.debug("Using configured end date: %s", self.end_date)
            return

        last_market_close_date = None
        if self._is_market_currently_open():
            self.logger.info("Market is currently open")
            last_market_close_date = self._get_last_closed_market_day()
        else:
            self.logger.info("Market is currently closed")

            # Ensure last_market_close_date is a scalar Timestamp,
            # not a Series or None
            local_tz = ZoneInfo(str(DateAndTimeConstants.TZ_UTC.value))
            dt_now = to_tzts_scaler(datetime.now(tz=local_tz))
            last_market_close_date = dt_now

        if last_market_close_date is not None:
            self._end_date = to_tzts_scaler(
                last_market_close_date, tz=DateAndTimeConstants.TZ_UTC.value
            )
            self.logger.info(
                "Using last market close date as end date: %s",
                self.end_date,
            )
        else:
            raise ValueError("No valid end date provided or found.")

        return

    @property
    def local_tz(self) -> ZoneInfo:
        """Get the local timezone for date handling."""
        return self._local_tz

    @local_tz.setter
    def local_tz(self, tz_str: Union[dict, str, ZoneInfo, None]) -> None:
        if isinstance(tz_str, dict):
            tz_str = tz_str.get("value", DateAndTimeConstants.TZ_UTC.value)
        if isinstance(tz_str, str):
            self._local_tz = ZoneInfo(tz_str)
        elif isinstance(tz_str, ZoneInfo):
            self._local_tz = tz_str
        elif tz_str is None:
            self._local_tz = ZoneInfo(DateAndTimeConstants.TZ_UTC.value)
        else:
            raise ValueError("Invalid timezone value provided")
        self.logger.debug("Local timezone set to: %s", self._local_tz)

    @property
    def api_key(self) -> str:
        """Get the API key for market data provider."""
        return self._api_key

    @api_key.setter
    def api_key(self, key: Union[dict, str, None]) -> None:
        if isinstance(key, dict):
            key = key.get("value", Defaults.DEFAULT_API_KEY)
        if key is None or str(key).strip() == "":
            key = Defaults.DEFAULT_API_KEY

        if key is not None and isinstance(key, str) and key.strip() != "":
            self._api_key = key.strip()
            # self.market_data_provider.api_key = self._api_key
            self.logger.debug("Market data API key set")
        else:
            self._api_key = Defaults.DEFAULT_API_KEY
            # self.market_data_provider.api_key = None
            self.logger.info(
                "No market data API key provided; using limited access"
            )

    def load_transactions(
        self, file_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load and validate transaction data from CSV file.

        Args:
            file_path: Path to transactions CSV. If None, uses config path.

        Returns:
            DataFrame with processed transaction data

        Raises:
            FileNotFoundError: If transaction file doesn't exist
            ValueError: If required columns are missing or data is invalid
        """
        if file_path is None:
            file_path = str(self.config.get_transactions_path())

        self.logger.info("📄 Loading transactions from: %s", file_path)

        # Get and store processed transactions
        self.transactions = load_validate_transactions(Path(file_path))

        self.logger.info(
            "✅ Loaded %d transactions for %d unique assets",
            len(self.transactions),
            self.transactions["ticker"].nunique(),
        )
        self.logger.info(
            "📅 Transactions date range: %s to %s",
            self.transactions["date"].dt.date.min(),
            self.transactions["date"].dt.date.max(),
        )
        self.logger.info(
            "🏦 %d accounts: %s",
            len(self.transactions["account"].unique()),
            ", ".join(self.transactions["account"].unique()),
        )
        total_invested = self.transactions[
            self.transactions["total_value"] > 0
        ]["total_value"].sum()
        msg = f"💰 Total invested: ${total_invested:,.2f}"
        self.logger.info(msg)

        return self.transactions

    def _is_market_currently_open(self) -> bool:
        """Check if the market is currently open."""
        nyse = mcal.get_calendar("NYSE")
        now = datetime.now(tz=ZoneInfo(DateAndTimeConstants.TZ_UTC))
        schedule = nyse.schedule(start_date=now.date(), end_date=now.date())

        self.logger.debug("Checking market status for %s", now)
        self.logger.debug("Market schedule for today:\n%s", schedule)

        if schedule.empty:
            self.logger.debug("Market is closed today (holiday or weekend)")
            return False

        market_open = schedule.iloc[0]["market_open"].to_pydatetime()
        market_close = schedule.iloc[0]["market_close"].to_pydatetime()
        self.logger.debug(
            "Market hours today: %s to %s", market_open, market_close
        )
        return market_open <= now <= market_close

    def _get_last_closed_market_day(self) -> pd.Timestamp:
        """Get the most recent market day (last trading day) that closed."""
        nyse = mcal.get_calendar("NYSE")
        today = datetime.now(tz=ZoneInfo("America/New_York"))
        schedule = nyse.schedule(
            start_date=today
            - timedelta(days=Defaults.DEFAULT_LOOK_FORWARD_PRICE_DATA),
            end_date=today,
        )
        if schedule.empty:
            raise ValueError(
                f"No recent market days found in the last "
                f"{Defaults.DEFAULT_LOOK_FORWARD_PRICE_DATA} days"
            )

        closed_days = schedule[schedule["market_close"] < today]
        last_closed_market_day = closed_days["market_close"].max()

        self.logger.debug("Using now time of %s", today)
        self.logger.debug(
            "Last closed market day is %s", last_closed_market_day
        )
        return pd.to_datetime(last_closed_market_day)

    def _get_price_for_date(
        self, ticker: str, price_date: pd.Timestamp
    ) -> float:
        """Get price for ticker on specific date with forward-fill logic."""
        if ticker not in self.market_data:
            raise ValueError(f"No market data available for {ticker}")

        ticker_data = self.market_data[ticker]
        target_date = to_tzts_scaler(
            price_date, tz=DateAndTimeConstants.TZ_UTC.value
        )
        if target_date is None:
            raise ValueError("Invalid price date provided")

        # Try exact date match first
        exact_match = ticker_data[
            ticker_data["date"].dt.date == target_date.date()
        ]
        if not exact_match.empty:
            return exact_match["close"].iloc[0]

        # Forward fill: use most recent price before target date
        self.logger.debug(
            "Forward-filling price for %s on %s", ticker, target_date
        )

        available_data = ticker_data[
            ticker_data["date"].dt.date <= target_date.date()
        ]

        if not available_data.empty:
            days_back = (
                target_date.date() - available_data["date"].iloc[-1].date()
            ).days
            if days_back <= 7:  # Only forward-fill up to 7 days
                return available_data["close"].iloc[-1]
            else:
                self.logger.warning(
                    "⚠️  Warning: No recent price data for %s near %s",
                    ticker,
                    target_date,
                )
                return available_data["close"].iloc[
                    -1
                ]  # Use it anyway but warn

        # Backward fill: use next available price after target date
        self.logger.debug(
            "Backward-filling price for %s on %s", ticker, target_date
        )
        future_data = ticker_data[
            ticker_data["date"].dt.date > target_date.date()
        ]
        if not future_data.empty:
            days_forward = (
                future_data["date"].iloc[0].date() - target_date.date()
            ).days
            self.logger.info(
                "Using future price for %s on %s (%d days forward)",
                ticker,
                target_date,
                days_forward,
            )
            return future_data["close"].iloc[0]

        # If we get here, no data exists at all
        raise ValueError(
            f"No price data available for {ticker} around {target_date}"
        )

    def _get_shares_held_on_date(
        self, ticker: str, date: pd.Timestamp, account: Optional[str] = None
    ) -> float:
        """
        Retrieves the number of shares held for a specific ticker on a given date.

        Args:
            ticker: The stock ticker symbol.
            date: The date for which to retrieve the share quantity.
            account: Optional account identifier to filter by account.

        Returns:
            The number of shares held on the specified date.
        """

        # Filter transactions for the specific ticker and date
        mask = (
            (self.transactions["ticker"] == ticker)
            & (self.transactions["date"].dt.date < date.date())
            & identify_quantity_change_transactions(
                self.transactions["transaction_type"]
            )
        )

        if account:
            mask &= self.transactions["account"] == account

        if self.start_date is not None:
            mask &= self.transactions["date"].dt.date >= self.start_date.date()

        shares_held = self.transactions.loc[mask, "quantity"].sum()
        if shares_held is None:
            shares_held = 0.0

        if isinstance(shares_held, pd.Series):
            shares_held = shares_held.sum()

        return float(shares_held)

    def _process_daily_transactions(
        self, date: pd.Timestamp
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Process all transactions for a specific date and return cash
        flows by account."""
        if isinstance(date, str):
            date = pd.to_datetime(date)

        day_transactions = self.transactions[
            self.transactions["date"].dt.date == date.date()
        ]

        self.logger.debug(
            "Processing daily %d transactions for %s",
            len(day_transactions),
            date.strftime("%Y-%m-%d"),
        )

        # Investment cash flows: affect cost basis (BUY/SELL)
        inv_cf = {"total": Defaults.ZERO_CASH_FLOW}

        # Income cash flows: dividends, fees (do not affect cost basis)
        inc_cf = {"total": Defaults.ZERO_CASH_FLOW}

        # Initialize all accounts
        for account in self.transactions["account"].unique():
            inv_cf[account] = Defaults.ZERO_CASH_FLOW
            inc_cf[account] = Defaults.ZERO_CASH_FLOW

        # Process each transaction
        for _, trans in day_transactions.iterrows():
            account = trans["account"]
            ttype = trans["transaction_type"]
            self.logger.debug(
                "  Processing %s of value $%.2f for %s in account %s",
                ttype,
                trans["total_value"],
                trans["ticker"],
                account,
            )

            if TransactionTypes.is_buy_or_sell(ttype):
                cf = trans["total_value"]  # +ve for BUY, -ve for SELL
                inv_cf[account] += cf
                inv_cf["total"] += cf

            elif TransactionTypes.is_any_dividend(ttype):
                cf = trans["total_value"]
                self.logger.debug(
                    "  Processing %s of cash flow $%.2f for %s in account %s",
                    ttype,
                    cf,
                    trans["ticker"],
                    account,
                )
                inc_cf[account] += cf
                inc_cf["total"] += cf

            elif TransactionTypes.is_fee(ttype):
                # To implement: Process fee transactions
                continue

            elif TransactionTypes.is_share_transfer(ttype):
                # To implement: Process share transfer transactions
                continue

            elif TransactionTypes.is_split(ttype):
                # To implement: Process split transactions
                continue

            elif TransactionTypes.is_other(ttype):
                # To implement: Process other transaction types
                continue

        self.logger.debug("Investment cash flows: %s", inv_cf)
        self.logger.debug("Income cash flows: %s", inc_cf)

        return inv_cf, inc_cf

    def build_portfolio_history(self) -> pd.DataFrame:
        """
        Build portfolio holdings and values over time.

        Returns:
            DataFrame with daily portfolio holdings, values, and weights
        """
        if self.transactions.empty:
            raise ValueError("Must load transactions first")

        self.logger.info("🏗️  Building portfolio history...")

        # Get all unique tickers and accounts
        tickers = self.transactions["ticker"].unique().tolist()
        accounts = self.transactions["account"].unique()

        # Initialize portfolio tracking by account and ticker
        portfolio_data = []

        # Track holdings by account and ticker: {account: {ticker: shares}}
        current_holdings = {
            account: {ticker: 0.0 for ticker in tickers}
            for account in accounts
        }

        if self.start_date is None:
            min_transaction_date = self.transactions["date"].min().date()
            self.start_date = min_transaction_date

            if self.start_date is None:
                raise ValueError(
                    "Start date is not set after loading transaction dates"
                )

        max_transaction_date = self.transactions["date"].max().date()
        if self.start_date.date() > max_transaction_date:  # type: ignore
            return pd.DataFrame()  # No data to process

        if self.start_date > self.end_date:  # type: ignore
            raise ValueError(
                "Start date must be on or before end date: "
                f"{self.start_date} > {self.end_date}"
            )

        # Default is all day between and including start and end dates that are
        # trading days.
        # However, will also include any transaction dates that fall outside
        # this range to ensure all transactions are processed.
        # Get actual NYSE trading days (excludes weekends AND holidays)

        # 1. Get all trading days in the period
        self.logger.debug(
            "Getting NYSE trading days from %s to %s",
            self.start_date,
            self.end_date,
        )
        nyse = mcal.get_calendar("NYSE")
        trading_days = nyse.schedule(
            start_date=self.start_date,
            end_date=self.end_date,
        )

        trading_dates_set = set(
            [pd.to_datetime(date).normalize() for date in trading_days.index]
        )
        trading_dates = pd.to_datetime(list(trading_dates_set)).tz_localize(
            DateAndTimeConstants.TZ_UTC.value
        )

        self.logger.debug("trading_dates:\n%s", trading_dates)

        # 2. Get all unique transaction dates between start and end dates
        # inclusive
        transaction_dates_mask = (
            self.transactions["date"].dt.date >= self.start_date.date()  # type: ignore
        ) & (
            self.transactions["date"].dt.date <= self.end_date.date()  # type: ignore
        )

        transaction_dates = list(
            set(
                pd.to_datetime(
                    self.transactions.loc[
                        transaction_dates_mask, "date"
                    ].dt.normalize()
                )
            )
        )

        # 3. Find transactions dates that are outside of trading days
        # self.logger.debug("Checking for transactions on non-trading days")
        non_trading_transaction_dates = set(transaction_dates) - set(
            trading_dates
        )

        if non_trading_transaction_dates:
            self.logger.warning(
                "⚠️  Warning: %d transactions on non-trading days:\n%s",
                len(non_trading_transaction_dates),
                "\n".join(str(date) for date in non_trading_transaction_dates),
            )

        all_dates_to_process = sorted(
            list(trading_dates.union(list(non_trading_transaction_dates)))
        )

        date_range = pd.to_datetime(all_dates_to_process, utc=True)

        self.logger.info(
            "📅 Processing %d trading days from %s to %s",
            len(date_range),
            pd.Timestamp(date_range[0]).date(),
            pd.Timestamp(date_range[-1]).date(),
        )

        self.logger.debug(
            "Starting to process %d transactions by date",
            len(self.transactions),
        )

        self.logger.debug(
            "Fetching market data for portfolio tickers: %s", tickers
        )

        force_refresh_market_data = self.config.get(
            "settings.force_refresh_market_data", False
        )

        if (
            force_refresh_market_data
            or self.market_data is None
            or not self.market_data
        ):
            self.market_data = self.market_data_provider.get_market_data(
                tickers=tickers,
                start_date_str=self.start_date.strftime("%Y-%m-%d"),  # type: ignore
                end_date_str=self.end_date.strftime("%Y-%m-%d"),  # type: ignore
            )

            benchmark_ticker = self.config.get(
                "settings.benchmark_ticker", Defaults.DEFAULT_BENCHMARK_TICKER
            )
            if isinstance(benchmark_ticker, dict):
                benchmark_ticker = benchmark_ticker.get(
                    "value", Defaults.DEFAULT_BENCHMARK_TICKER
                )
            if benchmark_ticker is None or benchmark_ticker.strip() == "":
                benchmark_ticker = Defaults.DEFAULT_BENCHMARK_TICKER

            # Separate benchmark data for easy access
            if benchmark_ticker in self.market_data:
                self.benchmark_data = self.market_data[benchmark_ticker].copy()

        # Validate dividend data for all tickers before building portfolio
        if self.config.get("dividend.auto_validate", True):
            self.logger.debug(
                "🔍 Validating dividend data against market data..."
            )

            if self.market_data is None or not self.market_data:
                self.logger.warning(
                    "⚠️  No market data available for dividend validation"
                )
            else:
                validator = DividendValidator(
                    transactions_df=self.transactions,
                    market_data_df=self.market_data,
                    start_date=self.start_date,
                    end_date=self.end_date,
                )

                messages, div_differences = validator.validate()

                warn_missing_dividends = self.config.get(
                    "dividend.warn_missing_dividends", True
                )
                if not isinstance(warn_missing_dividends, bool):
                    warn_missing_dividends = True
                if warn_missing_dividends:
                    for msg in messages:
                        if (
                            "mismatch" in msg.lower()
                            or "warning" in msg.lower()
                            or "missing" in msg.lower()
                            or "extra" in msg.lower()
                        ):
                            self.logger.warning("⚠️  %s", msg)
                        else:
                            self.logger.info("ℹ️  %s", msg)

                auto_calc_div_per_share = self.config.get(
                    "dividend.auto_calculate_div_per_share", True
                )
                if not isinstance(auto_calc_div_per_share, bool):
                    auto_calc_div_per_share = True

                if auto_calc_div_per_share and div_differences:
                    self.logger.debug(
                        "Automatically adjusting user dividends "
                        + "based on market data..."
                    )
                    for ticker, df in div_differences.items():
                        for _, row in df.iterrows():
                            date = row["date"]
                            new_div_per_share = row["value_per_share_market"]
                            if isinstance(new_div_per_share, pd.Series):
                                new_div_per_share = new_div_per_share.iloc[0]
                            elif not isinstance(
                                new_div_per_share, (int, float)
                            ):
                                new_div_per_share = 0
                            mask = (
                                (self.transactions["ticker"] == ticker)
                                & (
                                    self.transactions["date"].dt.date
                                    == date.date()
                                )
                                & (
                                    identify_any_dividend_transactions(
                                        self.transactions["transaction_type"]
                                    )
                                )
                                & (
                                    self.transactions["account"]
                                    == row["account"]
                                )
                            )

                            if not self.transactions.loc[mask].empty:
                                shares = self._get_shares_held_on_date(
                                    ticker, date, account=row["account"]
                                )

                                old_value_per_share = self.transactions.loc[
                                    mask, "value_per_share"
                                ]
                                if isinstance(old_value_per_share, pd.Series):
                                    old_value_per_share = (
                                        old_value_per_share.iloc[0]
                                    )
                                elif not isinstance(
                                    old_value_per_share, (int, float)
                                ):
                                    old_value_per_share = 0

                                old_total_value = self.transactions.loc[
                                    mask, "total_value"
                                ]
                                if isinstance(old_total_value, pd.Series):
                                    old_total_value = old_total_value.sum()
                                elif not isinstance(
                                    old_total_value, (int, float)
                                ):
                                    old_total_value = 0

                                new_total_value = -shares * new_div_per_share
                                self.logger.info(
                                    "ℹ️  Updating %s dividend on %s: "
                                    "per share from %.4f to %.4f, "
                                    "total from $%.2f to $%.2f",
                                    ticker,
                                    date.date(),
                                    old_value_per_share,
                                    new_div_per_share,
                                    old_total_value,
                                    new_total_value,
                                )
                                self.transactions.loc[
                                    mask, "value_per_share"
                                ] = new_div_per_share
                                self.transactions.loc[mask, "total_value"] = (
                                    new_total_value
                                )
                            else:
                                self.logger.warning(
                                    "⚠️ No matching transaction found to"
                                    + " update for %s on %s",
                                    ticker,
                                    date.date(),
                                )

        for date in date_range:
            day_transactions = self.transactions[
                self.transactions["date"].dt.date == date.date()
            ]

            for _, transaction in day_transactions.iterrows():
                account = transaction["account"]
                ticker = transaction["ticker"]
                ttype = transaction["transaction_type"]

                if TransactionTypes.is_quantity_changing(ttype):
                    shares = transaction["quantity"]

                    # Ensure account and ticker exist in our tracking
                    if account not in current_holdings:
                        current_holdings[account] = {
                            t: Defaults.ZERO_ASSET_VALUE for t in tickers
                        }
                    if ticker not in current_holdings[account]:
                        current_holdings[account][
                            ticker
                        ] = Defaults.ZERO_ASSET_VALUE

                    # For DIVIDEND, do not adjust holdings;
                    # handled in cash flow
                    current_holdings[account][ticker] += shares

            # Get market prices for this date
            day_data: Dict[str, Any] = {"date": date}
            total_portfolio_value = Defaults.ZERO_ASSET_VALUE
            account_totals = {}

            for account in accounts:
                account_value = Defaults.ZERO_ASSET_VALUE

                for ticker in tickers:
                    shares = current_holdings[account][ticker]

                    try:
                        price = self._get_price_for_date(ticker, date)
                    except ValueError as e:
                        self.logger.error(
                            "Skipping %s on %s: %s", ticker, date, e
                        )
                        price = (
                            Defaults.ZERO_PRICE
                        )  # Only set to 0 if truly no data exists

                    position_value = shares * price
                    account_value += position_value

                    # Store account-specific position data
                    day_data[f"{account}_{ticker}_shares"] = shares
                    day_data[f"{account}_{ticker}_price"] = price
                    day_data[f"{account}_{ticker}_value"] = position_value

                account_totals[account] = account_value
                total_portfolio_value += account_value
                day_data[f"{account}_total"] = account_value

            day_data["total_value"] = total_portfolio_value

            # Also calculate consolidated positions across all accounts
            for ticker in tickers:
                total_shares = sum(
                    current_holdings[account][ticker] for account in accounts
                )

                # Use the same price as calculated above
                if ticker in self.market_data:
                    ticker_data = self.market_data[ticker]
                    price_data = ticker_data[
                        ticker_data["date"].dt.date == date.date()
                    ]
                    if not price_data.empty:
                        price = price_data["close"].iloc[0]
                    else:
                        available_data = ticker_data[
                            ticker_data["date"].dt.date <= date.date()
                        ]
                        if not available_data.empty:
                            price = available_data["close"].iloc[-1]
                        else:
                            price = Defaults.ZERO_PRICE
                else:
                    price = Defaults.ZERO_PRICE

                day_data[f"{ticker}_total_shares"] = total_shares
                day_data[f"{ticker}_total_value"] = total_shares * price

            portfolio_data.append(day_data)

        portfolio_df = pd.DataFrame(portfolio_data)
        portfolio_df = portfolio_df.sort_values("date").reset_index(drop=True)
        portfolio_df["net_cash_flow"] = Defaults.ZERO_CASH_FLOW
        portfolio_df["weighted_cash_flow"] = Defaults.ZERO_CASH_FLOW
        portfolio_df["market_value_change"] = Defaults.ZERO_ASSET_VALUE
        portfolio_df["market_value_return"] = Defaults.ZERO_RETURN

        # Calculate weights by account and overall
        for account in accounts:
            account_total_col = f"{account}_total"
            if account_total_col in portfolio_df.columns:
                for ticker in tickers:
                    value_col = f"{account}_{ticker}_value"
                    weight_col = f"{account}_{ticker}_weight"
                    if value_col in portfolio_df.columns:
                        portfolio_df[weight_col] = (
                            portfolio_df[value_col]
                            / portfolio_df[account_total_col]
                        ).fillna(Defaults.DEFAULT_CASH_FLOW_WEIGHT)

        # Calculate overall weights
        for ticker in tickers:
            portfolio_df[f"{ticker}_weight"] = (
                portfolio_df[f"{ticker}_total_value"]
                / portfolio_df["total_value"]
            ).fillna(Defaults.DEFAULT_CASH_FLOW_WEIGHT)

        # Calculate daily cash flow adjusted returns
        period_cash_flow_weight = self.config.get(
            "advanced.performance.period_cash_flow_weight",
            Defaults.DEFAULT_CASH_FLOW_WEIGHT,
        )
        if isinstance(period_cash_flow_weight, dict):
            period_cash_flow_weight = period_cash_flow_weight.get(
                "value", Defaults.DEFAULT_CASH_FLOW_WEIGHT
            )
        if period_cash_flow_weight is None:
            period_cash_flow_weight = Defaults.DEFAULT_CASH_FLOW_WEIGHT
        period_cash_flow_weight = float(period_cash_flow_weight)

        for i, row in portfolio_df.iterrows():
            date = row["date"]
            if isinstance(date, str):
                date = pd.to_datetime(date)

            inv_cf, inc_cf = self._process_daily_transactions(date)
            total_cf = inv_cf["total"] + inc_cf["total"]

            portfolio_df.at[i, "investment_cash_flow"] = inv_cf["total"]
            portfolio_df.at[i, "income_cash_flow"] = inc_cf["total"]
            portfolio_df.at[i, "net_cash_flow"] = total_cf
            portfolio_df.at[i, "weighted_cash_flow"] = (
                total_cf * period_cash_flow_weight
            )

            # Store ALL account cash flows
            for account in accounts:
                total_account_cf = inv_cf.get(
                    account, Defaults.ZERO_CASH_FLOW
                ) + inc_cf.get(account, Defaults.ZERO_CASH_FLOW)

                portfolio_df.at[i, f"{account}_cash_flow"] = total_account_cf
                portfolio_df.at[i, f"{account}_weighted_cash_flow"] = (
                    total_account_cf * period_cash_flow_weight
                )

        # Calculate market value change (pure investment performance)
        for i in range(0, len(portfolio_df)):
            if i == 0:
                prev_value = Defaults.DEFAULT_ZERO
            else:
                prev_value = portfolio_df.iloc[i - 1]["total_value"]

            current_value = portfolio_df.iloc[i]["total_value"]
            external_cash_flow = portfolio_df.iloc[i]["investment_cash_flow"]

            # Dividends are considered internal cash flows and do not directly
            # affect the Modified Dietz calculation. Instead, the method
            # focuses on external cash flows, which are movements of value
            # into or out of the portfolio that are not related to investment
            # income.
            # Market value change = total change minus external cash flow impact
            market_change = current_value - prev_value - external_cash_flow

            if prev_value == 0:
                market_return = 0
            else:
                market_return = market_change / prev_value

            portfolio_df.iat[
                i, portfolio_df.columns.get_loc("market_value_change")
            ] = market_change

            portfolio_df.iat[
                i, portfolio_df.columns.get_loc("market_value_return")
            ] = market_return

        # For clarity, also store cash flow impact column
        self.logger.info("Calculating cash flow impact column")
        portfolio_df["cash_flow_impact"] = portfolio_df["net_cash_flow"]

        # Calculate Modified Dietz returns
        portfolio_df["portfolio_daily_return_mod_dietz"] = (
            calculate_modified_dietz_returns(portfolio_df)
        )

        # Calculate Time-Weighted Returns (TWR)
        portfolio_df["portfolio_daily_return_twr"] = (
            calculate_twr_daily_returns(portfolio_df)
        )

        # Calculate account-specific returns
        for account in accounts:
            account_total_col = f"{account}_total"
            if account_total_col in portfolio_df.columns:
                portfolio_df[f"{account}_mod_dietz_return"] = (
                    calculate_account_modified_dietz_returns(
                        portfolio_df, account
                    )
                )

                portfolio_df[f"{account}_mod_twr_return"] = (
                    calculate_account_twr_daily_returns(portfolio_df, account)
                )

        self.portfolio_history = portfolio_df

        self.logger.info(
            "✅ Portfolio history built: %d days", len(portfolio_df)
        )
        self.logger.info(
            "🏦 Tracking %d accounts: %s", len(accounts), ", ".join(accounts)
        )

        return portfolio_df

    def calculate_performance(self) -> "PerformanceResults":
        """
        Calculate comprehensive performance metrics for the portfolio and
        benchmark.

        Returns:
            PerformanceResults object containing all metrics and analysis
        """
        if self.portfolio_history.empty:
            portfolio_history = self.build_portfolio_history()
            if portfolio_history.empty:
                return PerformanceResults()

        self.logger.info("📊 Calculating performance metrics...")
        self.portfolio_history["date"] = pd.to_datetime(
            self.portfolio_history["date"], utc=True
        ).dt.normalize()

        portfolio_metrics = {}

        annual_trading_days = self.config.get(
            "settings.annual_trading_days",
            DateAndTimeConstants.DAYS_IN_TRADING_YEAR,
        )
        if isinstance(annual_trading_days, Dict):
            annual_trading_days = annual_trading_days.get(
                "value", DateAndTimeConstants.DAYS_IN_TRADING_YEAR
            )
        if annual_trading_days is None:
            annual_trading_days = DateAndTimeConstants.DAYS_IN_TRADING_YEAR
        annual_trading_days = int(annual_trading_days)

        annual_risk_free_rate = self.config.get(
            "settings.risk_free_rate", Defaults.DEFAULT_RISK_FREE_RATE
        )
        if isinstance(annual_risk_free_rate, Dict):
            annual_risk_free_rate = annual_risk_free_rate.get(
                "value", Defaults.DEFAULT_RISK_FREE_RATE
            )
        if annual_risk_free_rate is None:
            annual_risk_free_rate = Defaults.DEFAULT_RISK_FREE_RATE

        # Calculate portfolio performance metrics
        portfolio_metrics["mod_dietz"] = calculate_metrics(
            self.portfolio_history[
                "portfolio_daily_return_mod_dietz"
            ].dropna(),
            "Portfolio (Modified Dietz)",
            annual_trading_days,
            annual_risk_free_rate,
        )

        portfolio_metrics["twr"] = calculate_metrics(
            self.portfolio_history["portfolio_daily_return_twr"].dropna(),
            "Portfolio (TWR)",
            annual_trading_days,
            annual_risk_free_rate,
        )

        portfolio_metrics["irr"] = {
            "annualized_return": calculate_irr(
                self.portfolio_history, self.config
            )
        }

        # Calculate benchmark performance metrics
        benchmark_metrics = {}
        benchmark_returns = pd.Series(dtype=float)
        if not self.benchmark_data.empty:
            # Align benchmark data with portfolio dates
            aligned_benchmark_df = self._align_benchmark_returns()

            self.portfolio_history = pd.merge(
                self.portfolio_history,
                aligned_benchmark_df,
                on="date",
                how="left",
            )

            benchmark_returns = self.portfolio_history[
                "benchmark_returns"
            ].copy()
            benchmark_metrics = calculate_metrics(
                benchmark_returns[1:],
                "Benchmark",
                annual_trading_days,
                annual_risk_free_rate,
            )

        # Calculate relative performance metrics
        portfolio_returns = self.portfolio_history[
            "portfolio_daily_return_twr"
        ].copy()

        portfolio_returns = pd.Series(
            portfolio_returns.values,
            index=pd.to_datetime(self.portfolio_history["date"]),
        )

        if benchmark_returns.any():
            benchmark_returns = pd.Series(
                benchmark_returns.values,
                index=pd.to_datetime(self.portfolio_history["date"]),
            )

        relative_metrics = calculate_relative_metrics(
            portfolio_returns.dropna()[1:],
            (
                benchmark_returns.dropna()[1:]
                if benchmark_returns.any()
                else None
            ),
            annual_trading_days,
            annual_risk_free_rate,
        )

        # Create results object
        results = PerformanceResults(
            portfolio_metrics=portfolio_metrics,
            benchmark_metrics=benchmark_metrics,
            relative_metrics=relative_metrics,
            portfolio_history=self.portfolio_history,
            config=self.config,
        )

        self.performance_results = results

        self.logger.info("✅ Performance analysis complete!")
        return results

    def _align_benchmark_returns(self) -> pd.DataFrame:
        """Align benchmark returns with portfolio dates."""
        portfolio_dates = pd.to_datetime(
            self.portfolio_history["date"], utc=True
        ).dt.normalize()

        # Convert benchmark data to returns
        benchmark_df = self.benchmark_data.copy()
        benchmark_df["date"] = pd.to_datetime(
            benchmark_df["date"], utc=True
        ).dt.normalize()

        # Ensure ascending date order
        benchmark_df = benchmark_df.sort_values("date").reset_index(drop=True)

        # Calculate period on period returns
        # Use adjusted close prices for benchmark returns
        benchmark_df["benchmark_returns"] = benchmark_df[
            "adj_close"
        ].pct_change()

        # Reindex benchmark returns to match portfolio dates, forward-filling for missing days
        benchmark_df.set_index("date", inplace=True)
        aligned_returns = benchmark_df["benchmark_returns"].reindex(
            portfolio_dates, method="ffill"
        )

        # The first return is NaN after reindexing, fill with 0
        aligned_returns.iloc[0] = Defaults.ZERO_RETURN

        aligned_df = aligned_returns.reset_index()
        aligned_df.rename(columns={"index": "date"}, inplace=True)

        return aligned_df
