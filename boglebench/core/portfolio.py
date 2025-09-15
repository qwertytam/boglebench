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
from typing import Any, Dict, Optional, Tuple
from zoneinfo import ZoneInfo  # pylint: disable=wrong-import-order

import pandas as pd
import pandas_market_calendars as mcal  # type: ignore

from ..core.constants import (
    ConversionFactors,
    DateAndTimeConstants,
    Defaults,
    TransactionTypes,
)
from ..core.dividend_validator import (
    DividendValidator,
    identify_any_dividend_transactions,
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
from ..utils.tools import is_tz_aware, to_tzts, to_tzts_scaler
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

        self.start_date: Optional[pd.Timestamp] = None
        self.end_date: Optional[pd.Timestamp] = None

        self.transactions = pd.DataFrame()
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.portfolio_history = pd.DataFrame()
        self.benchmark_data = pd.DataFrame()
        self.performance_results = PerformanceResults()

        # Set up market data provider
        self.api_key = self.config.get(
            "api.alpha_vantage_key", Defaults.DEFAULT_API_KEY
        )
        if isinstance(self.api_key, dict):
            self.api_key = self.api_key.get("value", Defaults.DEFAULT_API_KEY)
        if self.api_key is None or str(self.api_key).strip() == "":
            self.api_key = Defaults.DEFAULT_API_KEY

        cache_enabled = self.config.get("settings.cache_market_data", True)
        if isinstance(cache_enabled, dict):
            cache_enabled = cache_enabled.get("value", True)
        if cache_enabled is None:
            cache_enabled = True

        cache_dir = self.config.get_market_data_path()

        self.market_data_provider = MarketDataProvider(
            api_key=self.api_key,
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
            force_cache_refresh=False,
        )

        # Suppress warnings for cleaner output
        warnings.filterwarnings("ignore", category=FutureWarning)
        self.logger.info("BogleBench analyzer initialized")

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
            self.transactions["date"].min(),
            self.transactions["date"].max(),
        )
        self.logger.info(
            "🏦 Accounts: %s", ", ".join(self.transactions["account"].unique())
        )
        total_invested = self.transactions[
            self.transactions["total_value"] > 0
        ]["total_value"].sum()
        msg = f"💰 Total invested: ${total_invested:,.2f}"
        self.logger.info(msg)

        return self.transactions

    def _set_start_date(self, first_transaction_date: DateLike) -> bool:
        """Set the analysis start date.

        Rules:
        1. If 'analysis.start_date' is set in config, use that.
        2. Otherwise, use the date of the first transaction as provided.

        Args:
            first_transaction_date: Date of the first transaction.

        Returns:
            bool: True if start date is has been set successfully as a
            timezone-aware timestamp, False otherwise.
        """
        first_transaction_tstz = to_tzts(
            first_transaction_date, tz=DateAndTimeConstants.TZ_UTC.value
        )
        if isinstance(first_transaction_tstz, pd.Series):
            if not first_transaction_tstz.empty:
                self.logger.debug(
                    "first_transaction_tstz is a Series, taking first value"
                )
                first_transaction_tstz = to_tzts_scaler(
                    first_transaction_tstz.iloc[0],
                    tz=DateAndTimeConstants.TZ_UTC.value,
                )
            else:
                raise ValueError("first_transaction_date Series is empty")

        # Ensure the config value is not a dict before passing to pd.Timestamp
        config_start_date = self.config.get("analysis.start_date", None)
        if isinstance(config_start_date, dict):
            config_start_date = config_start_date.get("value", None)

        if config_start_date is not None:
            config_start_date = to_tzts_scaler(
                config_start_date, tz=DateAndTimeConstants.TZ_UTC.value
            )
            self.start_date = config_start_date
            self.logger.info(
                "Using configured start date: %s", self.start_date
            )
        else:
            self.start_date = first_transaction_tstz
            self.logger.info(
                "Using first transaction date as start date: %s",
                self.start_date,
            )

        if self.start_date is None:
            self.logger.error("Failed to set start date")
            return False
        return is_tz_aware(self.start_date)

    def _get_start_date(self) -> pd.Timestamp:
        """Get the analysis start date, ensuring it's set."""
        if self.start_date is None:
            raise ValueError("Start date is not set")
        return self.start_date

    def _set_end_date(self) -> bool:
        """Set the analysis end date.

        Rules:
        1. If 'analysis.end_date' is set in config, use that.
        2. Otherwise, if the market is currently open, use the last closed market day.
        3. Otherwise, use the current datetime as the end date.
        Raises:
            ValueError: If no valid end date can be determined.

        Returns:
            bool: True if end date is has been set successfully as a
            timezone-aware timestamp, False otherwise.
        """
        last_market_close_date = None
        if self._is_market_currently_open():
            self.logger.info("Market is currently open")
            last_market_close_date = self._get_last_closed_market_day()
        else:
            self.logger.info("Market is currently closed")

            # Ensure last_market_close_date is a scalar Timestamp,
            # not a Series or None
            machine_tz = ZoneInfo(str(DateAndTimeConstants.TZ_UTC.value))
            dt_now = to_tzts(datetime.now(tz=machine_tz))
            if isinstance(dt_now, pd.Series):
                if not dt_now.empty:
                    last_market_close_date = to_tzts_scaler(
                        dt_now.iloc[0], tz=machine_tz
                    )
                else:
                    raise ValueError(
                        "Received empty Series for datetime.now()"
                    )
            else:
                last_market_close_date = dt_now

        # Ensure the config value is not a dict before passing to pd.Timestamp
        config_end_date = self.config.get(
            "analysis.end_date", last_market_close_date
        )
        if isinstance(config_end_date, dict):
            config_end_date = config_end_date.get(
                "value", last_market_close_date
            )

        if config_end_date is not None:
            self.end_date = to_tzts_scaler(
                config_end_date, tz=DateAndTimeConstants.TZ_UTC.value
            )
            self.logger.info("Using configured end date: %s", config_end_date)
        elif last_market_close_date is not None:
            self.end_date = to_tzts_scaler(
                last_market_close_date, tz=DateAndTimeConstants.TZ_UTC.value
            )
            self.logger.info(
                "Using last market close date as end date: %s",
                last_market_close_date,
            )
        else:
            raise ValueError("No valid end date provided or found.")

        if self.end_date is None:
            self.logger.error("Failed to set end date")
            return False
        return is_tz_aware(self.end_date)

    def _get_end_date(self) -> pd.Timestamp:
        """Get the analysis end date, ensuring it's set."""
        if self.end_date is None:
            raise ValueError("End date is not set")
        return self.end_date

    def _is_market_currently_open(self) -> bool:
        """Check if the market is currently open."""
        nyse = mcal.get_calendar("NYSE")
        now = datetime.now(tz=ZoneInfo(DateAndTimeConstants.TZ_UTC))
        schedule = nyse.schedule(start_date=now.date(), end_date=now.date())
        if schedule.empty:
            self.logger.debug("Market is closed today (holiday or weekend)")
            return False

        market_open = schedule.iloc[0]["market_open"].to_pydatetime()
        market_close = schedule.iloc[0]["market_close"].to_pydatetime()
        self.logger.debug(
            "Market hours today: %s to %s", market_open, market_close
        )
        self.logger.debug(
            "Going to compare market open %s %s, now %s %s, close %s %s",
            market_open,
            type(market_open),
            now,
            type(now),
            market_close,
            type(market_close),
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
        self.logger.debug("Transactions:\n%s", day_transactions)

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

            self.logger.debug(
                "  Is ttype %s in DIVIDEND? %s",
                ttype,
                TransactionTypes.is_dividend(ttype),
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

        # for ticker in tickers:
        #     try:
        #         auto_calc_div_per_share = self.config.get(
        #             "dividend.auto_calculate_div_per_share", True
        #         )
        #         if not isinstance(auto_calc_div_per_share, bool):
        #             auto_calc_div_per_share = True

        #         warn_missing_dividends = self.config.get(
        #             "dividend.warn_missing_dividends", True
        #         )
        #         if not isinstance(warn_missing_dividends, bool):
        #             warn_missing_dividends = True

        #         messages = self.compare_user_dividends_to_market(
        #             ticker,
        #             auto_calculate_div_per_share=auto_calc_div_per_share,
        #             warn_missing_dividends=warn_missing_dividends,
        #         )

        #         for msg in messages:
        #             if (
        #                 "mismatch" in msg.lower()
        #                 or "warning" in msg.lower()
        #             ):
        #                 self.logger.warning("⚠️  %s", msg)
        #             else:
        #                 self.logger.info("ℹ️  %s", msg)

        #     except ValueError as e:
        #         self.logger.error(
        #             "Could not validate dividends for %s: %s", ticker, e
        #         )

        # Initialize portfolio tracking by account and ticker
        portfolio_data = []

        # Track holdings by account and ticker: {account: {ticker: shares}}
        current_holdings = {
            account: {ticker: 0.0 for ticker in tickers}
            for account in accounts
        }

        min_transaction_date = self.transactions["date"].min().date()
        max_transaction_date = self.transactions["date"].max().date()
        self._set_start_date(min_transaction_date)

        if self._get_start_date().date() > max_transaction_date:
            return pd.DataFrame()  # No data to process

        self._set_end_date()

        if self._get_start_date() > self._get_end_date():
            raise ValueError(
                "Start date must be on or before end date: "
                f"{self._get_start_date()} > {self._get_end_date()}"
            )

        # Default is all day between and including start and end dates that are
        # trading days.
        # However, will also include any transaction dates that fall outside
        # this range to ensure all transactions are processed.
        # Get actual NYSE trading days (excludes weekends AND holidays)

        # 1. Get all trading days in the period
        self.logger.debug(
            "Getting NYSE trading days from %s to %s",
            self._get_start_date(),
            self._get_end_date(),
        )
        nyse = mcal.get_calendar("NYSE")
        trading_days = nyse.schedule(
            start_date=self._get_start_date(),
            end_date=self._get_end_date().date(),
        )

        self.logger.debug("trading_days:\n%s", trading_days)
        self.logger.debug("type of trading_days:\n%s", trading_days.dtypes)
        self.logger.debug(
            "type of trading_days index:\n%s", trading_days.index.dtype
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
            self.transactions["date"].dt.date >= self._get_start_date().date()
        ) & (self.transactions["date"].dt.date <= self._get_end_date().date())

        transaction_dates = list(
            set(
                pd.to_datetime(
                    self.transactions.loc[
                        transaction_dates_mask, "date"
                    ].dt.normalize()
                )
            )
        )

        self.logger.debug("Trading days sample:\n%s", trading_days.head(10))
        self.logger.debug(
            "Transaction dates sample:\n%s", transaction_dates[:10]
        )

        # 3. Find transactions dates that are outside of trading days
        self.logger.debug("Checking for transactions on non-trading days")
        non_trading_transaction_dates = set(transaction_dates) - set(
            trading_dates
        )

        if non_trading_transaction_dates:
            self.logger.warning(
                "⚠️  Warning: %d transactions on non-trading days:\n%s",
                len(non_trading_transaction_dates),
                "\n".join(str(date) for date in non_trading_transaction_dates),
            )

        self.logger.debug("trading_dates:\n%s", trading_dates)
        self.logger.debug(
            "non_trading_transaction_dates:\n%s", non_trading_transaction_dates
        )

        all_dates_to_process = sorted(
            list(trading_dates.union(list(non_trading_transaction_dates)))
        )

        date_range = pd.to_datetime(all_dates_to_process, utc=True)

        self.logger.info(
            "📅 Processing %d trading days from %s to %s",
            len(date_range),
            pd.Timestamp(date_range[0]),
            pd.Timestamp(date_range[-1]),
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
        self.logger.debug(
            "Force refresh of market data: %s", force_refresh_market_data
        )
        if (
            force_refresh_market_data
            or self.market_data is None
            or not self.market_data
        ):
            self.market_data = self.market_data_provider.get_market_data(
                tickers=tickers,
                start_date_str=self._get_start_date().strftime("%Y-%m-%d"),
                end_date_str=self._get_end_date().strftime("%Y-%m-%d"),
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
                            )

                            if not self.transactions.loc[mask].empty:
                                shares = self.transactions.loc[
                                    mask, "quantity"
                                ]
                                if isinstance(shares, pd.Series):
                                    shares_sum = shares.sum()
                                else:
                                    shares_sum = (
                                        shares
                                        if isinstance(shares, (int, float))
                                        else 0
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

                                new_total_value = (
                                    shares_sum * new_div_per_share
                                )
                                self.logger.info(
                                    "Updating %s dividend on %s: "
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
                                    "No matching transaction found to"
                                    + " update for %s on %s",
                                    ticker,
                                    date.date(),
                                )

        for date in date_range:
            # Process any transactions on this date
            self.logger.debug(
                "Processing transaction dates:\n%s\nagainst date %s",
                self.transactions["date"].dt.date,
                date.date(),
            )
            self.logger.debug(
                "These dates have type %s and %s",
                type(self.transactions["date"].dt.date),
                type(date.date()),
            )

            day_transactions = self.transactions[
                self.transactions["date"].dt.date == date.date()
            ]

            self.logger.debug(
                "Processing %s transactions for %s:\n%s",
                len(day_transactions),
                date.date(),
                day_transactions,
            )

            for _, transaction in day_transactions.iterrows():
                account = transaction["account"]
                ticker = transaction["ticker"]
                ttype = transaction["transaction_type"]

                if TransactionTypes.is_quantity_changing(ttype):
                    shares = transaction["quantity"]

                    self.logger.debug("Have %s shares for %s", shares, ttype)

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
            self.logger.debug("Calculating portfolio value for %s", date)
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

                    self.logger.debug(
                        "Calculated price for %s on %s: $%.2f",
                        ticker,
                        date,
                        price,
                    )
                    position_value = shares * price
                    account_value += position_value

                    self.logger.debug(
                        "Storing shares %.4f of %s", shares, ticker
                    )
                    self.logger.debug(
                        "Storing price $%.2f for %s on %s", price, ticker, date
                    )
                    self.logger.debug(
                        "Storing position value $%.2f for %s on %s",
                        position_value,
                        ticker,
                        date,
                    )

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

                self.logger.debug(
                    "Storing total shares %.4f of %s", total_shares, ticker
                )
                self.logger.debug(
                    "Storing total value $%.2f of %s",
                    total_shares * price,
                    ticker,
                )

                day_data[f"{ticker}_total_shares"] = total_shares
                day_data[f"{ticker}_total_value"] = total_shares * price

            portfolio_data.append(day_data)

        # Convert to DataFrame
        self.logger.debug("Converting portfolio data to DataFrame")
        portfolio_df = pd.DataFrame(portfolio_data)

        self.logger.debug(
            "Portfolio DataFrame sample:\n%s", portfolio_df.head(10)
        )
        self.logger.debug(
            "Portfolio dates sample:\n%s", portfolio_df["date"].head(10)
        )
        # Ensure ascending date order
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

        # Calculate daily cash flow adjust returns
        # Calculate cash flows for each day using helper function
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

            portfolio_df.at[
                i, portfolio_df.columns.get_loc("market_value_change")
            ] = market_change

            portfolio_df.at[
                i, portfolio_df.columns.get_loc("market_value_return")
            ] = market_return

        # For clarity, also store cash flow impact column
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

    # def fetch_dividend_data(self, ticker: str) -> pd.DataFrame:
    #     """
    #     Extracts dividend data for a given ticker from already-fetched
    #     self.market_data.
    #     Returns a DataFrame with columns: 'date', 'dividend'
    #     """
    #     if ticker not in self.market_data:
    #         raise ValueError(f"Market data for ticker {ticker} not found. ")

    #     df = self.market_data[ticker]
    #     # Only consider days where a dividend was paid
    #     self.logger.debug("Extracting dividend data for %s", ticker)
    #     dividend_df = df[df["dividend"] > 0][["date", "dividend"]].copy()
    #     dividend_df["value_per_share"] = dividend_df["dividend"]
    #     dividend_df["div_type"] = DividendTypes.CASH

    #     # Normalize date for merging
    #     dividend_df["date"] = pd.to_datetime(
    #         dividend_df["date"], utc=True
    #     ).dt.normalize()
    #     dividend_df = dividend_df.rename(columns={"date": "div_pay_date"})

    #     dividend_df["div_ex_date"] = pd.NaT
    #     dividend_df["div_record_date"] = pd.NaT

    #     return_columns = [
    #         "div_pay_date",
    #         "value_per_share",
    #         "div_type",
    #         "div_ex_date",
    #         "div_record_date",
    #     ]

    #     return dividend_df[return_columns]

    # def _get_shares_held_on_date(
    #     self, ticker: str, account: str, target_date: pd.Timestamp
    # ) -> float:
    #     """
    #     Calculate shares held for a specific ticker and account on a given date.
    #     Looks at all BUY/SELL/DIVIDEND_REINVEST transactions up to target_date.
    #     Note: DIVIDEND_REINVEST increases shares held only up to one day before
    #     the dividend pay date.
    #     """
    #     if self.transactions.empty:
    #         raise ValueError("Must load transactions first")
    #     if ticker not in self.transactions["ticker"].values:
    #         raise ValueError(f"No transactions found for ticker {ticker}")
    #     if account not in self.transactions["account"].values:
    #         raise ValueError(f"No transactions found for account {account}")
    #     if isinstance(target_date, str):
    #         target_date = pd.to_datetime(target_date)

    #     # Up to and including target_date for BUY/SELL
    #     relevant_transactions_buy_sell = self.transactions[
    #         (self.transactions["ticker"] == ticker)
    #         & (self.transactions["account"] == account)
    #         & (self.transactions["date"] <= target_date)
    #         & (self.transactions["transaction_type"].isin(["BUY", "SELL"]))
    #     ].copy()

    #     # Only before target_date for DIVIDEND_REINVEST
    #     relevant_transactions_div_reinvest = self.transactions[
    #         (self.transactions["ticker"] == ticker)
    #         & (self.transactions["account"] == account)
    #         & (self.transactions["date"] < target_date)
    #         & (self.transactions["transaction_type"] == "DIVIDEND_REINVEST")
    #     ].copy()

    #     relevant_transactions = pd.concat(
    #         [
    #             relevant_transactions_buy_sell,
    #             relevant_transactions_div_reinvest,
    #         ]
    #     )

    #     if relevant_transactions.empty:
    #         self.logger.warning(
    #             "⚠️  No transactions found for %s in %s before %s",
    #             ticker,
    #             account,
    #             target_date.date(),
    #         )
    #         return Defaults.DEFAULT_ZERO

    #     # Sum all share changes (SELL transactions already have negative shares)
    #     total_shares = relevant_transactions["quantity"].sum()
    #     # Ensure non-negative
    #     result = max(0.0, total_shares)

    #     self.logger.debug(
    #         "Shares held for %s in %s on %s: %s",
    #         ticker,
    #         account,
    #         target_date.date(),
    #         result,
    #     )
    #     return result

    # def _perform_dividend_comparison(
    #     self,
    #     ticker: str,
    #     user_dividends: pd.DataFrame,
    #     error_margin: float,
    #     warn_missing_dividends: bool = True,
    # ) -> List[str]:
    #     """Perform the actual dividend comparison with flexible matching."""

    #     messages: list[str] = []
    #     if user_dividends.empty:
    #         return messages

    #     # Group by date and sum amounts in case both cash+reinvest are logged
    #     # separately
    #     self.logger.debug("Grouping user dividends for %s by pay date", ticker)
    #     self.logger.debug(
    #         "User dividends before grouping:\n%s", user_dividends
    #     )
    #     self.logger.debug(
    #         "User dividends columns: %s", user_dividends.columns.tolist()
    #     )
    #     user_divs_grouped = user_dividends.groupby(
    #         "div_pay_date", as_index=False
    #     ).apply(aggregate_dividends)

    #     self.logger.debug(
    #         "User dividends grouped by pay date:\n%s", user_divs_grouped
    #     )
    #     user_divs_grouped["div_pay_date"] = user_divs_grouped[
    #         "div_pay_date"
    #     ].dt.normalize()

    #     self.logger.debug("Fetching market dividends for %s", ticker)
    #     market_dividends = self.fetch_dividend_data(ticker)

    #     self.logger.debug(
    #         "Market dividends fetched for %s:\n%s", ticker, market_dividends
    #     )
    #     market_dividends["div_pay_date"] = market_dividends[
    #         "div_pay_date"
    #     ].dt.normalize()

    #     self.logger.debug(
    #         "Comparing %d user dividends to %d market dividends for %s",
    #         len(user_divs_grouped),
    #         len(market_dividends),
    #         ticker,
    #     )
    #     self.logger.debug("User dividends:\n%s", user_divs_grouped)
    #     self.logger.debug("Market dividends:\n%s", market_dividends)

    #     # Use outer join to catch missing dividends if warn_missing_dividends is True
    #     how_merge: Literal["left", "outer"] = (
    #         "outer" if warn_missing_dividends else "left"
    #     )
    #     self.logger.debug(
    #         "Merging user and market dividends for %s with '%s' join with date columns %s and %s",
    #         ticker,
    #         how_merge,
    #         user_divs_grouped["div_pay_date"].head(),
    #         market_dividends["div_pay_date"].head(),
    #     )
    #     merged = pd.merge(
    #         user_divs_grouped,
    #         market_dividends,
    #         on="div_pay_date",
    #         how=how_merge,
    #         suffixes=("_user", "_mkt"),
    #     )

    #     self.logger.debug("user_divs head:\n%s", user_divs_grouped.head())
    #     self.logger.debug("mkt_divs head:\n%s", market_dividends.head())
    #     self.logger.debug(
    #         "Merged user and market dividends for %s:\n%s", ticker, merged
    #     )

    #     # Check for missing user dividends
    #     if warn_missing_dividends:
    #         missing_user_divs = merged[
    #             (merged["total_value"].isna())
    #             & (merged["value_per_share_mkt"].notna())
    #         ]

    #         for _, row in missing_user_divs.iterrows():
    #             msg = (
    #                 f"Missing user dividend for {ticker} "
    #                 f"on {row['div_pay_date'].date()}: "
    #                 f"Market shows ${row['value_per_share_mkt']:.4f} per share"
    #             )
    #             messages.append(msg)

    #     # Check for mismatches in recorded dividends
    #     recorded_dividends = merged[
    #         (merged["total_value"].notna())
    #         & (merged["value_per_share_mkt"].notna())
    #     ]

    #     for _, row in recorded_dividends.iterrows():
    #         issues = []

    #         # Compare per-share amounts if user div_per_share is available
    #         if (
    #             pd.notna(row["value_per_share_user"])
    #             and row["value_per_share_user"] > 0
    #         ):
    #             self.logger.debug("Found valid dividend row to compare")
    #             self.logger.debug("Row data: %s", row.to_dict())
    #             if (
    #                 abs(
    #                     row["value_per_share_user"]
    #                     - row["value_per_share_mkt"]
    #                 )
    #                 > error_margin
    #             ):
    #                 issues.append(
    #                     f"per-share mismatch (user: "
    #                     f"${row['value_per_share_user']:.4f}, "
    #                     f"market: ${row['value_per_share_mkt']:.4f})"
    #                 )

    #         # Compare div_type if provided
    #         if pd.notna(row["div_type_user"]) and row["div_type_user"] != "":
    #             if row["div_type_user"] != row["div_type_mkt"]:
    #                 issues.append(
    #                     f"type mismatch (user: {row['div_type_user']}, "
    #                     f"market: {row['div_type_mkt']})"
    #                 )

    #         if issues:
    #             messages.append(
    #                 f"Dividend mismatch for {ticker} on "
    #                 f"{row['div_pay_date'].date()}: {'; '.join(issues)} "
    #                 f"(user total: ${row['total_value']:.2f})"
    #             )

    #     return messages

    # def validate_dividend_data(
    #     self,
    #     tickers: Optional[list] = None,
    #     auto_calculate_div_per_share: bool = True,
    #     warn_missing_dividends: bool = True,
    # ) -> None:
    #     """
    #     Validate dividend data for specified tickers or all tickers in portfolio.

    #     Args:
    #         tickers: List of tickers to validate. If None, validates all tickers.
    #         auto_calculate_div_per_share: Whether to auto-calculate missing div_per_share
    #         warn_missing_dividends: Whether to warn about missing dividend entries
    #     """
    #     if self.transactions.empty:
    #         raise ValueError("Must load transactions first")

    #     if tickers is None:
    #         tickers = self.transactions["ticker"].unique().tolist()

    #         if tickers is None or len(tickers) == 0:
    #             raise ValueError(
    #                 "No tickers found in transactions to validate."
    #             )

    #     self.logger.info(
    #         "🔍 Validating dividend data for %d tickers...", len(tickers or [])
    #     )

    #     for ticker in tickers:
    #         try:
    #             self.compare_user_dividends_to_market(
    #                 ticker,
    #                 auto_calculate_div_per_share=auto_calculate_div_per_share,
    #                 warn_missing_dividends=warn_missing_dividends,
    #             )
    #         except ValueError as e:
    #             self.logger.error(
    #                 "Error validating dividends for %s: %s", ticker, e
    #             )

    #     self.logger.info("✅ Dividend validation complete")

    # def _find_dividend_transactions(self, series: pd.Series) -> pd.Series:
    #     """Helper function to identify dividend transactions in a Series."""

    #     valid_types = TransactionTypes.all_dividend_types()
    #     return series.isin(valid_types)

    # def compare_user_dividends_to_market(
    #     self,
    #     ticker: str,
    #     error_margin: float = 0.01,
    #     auto_calculate_div_per_share: bool = True,
    #     warn_missing_dividends: bool = True,
    # ) -> List[str]:
    #     """
    #     Compare user-provided dividend and dividend reinvestment transactions
    #     with market dividend data. Prints a warning if values differ by more
    #     than the error margin. Sums amounts for the same date.

    #     Args:
    #         ticker: Stock ticker symbol
    #         error_margin: Acceptable difference for matching dividends
    #         auto_calculate_div_per_share: If True, calculate missing
    #             div_per_share from holdings
    #         warn_missing_dividends: If True, warn about market dividends not
    #             in user data
    #         default_div_type: Default dividend type when not specified by user
    #     Returns:
    #         A list of strings, where each string is a message about a
    #         mismatch or calculation.
    #     """

    #     if self.transactions.empty:
    #         raise ValueError("No transactions loaded to compare against.")

    #     user_dividends = self.transactions[
    #         (self.transactions["ticker"] == ticker)
    #         & (
    #             self._find_dividend_transactions(
    #                 self.transactions["transaction_type"]
    #             )
    #         )
    #     ].copy()

    #     self.logger.debug(
    #         "Comparing user dividends for %s:\n%s", ticker, user_dividends
    #     )

    #     if user_dividends.empty:
    #         self.logger.debug(
    #             "No user dividends found for %s to compare.", ticker
    #         )
    #         return ["No user dividends found to compare."]

    #     if "div_type" not in user_dividends.columns:
    #         user_dividends["div_type"] = DividendTypes.CASH

    #     if "div_pay_date" not in user_dividends.columns:
    #         user_dividends["div_pay_date"] = user_dividends["date"]

    #     if "value_per_share" not in user_dividends.columns:
    #         user_dividends["value_per_share"] = Defaults.ZERO_DIVIDEND

    #     # Process dividends per account first ---
    #     all_enhanced_dividends = []
    #     all_validation_messages = []

    #     # Group by account AND date to handle each dividend event individually
    #     grouped_by_account_and_date = user_dividends.groupby(
    #         ["account", "date"]
    #     )

    #     for (account, date), group in grouped_by_account_and_date:
    #         self.logger.debug(
    #             "Validating dividend for %s in account %s on %s",
    #             ticker,
    #             account,
    #             date.date(),
    #         )
    #         enhanced_group, validation_messages = (
    #             self._validate_and_enhance_dividend_data(
    #                 group,
    #                 ticker,
    #                 account,
    #                 auto_calculate_div_per_share,
    #             )
    #         )
    #         all_enhanced_dividends.append(enhanced_group)
    #         all_validation_messages.extend(validation_messages)

    #     if not all_enhanced_dividends:
    #         return all_validation_messages

    #     final_user_dividends = pd.concat(all_enhanced_dividends)
    #     final_user_dividends["div_pay_date"] = pd.to_datetime(
    #         final_user_dividends["div_pay_date"], utc=True
    #     )

    #     self.logger.debug(
    #         "final_user_dividends:\n%s", final_user_dividends.head()
    #     )

    #     # Perform the comparison
    #     comparison_messages = self._perform_dividend_comparison(
    #         ticker, final_user_dividends, error_margin, warn_missing_dividends
    #     )

    #     self.logger.debug("Completed dividend comparison for %s", ticker)
    #     self.logger.debug(comparison_messages)

    #     return all_validation_messages + comparison_messages

    # def _validate_and_enhance_dividend_data(
    #     self,
    #     dividend_group: pd.DataFrame,
    #     ticker: str,
    #     account: str,
    #     auto_calculate_div_per_share: bool = True,
    # ) -> Tuple[pd.DataFrame, List[str]]:
    #     """
    #     Validate and enhance a specific group of user dividend transactions
    #     (for a single account and date).
    #     """
    #     enhanced_group = dividend_group.copy()
    #     info_messages = []

    #     # Consolidate the group first (e.g., partial reinvest)
    #     total_amount = enhanced_group["total_value"].sum()
    #     div_date = enhanced_group["date"].iloc[0]
    #     pay_date = (
    #         enhanced_group["div_pay_date"].dropna().iloc[0]
    #         if not enhanced_group["div_pay_date"].dropna().empty
    #         else div_date
    #     )

    #     # Calculate div_per_share if needed
    #     # Use the first non-zero div_per_share provided, otherwise calculate
    #     provided_dps = enhanced_group[enhanced_group["value_per_share"] > 0][
    #         "value_per_share"
    #     ]
    #     final_dps = 0.0

    #     self.logger.debug(
    #         "Validating dividend for %s in %s on %s and auto calculate %s",
    #         ticker,
    #         account,
    #         pay_date.date(),
    #         auto_calculate_div_per_share,
    #     )

    #     if not provided_dps.empty:
    #         final_dps = provided_dps.iloc[0]
    #     elif auto_calculate_div_per_share:
    #         shares_held = self._get_shares_held_on_date(
    #             ticker, account, pay_date
    #         )
    #         if shares_held > 0 and total_amount != 0:
    #             final_dps = total_amount / shares_held
    #             info_messages.append(
    #                 f"Calculated div/share for {ticker} in {account} "
    #                 f"on {pay_date.date()} "
    #                 f"as ${np.abs(final_dps):.4f} (${np.abs(total_amount):.2f} / "
    #                 f"{shares_held:.4f} shares)"
    #             )
    #         else:
    #             info_messages.append(
    #                 f"Warning: Could not calculate div/share for {ticker} in "
    #                 f"{account} on {pay_date.date()} "
    #                 f"(shares={shares_held}, amount=${total_amount})"
    #             )

    #         # Apply the final calculated/validated DPS to all rows in the group
    #         enhanced_group["value_per_share"] = final_dps
    #         enhanced_group["div_pay_date"] = pd.to_datetime(pay_date)

    #     return enhanced_group, info_messages

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
        self.logger.debug(
            "Portfolio history columns:\n%s", self.portfolio_history.columns
        )
        self.portfolio_history["date"] = pd.to_datetime(
            self.portfolio_history["date"], utc=True
        ).dt.normalize()

        portfolio_metrics = {}

        self.logger.debug(
            "Portfolio history sample:\n%s", self.portfolio_history.head(n=10)
        )

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

            self.logger.debug(
                "Aligned benchmark %s returns:\n%s",
                len(benchmark_returns),
                benchmark_returns.head(n=10)
                * ConversionFactors.DECIMAL_TO_PERCENT,
            )

            self.logger.debug(
                "portfolio_history before merging benchmark:\n%s",
                self.portfolio_history.head(n=10),
            )

            self.portfolio_history = pd.merge(
                self.portfolio_history,
                aligned_benchmark_df,
                on="date",
                how="left",
            )

            self.logger.debug(
                "portfolio_history after merging benchmark:\n%s",
                self.portfolio_history.head(n=10),
            )

            benchmark_returns = self.portfolio_history[
                "Benchmark_Returns"
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

        self.logger.debug(
            "Portfolio returns sample:\n%s",
            portfolio_returns.head(n=10)
            * ConversionFactors.DECIMAL_TO_PERCENT,
        )

        portfolio_returns = pd.Series(
            portfolio_returns.values,
            index=pd.to_datetime(self.portfolio_history["date"]),
        )

        self.logger.debug(
            "Portfolio returns after index update:\n%s",
            portfolio_returns.head(n=10)
            * ConversionFactors.DECIMAL_TO_PERCENT,
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
        benchmark_df["Benchmark_Returns"] = benchmark_df[
            "adj_close"
        ].pct_change()

        # Reindex benchmark returns to match portfolio dates, forward-filling for missing days
        benchmark_df.set_index("date", inplace=True)
        aligned_returns = benchmark_df["Benchmark_Returns"].reindex(
            portfolio_dates, method="ffill"
        )

        # The first return is NaN after reindexing, fill with 0
        aligned_returns.iloc[0] = Defaults.ZERO_RETURN

        aligned_df = aligned_returns.reset_index()
        aligned_df.rename(columns={"index": "date"}, inplace=True)

        return aligned_df
