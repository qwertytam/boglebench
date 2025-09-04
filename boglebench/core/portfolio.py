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

import re
import warnings
from datetime import datetime, timedelta, tzinfo
from pathlib import Path
from typing import Dict, Optional, Union
from zoneinfo import ZoneInfo  # pylint: disable=wrong-import-order

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal  # type: ignore
from alpha_vantage.timeseries import TimeSeries  # type: ignore

from ..utils.config import ConfigManager
from ..utils.logging_config import get_logger, setup_logging
from ..utils.timing import timed_operation
from ..utils.tools import cagr, ensure_timestamp, to_tz_mixed
from ..utils.workspace import WorkspaceContext

TO_PERCENT = 100
DEFAULT_TRADING_DAYS = 252
DEFAULT_VALUE = 0.0
DEFAULT_RETURN = DEFAULT_VALUE
DEFAULT_ASSET_VALUE = DEFAULT_VALUE
DEFAULT_PRICE = DEFAULT_VALUE
DEFAULT_CASH_FLOW = DEFAULT_VALUE
DEFAULT_WEIGHT = DEFAULT_VALUE
DEFAULT_RISK_FREE_RATE = 0.02
DEFAULT_LOOK_FORWARD_PRICE_DATA = 10  # days


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
        print("DEBUG: Config loaded from:", self.config.config_path)
        print("DEBUG: BogleBench Setting up logging...")
        setup_logging()  # Initialize after workspace context is set
        self.logger = get_logger("core.portfolio")

        self.config = ConfigManager(config_path)
        self.transactions = None
        self.market_data = {}
        self.portfolio_history = None
        self.benchmark_data = None
        self.performance_results = None

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
            file_path = self.config.get_transactions_path()

        if not Path(file_path).exists():
            raise FileNotFoundError(f"Transaction file not found: {file_path}")

        print(f"📄 Loading transactions from: {file_path}")

        # Load CSV with flexible parsing
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")

        # Validate required columns
        required_columns = [
            "date",
            "ticker",
            "transaction_type",
            "shares",
            "price_per_share",
        ]

        missing_columns = [
            col for col in required_columns if col not in df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Clean and validate data
        df = self._clean_transaction_data(df)

        # Store processed transactions
        self.transactions = df

        print(
            f"✅ Loaded {len(df)} transactions for "
            f"{df['ticker'].nunique()} unique assets"
        )
        print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"🏦 Accounts: {', '.join(df['account'].unique())}")
        total_invested = df[df["total_value"] > 0]["total_value"].sum()
        print(f"💰 Total invested: ${total_invested:,.2f}")

        return df

    def _is_iso8601_date(self, date_str: str) -> bool:
        """Check if date string is in ISO8601 format (YYYY-MM-DD)."""

        # Pattern for YYYY-MM-DD format
        iso_pattern = r"^\d{4}-\d{2}-\d{2}"

        return bool(re.match(iso_pattern, date_str))

    def _clean_transaction_data(
        self,
        df: pd.DataFrame,
        default_tz: Union[str, tzinfo] = "America/New_York",
    ) -> pd.DataFrame:
        """Clean and validate transaction data."""
        # Make a copy to avoid modifying original
        df = df.copy()

        # Convert date column - enforce ISO8601 format
        try:
            # First check if dates are in ISO8601 format (YYYY-MM-DD)
            for i, date_str in enumerate(df["date"]):
                if not self._is_iso8601_date(str(date_str)):
                    raise ValueError(
                        f"Date at row {i} ('{date_str}')"
                        f" is not in ISO8601 format (YYYY-MM-DD)."
                        f" Please use format like '2023-01-15'."
                    )

            # df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
            df["date"] = to_tz_mixed(
                df["date"], tz=default_tz, format="%Y-%m-%d"
            )
            self.logger.debug(
                "Converted 'date' column to type %s", df["date"].dtype
            )
        except ValueError as e:
            if "is not in ISO8601 format" in str(e):
                raise e  # Re-raise our custom error
            else:
                raise ValueError(f"Error parsing dates: {e}") from e
        except Exception as e:
            raise ValueError(f"Error parsing dates: {e}") from e

        # Clean ticker symbols (uppercase, strip whitespace)
        df["ticker"] = df["ticker"].str.upper().str.strip()

        # Validate required columns
        reqd_columns = [
            "date",
            "ticker",
            "transaction_type",
            "shares",
            "price_per_share",
        ]
        opt_columns = [
            "amount",
            "account",
            "group1",
            "group2",
            "group3",
            "notes",
        ]

        missing_cols = [col for col in reqd_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Add optional columns if not present with default values
        for col in opt_columns:
            if col not in df.columns:
                if col == "account":
                    df[col] = "Default"
                elif col.startswith("group"):
                    df[col] = "Unassigned"
                elif col == "notes":
                    df[col] = ""
                elif col == "amount":
                    df[col] = DEFAULT_VALUE

                self.logger.debug(
                    "ℹ️  No '%s' column found. Added default values.", col
                )

        # Clean account names (strip whitespace, title case)
        df["account"] = df["account"].str.strip().str.title()

        # Clean Group columns (strip whitespace, title case)
        for group_col in ["group1", "group2", "group3"]:
            if group_col in df.columns:
                df[group_col] = (
                    df[group_col]
                    .fillna("Unassigned")
                    .astype(str)
                    .str.strip()
                    .str.title()
                )

        # Clean Notes column (strip whitespace only, preserve case)
        if "notes" in df.columns:
            df["notes"] = df["notes"].fillna("").astype(str).str.strip()

        # Validate transaction types
        valid_types = ["BUY", "SELL", "DIVIDEND", "SPLIT", "MERGER", "FEE"]
        df["transaction_type"] = df["transaction_type"].str.upper().str.strip()
        invalid_types = df[~df["transaction_type"].isin(valid_types)]
        if not invalid_types.empty:
            invalid_type_list = invalid_types["transaction_type"].unique()
            raise ValueError(
                f"Invalid transaction types found: {invalid_type_list}"
            )

        # Validate numeric fields
        numeric_columns = ["shares", "price_per_share", "amount"]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isna().any():
                raise ValueError(f"Invalid numeric values in column: {col}")
            if col in ["shares", "price_per_share"]:
                if (df[col] <= 0).any():
                    raise ValueError(
                        f"Non-positive values found in column: {col}"
                    )

        # Calculate total value for each transaction
        df["total_value"] = df["shares"] * df["price_per_share"]

        # For DIVIDEND and FEE transactions, use 'amount' column
        df.loc[df["transaction_type"] == "DIVIDEND", "total_value"] = df[
            "amount"
        ]
        df.loc[df["transaction_type"] == "FEE", "total_value"] = -df["amount"]

        # For SELL transactions, make shares negative for easier calculations
        df.loc[df["transaction_type"] == "SELL", "shares"] *= -1
        df.loc[df["transaction_type"] == "SELL", "total_value"] *= -1

        # Sort by date
        df = df.sort_values("date").reset_index(drop=True)
        self.logger.debug(
            "Cleaned transaction data:\n%s \nwith columns of types:\n%s",
            df.head(),
            df.dtypes,
        )
        return df

    def _validate_api_key(self):
        """Validate that required API keys are configured."""
        provider = self.config.get("api.data_provider", "alpha_vantage")

        if provider == "alpha_vantage":
            api_key = self.config.get("api.alpha_vantage_key")
            if not api_key:
                raise ValueError(
                    "Alpha Vantage API key required. "
                    "Add 'alpha_vantage_key' to your config.yaml under 'api' section. "
                    "Get free key: https://www.alphavantage.co/support/#api-key"
                )

    def _is_market_currently_open(self) -> bool:
        """Check if the market is currently open."""
        nyse = mcal.get_calendar("NYSE")
        now = datetime.now(tz=ZoneInfo("UTC"))
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
            start_date=today - timedelta(days=DEFAULT_LOOK_FORWARD_PRICE_DATA),
            end_date=today,
        )
        if schedule.empty:
            raise ValueError(
                f"No recent market days found in the last "
                f"{DEFAULT_LOOK_FORWARD_PRICE_DATA} days"
            )

        closed_days = schedule[schedule["market_close"] < today]
        last_closed_market_day = closed_days["market_close"].max()

        self.logger.debug(
            "Last closed market day is %s", last_closed_market_day
        )
        return pd.to_datetime(last_closed_market_day)

    def fetch_market_data(
        self,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        force_refresh: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch daily market data for all assets in portfolio plus benchmark.

        Args:
            start_date: Start date for market data (YYYY-MM-DD).
                       If None, uses first transaction date.
            end_date: End date for market data (YYYY-MM-DD).
                     If None, uses today.
            force_refresh: If True, refresh data even if cached.

        Returns:
            Dictionary mapping ticker symbols to price DataFrames
        """
        if self.transactions is None:
            raise ValueError(
                "Must load transactions first using load_transactions()"
            )

        # Ensure start_dt and end_dt are pd.Timestamp and not None
        default_start_date = None
        if start_date is None:
            self.logger.info(
                "No start date provided. "
                + "Defaulting to %s days before first transaction.",
                DEFAULT_LOOK_FORWARD_PRICE_DATA,
            )
            self.logger.debug(
                "self.transactions['date'].min() is %s and type %s",
                self.transactions["date"].min(),
                type(self.transactions["date"].min()),
            )
            default_start_date = self.transactions["date"].min() - timedelta(
                days=DEFAULT_LOOK_FORWARD_PRICE_DATA
            )
            self.logger.debug(
                "default_start_date is %s",
                default_start_date,
            )
        start_date = ensure_timestamp(start_date, default_start_date)
        self.logger.debug(
            "start_date after ensure_timestamp is %s",
            start_date,
        )

        default_end_date = None
        if self._is_market_currently_open():
            self.logger.info("Market is currently open")
            default_end_date = self._get_last_closed_market_day()
        else:
            self.logger.info("Market is currently closed")
            default_end_date = to_tz_mixed(datetime.now())

            if default_end_date is None:
                raise ValueError("default_end_date is None after to_tz_mixed")

        end_date = pd.Timestamp(
            self.config.get("analysis.default_end_date", None)
        )
        end_date = to_tz_mixed(end_date)
        if end_date is None:
            self.logger.warning(
                "No valid end date provided or found. Defaulting to %s",
                default_end_date.strftime("%Y-%m-%d"),
            )

        # Ensure default_end_date is a scalar Timestamp, not a Series
        if isinstance(default_end_date, pd.Series):
            if not default_end_date.empty:
                default_end_date = default_end_date.iloc[0]
            else:
                raise ValueError("default_end_date Series is empty")
        end_date = ensure_timestamp(end_date, default_end_date)
        end_date = min(end_date, default_end_date)

        # Get list of all tickers (portfolio + benchmark)
        portfolio_tickers = self.transactions["ticker"].unique().tolist()
        benchmark_ticker = self.config.get("settings.benchmark_ticker", "SPY")
        all_tickers = portfolio_tickers + [benchmark_ticker]
        all_tickers = list(set(all_tickers))  # Remove duplicates

        self.logger.info(
            "📊 Fetching market data for %s assets...", len(all_tickers)
        )

        self.logger.info("Using ticker %s for benchmark", benchmark_ticker)

        self.logger.info(
            "📅 Date range: %s to %s",
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )

        # Fetch data for each ticker
        market_data = {}
        failed_tickers = []

        # Checking if caching is enabled
        cache_enabled = self.config.get("settings.cache_market_data", True)
        cache_dir = self.config.get_market_data_path()
        self.logger.debug("Cache enabled: %s", cache_enabled)
        self.logger.debug("Cache directory: %s", cache_dir)
        self.logger.debug("Cache directory exists: %s", cache_dir.exists())

        for ticker in all_tickers:
            try:
                with timed_operation("Fetching market data", self.logger):
                    self.logger.info("  Fetching market data for %s", ticker)

                    # Check cache first (if enabled and not forcing refresh)
                    cached_data = self._get_cached_data(
                        ticker, start_date, end_date
                    )
                    if cached_data is not None and not force_refresh:
                        self.logger.debug(
                            "Cache HIT for %s - using cached data", ticker
                        )
                        market_data[ticker] = cached_data
                        continue
                    else:
                        self.logger.debug(
                            "cached_data is None: %s", cached_data is None
                        )
                        self.logger.debug("Force refresh: %s", force_refresh)
                        self.logger.debug(
                            "Cache MISS for %s - downloading from API", ticker
                        )

                    # Get Alpha Vantage API key
                    api_key = self.config.get("api.alpha_vantage_key")
                    if not api_key:
                        raise ValueError(
                            "Alpha Vantage API key required. "
                            "Get free key at https://www.alphavantage.co/support/#api-key"
                        )

                    # Download from Alpha Vantage
                    ts = TimeSeries(key=api_key, output_format="pandas")
                    # pylint: disable-next=unbalanced-tuple-unpacking
                    hist, _ = ts.get_daily_adjusted(  # type: ignore
                        symbol=ticker, outputsize="full"
                    )

                    if hist.empty:  # pylint: disable=no-member # type: ignore
                        failed_tickers.append(ticker)
                        continue

                    # Rename columns
                    hist = (
                        # pylint: disable-next=no-member
                        hist.rename(  # type: ignore
                            columns={
                                "1. open": "open",
                                "2. high": "high",
                                "3. low": "low",
                                "4. close": "close",
                                "5. adjusted close": "adj_close",
                                "6. volume": "volume",
                                "7. dividend amount": "dividend",
                                "8. split coefficient": "split_coefficient",
                            }
                        )
                    )
                    hist = hist[
                        [
                            "open",
                            "high",
                            "low",
                            "close",
                            "adj_close",
                            "volume",
                            "dividend",
                            "split_coefficient",
                        ]
                    ]
                    hist.index.name = "date"
                    hist = hist.reset_index()

                    # Ensure data is sorted date ascending
                    hist = hist.sort_values("date").reset_index(drop=True)

                    # Filter date range
                    hist["date"] = to_tz_mixed(hist["date"])
                    self.logger.debug(
                        "Data range for %s: %s to %s before filtering",
                        ticker,
                        hist["date"].min(),
                        hist["date"].max(),
                    )

                    self.logger.debug(
                        "start_date is %s and end_date is %s",
                        start_date,
                        end_date,
                    )
                    self.logger.debug(
                        "hist['date'] is %s and type %s",
                        hist["date"].head(),
                        type(hist["date"]),
                    )

                    hist = hist[
                        (hist["date"] >= start_date)
                        & (hist["date"] <= end_date)
                    ]
                    self.logger.debug(
                        "Data range for %s: %s to %s "
                        + "AFTER filtering against %s to %s",
                        ticker,
                        hist["date"].min(),
                        hist["date"].max(),
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d"),
                    )

                    # Cache the data
                    self._cache_data(ticker, hist)

                    market_data[ticker] = hist

            except (OSError, ValueError) as e:
                self.logger.error("  ❌ Failed to download %s: %s", ticker, e)
                failed_tickers.append(ticker)

        if failed_tickers:
            self.logger.error(
                "⚠️  Failed to download data for: %s", failed_tickers
            )
            if benchmark_ticker in failed_tickers:
                self.logger.warning(
                    "❌ Warning: Benchmark %s data unavailable",
                    benchmark_ticker,
                )

        # Store market data
        self.logger.debug(
            "Fetched market data for %s tickers", len(market_data)
        )
        self.market_data = market_data

        # Separate benchmark data for easy access
        if benchmark_ticker in market_data:
            self.benchmark_data = market_data[benchmark_ticker].copy()

        print(f"✅ Successfully downloaded data for {len(market_data)} assets")
        return market_data

    def _get_cached_data(
        self, ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> Optional[pd.DataFrame]:
        """Check if we have cached data for the ticker and date range."""
        cache_dir = self.config.get_market_data_path()
        cache_file = cache_dir / f"{ticker}.parquet"

        self.logger.debug("Checking cache for %s at %s", ticker, cache_file)
        self.logger.debug("Cache file exists: %s", cache_file.exists())

        if not cache_file.exists():
            self.logger.debug("No cache file for %s", ticker)
            return None

        try:
            cached_df = pd.read_parquet(cache_file)
            cached_df["date"] = to_tz_mixed(cached_df["date"])

            # Ensure ascending date order
            cached_df = cached_df.sort_values("date").reset_index(drop=True)

            # Check if cached data covers our date range
            cached_start = to_tz_mixed(cached_df["date"].min())
            cached_end = to_tz_mixed(cached_df["date"].max())

            self.logger.debug(
                "Cached data for %s from %s to %s with start %s and end %s",
                ticker,
                cached_start,
                cached_end,
                start_date,
                end_date,
            )

            # Matching dates to trading days only
            nyse = mcal.get_calendar("NYSE")

            requested_schedule = nyse.schedule(
                start_date=start_date, end_date=end_date
            )

            if requested_schedule.empty:
                self.logger.debug(
                    "No trading days in requested range %s to %s",
                    start_date,
                    end_date,
                )
                return None
            first_trading_day = to_tz_mixed(requested_schedule.index.min())
            last_trading_day = to_tz_mixed(requested_schedule.index.max())

            self.logger.debug(
                "Comparing requested trading days from %s %s to %s %s",
                first_trading_day,
                type(first_trading_day),
                last_trading_day,
                type(last_trading_day),
            )
            self.logger.debug(
                "With cached data from %s %s to %s %s",
                cached_start,
                type(cached_start),
                cached_end,
                type(cached_end),
            )

            if (
                cached_start <= first_trading_day
                and cached_end >= last_trading_day
            ):
                self.logger.debug("Cache is valid for requested range")
                # Filter to requested date range
                mask = cached_df["date"].isin(
                    to_tz_mixed(requested_schedule.index)
                )
                self.logger.debug(
                    "Cache valid for %s from %s to %s",
                    ticker,
                    cached_start,
                    cached_end,
                )
                self.logger.debug(
                    "Head and tail of cached data for %s:\n%s\n%s",
                    ticker,
                    cached_df.head(n=5),
                    cached_df.tail(n=5),
                )
                return cached_df[mask].copy()
            else:
                self.logger.debug(
                    "Cache for %s is out of range (%s to %s)",
                    ticker,
                    cached_start,
                    cached_end,
                )
                return None

        except (OSError, ValueError) as e:
            # If there's any issue with cached data, ignore it
            self.logger.warning(
                (
                    "⚠️  Invalid cache for %s, with start date %s and "
                    "end date %s; ignoring error: %s"
                ),
                ticker,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                e,
            )

        return None

    def _cache_data(self, ticker: str, data: pd.DataFrame) -> None:
        """Cache market data to disk."""
        if not self.config.get("settings.cache_market_data", True):
            self.logger.debug("Caching disabled, skipping cache for %s", ticker)
            return

        cache_dir = self.config.get_market_data_path()
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_file = cache_dir / f"{ticker}.parquet"

        self.logger.debug("Caching data for %s at %s", ticker, cache_file)
        self.logger.debug("Data shape: %s", data.shape)

        try:
            data.to_parquet(cache_file, index=False)
            self.logger.debug("Cache saved for %s", ticker)
        except Exception as e:
            self.logger.warning(
                "⚠️  Warning: Could not cache data for %s: %s", ticker, e
            )

    def _get_price_for_date(
        self, ticker: str, target_date: pd.Timestamp
    ) -> float:
        """Get price for ticker on specific date with forward-fill logic."""
        if ticker not in self.market_data:
            raise ValueError(f"No market data available for {ticker}")

        ticker_data = self.market_data[ticker]
        target_date = ensure_timestamp(target_date)

        # Try exact date match first
        # self.logger.debug("Looking for price of %s on %s", ticker, target_date)
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
                print(
                    f"Warning: No recent price data for {ticker} near {target_date}"
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

    def _calculate_modified_dietz_returns(
        self, portfolio_df: pd.DataFrame
    ) -> pd.Series:
        """Calculate portfolio returns using Modified Dietz method."""
        returns = []

        for i in range(len(portfolio_df)):
            if i == 0:
                # First day - beginning_value is zero
                beginning_value = np.float64(0.0)  # zero
            else:
                beginning_value = portfolio_df.iloc[i - 1]["total_value"]

            # Get values
            ending_value = portfolio_df.iloc[i]["total_value"]
            net_cash_flow = portfolio_df.iloc[i]["net_cash_flow"]
            weighted_cash_flow = portfolio_df.iloc[i]["weighted_cash_flow"]

            # Modified Dietz formula
            denominator = beginning_value + weighted_cash_flow

            if denominator <= 0:
                # Handle edge case: no beginning value or negative denominator
                returns.append(DEFAULT_RETURN)
            else:
                daily_return = (
                    ending_value - beginning_value - net_cash_flow
                ) / denominator
                returns.append(daily_return)

        return pd.Series(returns)

    def _process_daily_transactions(
        self, date: pd.Timestamp
    ) -> Dict[str, float]:
        """Process all transactions for a specific date and return cash
        flows by account."""
        if isinstance(date, str):
            date = pd.to_datetime(date)

        day_transactions = self.transactions[
            self.transactions["date"].dt.date == date  # .date()
        ]

        cash_flows = {"total": DEFAULT_CASH_FLOW}

        # Initialize all accounts
        for account in self.transactions["account"].unique():
            cash_flows[account] = DEFAULT_CASH_FLOW

        # Process each transaction
        for _, trans in day_transactions.iterrows():
            account = trans["account"]
            ttype = trans["transaction_type"]

            if ttype in ["DIVIDEND", "FEE"]:
                cash_flow = trans["amount"]
            elif ttype in ["DIVIDEND_REINVEST"]:
                cash_flow = 0.0
            else:
                # Already signed correctly (negative for SELL)
                cash_flow = trans["total_value"]

            cash_flows[account] += cash_flow
            cash_flows["total"] += cash_flow

        return cash_flows

    def _calculate_account_modified_dietz_returns(
        self, portfolio_df: pd.DataFrame, account: str
    ) -> pd.Series:
        """Calculate account-level returns using Modified Dietz method."""
        returns = []
        account_total_col = f"{account}_total"
        account_cash_flow_col = f"{account}_cash_flow"
        account_weighted_cash_flow_col = f"{account}_weighted_cash_flow"

        for i in range(len(portfolio_df)):
            if i == 0:
                returns.append(DEFAULT_RETURN)
                continue

            beginning_value = portfolio_df.iloc[i - 1][account_total_col]
            ending_value = portfolio_df.iloc[i][account_total_col]
            net_cash_flow = portfolio_df.iloc[i][account_cash_flow_col]
            weighted_cash_flow = portfolio_df.iloc[i][
                account_weighted_cash_flow_col
            ]

            denominator = beginning_value + weighted_cash_flow

            if denominator <= 0:
                returns.append(DEFAULT_RETURN)
            else:
                daily_return = (
                    ending_value - beginning_value - net_cash_flow
                ) / denominator
                returns.append(daily_return)

        return pd.Series(returns)

    def build_portfolio_history(self) -> pd.DataFrame:
        """
        Build portfolio holdings and values over time.

        Returns:
            DataFrame with daily portfolio holdings, values, and weights
        """
        if self.transactions is None:
            raise ValueError("Must load transactions first")
        if not self.market_data:
            raise ValueError("Must fetch market data first")

        self.logger.info("🏗️  Building portfolio history...")

        # Get all unique tickers and accounts
        tickers = self.transactions["ticker"].unique()
        accounts = self.transactions["account"].unique()

        # Initialize portfolio tracking by account and ticker
        portfolio_data = []
        # Track holdings by account and ticker: {account: {ticker: shares}}
        current_holdings = {
            account: {ticker: 0.0 for ticker in tickers} for account in accounts
        }

        # Get full date range for analysis
        start_date = self.transactions["date"].min()
        end_date = max([df["date"].max() for df in self.market_data.values()])

        # Get actual NYSE trading days (excludes weekends AND holidays)
        self.logger.debug(
            "Getting NYSE trading days from %s to %s", start_date, end_date
        )
        nyse = mcal.get_calendar("NYSE")
        trading_days = nyse.schedule(start_date=start_date, end_date=end_date)
        date_range = trading_days.index.date  # Convert to date objects

        # Check if all transaction dates are trading days
        self.logger.debug("Validating transaction dates against trading days")
        transaction_dates = set(self.transactions["date"].dt.date)
        trading_dates = set(date_range)

        self.logger.debug("Checking for transactions on non-trading days")
        non_trading_dates = transaction_dates - trading_dates
        if non_trading_dates:
            non_trading_sorted = sorted(non_trading_dates)
            self.logger.warning(
                "⚠️  Warning: %d transactions on non-trading days:",
                len(non_trading_dates),
            )

            for date in non_trading_sorted[:5]:
                day_name = pd.to_datetime(date).strftime("%A")
                # Check if it's a weekend or holiday
                if day_name in ["Saturday", "Sunday"]:
                    reason = "weekend"
                else:
                    reason = "market holiday"
                print(f"   {date} ({day_name}, {reason})")
            if len(non_trading_sorted) > 5:
                print(f"   ... and {len(non_trading_sorted) - 5} more")
            print("   These transactions will use forward-filled prices.")

        self.logger.debug("Processing each trading day for portfolio history")
        for date in date_range:
            # Process any transactions on this date
            day_transactions = self.transactions[
                self.transactions["date"].dt.date == date  # date.date()
            ]

            for _, transaction in day_transactions.iterrows():
                account = transaction["account"]
                ticker = transaction["ticker"]
                ttype = transaction["transaction_type"]

                if ttype in ["BUY", "SELL", "DIVIDEND_REINVEST"]:
                    shares = transaction["shares"]

                    # Ensure account and ticker exist in our tracking
                    if account not in current_holdings:
                        current_holdings[account] = {
                            t: DEFAULT_ASSET_VALUE for t in tickers
                        }
                    if ticker not in current_holdings[account]:
                        current_holdings[account][ticker] = DEFAULT_ASSET_VALUE

                    # For DIVIDEND, do not adjust holdings;
                    # handled in cash flow
                    current_holdings[account][ticker] += shares

            # Get market prices for this date
            self.logger.debug(
                "Calculating portfolio value for %s", date
            )  # date.date())
            day_data = {"date": date}
            total_portfolio_value = DEFAULT_ASSET_VALUE
            account_totals = {}

            for account in accounts:
                account_value = DEFAULT_ASSET_VALUE

                for ticker in tickers:
                    shares = current_holdings[account][ticker]

                    try:
                        # self.logger.debug(
                        #     "Getting price for %s on %s", ticker, date
                        # )
                        price = self._get_price_for_date(ticker, date)
                    except ValueError as e:
                        self.logger.error(f"Skipping {ticker} on {date}: {e}")
                        price = DEFAULT_PRICE  # Only set to 0 if truly no data exists

                    # self.logger.debug(
                    #     "Calculated price for %s on %s: $%.2f",
                    #     ticker,
                    #     date,
                    #     price,
                    # )
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
                        ticker_data["date"].dt.date == date  # date.date()
                    ]
                    if not price_data.empty:
                        price = price_data["close"].iloc[0]
                    else:
                        available_data = ticker_data[
                            ticker_data["date"].dt.date <= date  # date.date()
                        ]
                        if not available_data.empty:
                            price = available_data["close"].iloc[-1]
                        else:
                            price = DEFAULT_PRICE
                else:
                    price = DEFAULT_PRICE

                day_data[f"{ticker}_total_shares"] = total_shares
                day_data[f"{ticker}_total_value"] = total_shares * price

            portfolio_data.append(day_data)

        # Convert to DataFrame
        self.logger.debug("Converting portfolio data to DataFrame")
        portfolio_df = pd.DataFrame(portfolio_data)

        # Ensure ascending date order
        portfolio_df = portfolio_df.sort_values("date").reset_index(drop=True)

        portfolio_df["net_cash_flow"] = DEFAULT_CASH_FLOW
        portfolio_df["weighted_cash_flow"] = DEFAULT_CASH_FLOW
        portfolio_df["market_value_change"] = DEFAULT_ASSET_VALUE
        portfolio_df["market_value_return"] = DEFAULT_RETURN

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
                        ).fillna(DEFAULT_WEIGHT)

        # Calculate overall weights
        for ticker in tickers:
            portfolio_df[f"{ticker}_weight"] = (
                portfolio_df[f"{ticker}_total_value"]
                / portfolio_df["total_value"]
            ).fillna(DEFAULT_WEIGHT)

        # Calculate daily cash flow adjust returns
        # Calculate cash flows for each day using helper function
        period_cash_flow_weight = float(
            self.config.get(
                "advanced.performance.period_cash_flow_weight", DEFAULT_WEIGHT
            )
        )
        for i, row in portfolio_df.iterrows():
            date = row["date"]
            if isinstance(date, str):
                date = pd.to_datetime(date)

            cash_flows = self._process_daily_transactions(date)

            portfolio_df.at[i, "net_cash_flow"] = cash_flows["total"]
            portfolio_df.at[i, "weighted_cash_flow"] = (
                cash_flows["total"] * period_cash_flow_weight
            )  # Mid-day weighting

            # Store ALL account cash flows
            for account in accounts:
                portfolio_df.at[i, f"{account}_cash_flow"] = cash_flows.get(
                    account, DEFAULT_CASH_FLOW
                )
                portfolio_df.at[i, f"{account}_weighted_cash_flow"] = (
                    cash_flows.get(account, DEFAULT_CASH_FLOW)
                    * period_cash_flow_weight
                )

        # Calculate market value change (pure investment performance)
        for i in range(0, len(portfolio_df)):
            if i == 0:
                prev_value = np.float64(0.0)  # zero
            else:
                prev_value = portfolio_df.iloc[i - 1]["total_value"]

            current_value = portfolio_df.iloc[i]["total_value"]
            net_cash_flow = portfolio_df.iloc[i]["net_cash_flow"]

            # Market value change = total change minus cash flow impact
            market_change = current_value - prev_value - net_cash_flow

            if prev_value == 0:
                market_return = 0
            else:
                market_return = market_change / prev_value

            portfolio_df.iloc[
                i, portfolio_df.columns.get_loc("market_value_change")
            ] = market_change

            portfolio_df.iloc[
                i, portfolio_df.columns.get_loc("market_value_return")
            ] = market_return

        # For clarity, also store cash flow impact column
        portfolio_df["cash_flow_impact"] = portfolio_df["net_cash_flow"]

        # Calculate Modified Dietz returns
        portfolio_df["portfolio_return"] = (
            self._calculate_modified_dietz_returns(portfolio_df)
        )

        # Handle edge case where previous total_value is zero
        denominator_is_zero_mask = portfolio_df["total_value"].shift(1) == 0
        portfolio_df.loc[denominator_is_zero_mask, "portfolio_return"] = (
            DEFAULT_RETURN
        )

        self.logger.debug(
            "Portfolio returns had %d NaN values; replaced with %s",
            portfolio_df["portfolio_return"].isna().sum(),
            DEFAULT_RETURN,
        )

        # Calculate account-specific returns
        for account in accounts:
            account_total_col = f"{account}_total"
            if account_total_col in portfolio_df.columns:
                portfolio_df[f"{account}_return"] = (
                    self._calculate_account_modified_dietz_returns(
                        portfolio_df, account
                    )
                )

        self.portfolio_history = portfolio_df

        self.logger.info(
            "✅ Portfolio history built: %d days", len(portfolio_df)
        )
        self.logger.info(
            "🏦 Tracking %d accounts: %s", len(accounts), ", ".join(accounts)
        )

        return portfolio_df

    def fetch_dividend_data(self, ticker: str) -> pd.DataFrame:
        """
        Extracts dividend data for a given ticker from already-fetched
        self.market_data.
        Returns a DataFrame with columns: 'date', 'dividend'
        """
        if ticker not in self.market_data:
            raise ValueError(
                f"Market data for ticker {ticker} not found. "
                f"Did you run fetch_market_data()?"
            )

        df = self.market_data[ticker]
        # Only consider days where a dividend was paid
        dividend_df = df[df["dividend"] > 0][["date", "dividend"]].copy()

        # Normalize date for merging
        dividend_df["date"] = pd.to_datetime(dividend_df["date"]).dt.normalize()
        return dividend_df

    def compare_user_dividends_to_alphavantage(self, ticker: str):
        """Compare user-provided dividend amounts to fetched market data and
        warn if mismatched."""
        ERROR_MARGIN = 0.01  # Allowable difference in dollars

        user_dividends = self.transactions[
            (self.transactions["ticker"] == ticker)
            & (
                self.transactions["transaction_type"].isin(
                    ["DIVIDEND", "DIVIDEND_REINVEST"]
                )
            )
        ][["date", "amount"]].copy()
        user_dividends["date"] = user_dividends["date"].dt.normalize()

        market_dividends = self.fetch_dividend_data(ticker)
        market_dividends["date"] = market_dividends["date"].dt.normalize()

        merged = pd.merge(
            user_dividends,
            market_dividends,
            on="date",
            how="left",
            suffixes=("_user", "_av"),
        )
        mismatches = merged[
            (merged["dividend"].notnull())
            & (abs(merged["amount"] - merged["dividend"]) > ERROR_MARGIN)
        ]
        for _, row in mismatches.iterrows():
            print(
                f"⚠️  Dividend mismatch for {ticker} on {row['date'].date()}: "
                f"user={row['amount']} vs AlphaVantage={row['dividend']}"
            )

    def calculate_performance(self) -> "PerformanceResults":
        """
        Calculate comprehensive performance metrics for the portfolio and benchmark.

        Returns:
            PerformanceResults object containing all metrics and analysis
        """
        if self.portfolio_history is None:
            self.build_portfolio_history()

        print("📊 Calculating performance metrics...")

        # Calculate portfolio performance metrics
        portfolio_metrics = self._calculate_metrics(
            self.portfolio_history["portfolio_return"].dropna(), "Portfolio"
        )

        # Calculate benchmark performance metrics
        benchmark_metrics = {}
        if self.benchmark_data is not None:
            # Align benchmark data with portfolio dates
            benchmark_returns = self._align_benchmark_returns()
            self.portfolio_history["Benchmark_Returns"] = benchmark_returns
            benchmark_metrics = self._calculate_metrics(
                benchmark_returns[1:], "Benchmark"
            )

        # Calculate relative performance metrics
        relative_metrics = self._calculate_relative_metrics(
            self.portfolio_history["portfolio_return"].dropna()[1:],
            benchmark_returns[1:] if self.benchmark_data is not None else None,
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

        print("✅ Performance analysis complete!")
        return results

    def _align_benchmark_returns(self) -> pd.Series:
        """Align benchmark returns with portfolio dates."""
        portfolio_dates = self.portfolio_history["date"]

        # Convert benchmark data to returns
        benchmark_df = self.benchmark_data.copy()

        # Ensure ascending date order
        benchmark_df = benchmark_df.sort_values("date").reset_index(drop=True)

        # Calculate period on period returns
        # Use adjusted close prices for benchmark returns
        benchmark_df["return"] = benchmark_df["adj_close"].pct_change()

        # Align dates
        aligned_returns = []
        for date in portfolio_dates:
            benchmark_data_for_date = benchmark_df[
                benchmark_df["date"].dt.date == date  # date.date()
            ]

            if not benchmark_data_for_date.empty:
                aligned_returns.append(
                    benchmark_data_for_date["return"].iloc[0]
                )
            else:
                aligned_returns.append(np.nan)

        # Bechmark returns are based on previous to current close, so the first
        # return is by definition zero, as there is no previous close
        aligned_returns[0] = np.float64(0.0)

        return pd.Series(aligned_returns).dropna()

    def _calculate_metrics(self, returns: pd.Series, name: str) -> Dict:
        """Calculate performance metrics for a return series."""
        if returns.empty:
            return {}

        # Basic statistics
        total_periods = len(returns)
        annual_trading_days = int(
            self.config.get(
                "settings.annual_trading_days", DEFAULT_TRADING_DAYS
            )
        )
        year_fraction = total_periods / annual_trading_days

        returns = pd.to_numeric(returns, errors="coerce").dropna()

        self.logger.debug(
            "%s: First 10 daily returns:\n%s",
            name,
            returns.head(n=10) * TO_PERCENT,
        )

        total_return = float((1 + returns).prod() - 1)
        self.logger.debug(
            "Calculating CAGR with total_return=%.6f%%, year_fraction=%.6f",
            total_return * TO_PERCENT,
            year_fraction,
        )
        annualized_return = cagr(1, 1 + total_return, year_fraction)

        self.logger.info(
            "%s: Periods: %d, Years: %.2f",
            name,
            total_periods,
            year_fraction,
        )

        self.logger.debug(
            "%s: First date: %s, Last date: %s",
            name,
            returns.index[0] if len(returns) > 0 else "N/A",
            returns.index[-1] if len(returns) > 0 else "N/A",
        )

        self.logger.info(
            "%s: Total Return: %.2f%%, Annualized Return: %.2f%%",
            name,
            total_return * TO_PERCENT,
            annualized_return * TO_PERCENT,
        )

        volatility = returns.std(ddof=1)  # Daily volatility, sample stddev
        annual_volatility = volatility * np.sqrt(
            annual_trading_days
        )  # Annualized volatility

        self.logger.info(
            "%s: Volatility: %.2f%% (period) %.2f%% (annualized)",
            name,
            volatility * TO_PERCENT,
            annual_volatility * TO_PERCENT,
        )

        # Risk-adjusted metrics
        annual_risk_free_rate = float(
            self.config.get("settings.risk_free_rate", DEFAULT_RISK_FREE_RATE)
        )
        daily_risk_free_rate = cagr(
            1, 1 + annual_risk_free_rate, annual_trading_days
        )
        excess_returns = returns - daily_risk_free_rate
        excess_mean_returns = excess_returns.mean()
        sharpe_ratio = (
            excess_mean_returns / volatility * np.sqrt(annual_trading_days)
            if returns.std() > 0
            else 0
        )

        # self.logger.debug(
        #     "excess returns:\n%s\ndaily_risk_free_rate=%.6f, "
        #     + "excess_mean_returns=%.6f, volatility=%.6f, sharpe_ratio=%.6f",
        #     excess_returns,
        #     daily_risk_free_rate,
        #     excess_mean_returns,
        #     volatility,
        #     sharpe_ratio,
        # )

        # Drawdown analysis
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # self.logger.info(
        #     "cumulative returns:\n%s\nrunning max:\n%s\n"
        #     + "drawdown:\n%s\nmax drawdown:%.6f%%",
        #     cumulative_returns,
        #     running_max,
        #     drawdown,
        #     max_drawdown * TO_PERCENT,
        # )

        # Additional metrics
        positive_periods = (returns > 0).sum()
        win_rate = positive_periods / total_periods if total_periods > 0 else 0

        return {
            "name": name,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": annual_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_periods": total_periods,
            "start_date": returns.index[0] if len(returns) > 0 else None,
            "end_date": returns.index[-1] if len(returns) > 0 else None,
        }

    def _calculate_relative_metrics(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series],
    ) -> Dict:
        """Calculate relative performance metrics vs benchmark."""
        if benchmark_returns is None or portfolio_returns.empty:
            return {}

        annual_trading_days = int(
            self.config.get(
                "settings.annual_trading_days", DEFAULT_TRADING_DAYS
            )
        )

        # Align the series
        min_length = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns.iloc[-min_length:]
        benchmark_returns = benchmark_returns.iloc[-min_length:]

        # Calculate excess returns (alpha)
        excess_returns = portfolio_returns - benchmark_returns

        # Tracking error (volatility of excess returns)
        tracking_error = excess_returns.std() * np.sqrt(annual_trading_days)

        # Information ratio (excess return / tracking error)
        information_ratio = (
            (excess_returns.mean() * annual_trading_days) / tracking_error
            if tracking_error > 0
            else 0
        )

        # Beta calculation
        covariance = np.cov(portfolio_returns, benchmark_returns, ddof=1)[0, 1]
        benchmark_variance = np.var(benchmark_returns, ddof=1)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

        # Jensen's Alpha (risk-adjusted excess return)
        risk_free_rate = self.config.get(
            "settings.risk_free_rate", DEFAULT_RISK_FREE_RATE
        )
        daily_risk_free_rate = cagr(1, 1 + risk_free_rate, annual_trading_days)
        portfolio_excess = portfolio_returns.mean() - daily_risk_free_rate
        benchmark_excess = benchmark_returns.mean() - daily_risk_free_rate
        jensens_alpha = portfolio_excess - (beta * benchmark_excess)
        jensens_alpha_annualized = jensens_alpha * annual_trading_days

        return {
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
            "beta": beta,
            "jensens_alpha": jensens_alpha_annualized,
            "correlation": np.corrcoef(portfolio_returns, benchmark_returns)[
                0, 1
            ],
        }


class PerformanceResults:
    """Container for portfolio performance analysis results."""

    def __init__(
        self,
        portfolio_metrics: Dict,
        benchmark_metrics: Dict,
        relative_metrics: Dict,
        portfolio_history: pd.DataFrame,
        config: ConfigManager,
    ):
        self.portfolio_metrics = portfolio_metrics
        self.benchmark_metrics = benchmark_metrics
        self.relative_metrics = relative_metrics
        self.portfolio_history = portfolio_history
        self.config = config

    def summary(self) -> str:
        """Generate a summary report of the performance analysis."""
        lines = []
        lines.append("=" * 60)
        lines.append("🎯 BOGLEBENCH PERFORMANCE ANALYSIS")
        lines.append("   'Stay the course' - John C. Bogle")
        lines.append("=" * 60)

        # Portfolio metrics
        if self.portfolio_metrics:
            p = self.portfolio_metrics
            lines.append("\n📊 PORTFOLIO PERFORMANCE")
            lines.append(f"  Total Return:        {p['total_return']:.2%}")
            lines.append(f"  Annualized Return:   {p['annualized_return']:.2%}")
            lines.append(f"  Volatility:          {p['volatility']:.2%}")
            lines.append(f"  Sharpe Ratio:        {p['sharpe_ratio']:.3f}")
            lines.append(f"  Max Drawdown:        {p['max_drawdown']:.2%}")
            lines.append(f"  Win Rate:            {p['win_rate']:.2%}")

        # Benchmark metrics
        if self.benchmark_metrics:
            b = self.benchmark_metrics
            benchmark_name = self.config.get(
                "settings.benchmark_ticker", "Benchmark"
            )
            lines.append(f"\n📈 {benchmark_name} PERFORMANCE")
            lines.append(f"  Total Return:        {b['total_return']:.2%}")
            lines.append(f"  Annualized Return:   {b['annualized_return']:.2%}")
            lines.append(f"  Volatility:          {b['volatility']:.2%}")
            lines.append(f"  Sharpe Ratio:        {b['sharpe_ratio']:.3f}")
            lines.append(f"  Max Drawdown:        {b['max_drawdown']:.2%}")

        # Relative performance
        if self.relative_metrics:
            r = self.relative_metrics
            lines.append("\n🎯 RELATIVE PERFORMANCE")
            lines.append(f"  Information Ratio:   {r['information_ratio']:.3f}")
            lines.append(f"  Tracking Error:      {r['tracking_error']:.2%}")
            lines.append(f"  Beta:                {r['beta']:.3f}")
            lines.append(f"  Jensen's Alpha:      {r['jensens_alpha']:.2%}")
            lines.append(f"  Correlation:         {r['correlation']:.3f}")

        lines.append("\n" + "=" * 60)
        lines.append(
            "💡 Remember: Past performance doesn't guarantee future results."
        )
        lines.append(
            "   Focus on low costs, diversification, and long-term discipline."
        )

        return "\n".join(lines)

    def get_portfolio_returns(self) -> pd.Series:
        """Get portfolio return series."""
        return self.portfolio_history["portfolio_return"].dropna()

    def get_cumulative_returns(self) -> pd.Series:
        """Get cumulative portfolio returns."""
        returns = self.get_portfolio_returns()
        return (1 + returns).cumprod() - 1

    def get_account_summary(self) -> pd.DataFrame:
        """Get summary of portfolio value by account."""
        if self.portfolio_history is None:
            return pd.DataFrame()

        annual_trading_days = int(
            self.config.get(
                "settings.annual_trading_days", DEFAULT_TRADING_DAYS
            )
        )

        # Get the latest date data
        latest_data = self.portfolio_history.iloc[-1]

        accounts = (
            self.config.transactions["account"].unique()
            if hasattr(self.config, "transactions")
            else []
        )
        if not accounts and hasattr(self, "portfolio_history"):
            # Extract account names from column names
            account_cols = [
                col
                for col in self.portfolio_history.columns
                if col.endswith("_total")
            ]
            accounts = [col.replace("_total", "") for col in account_cols]

        account_data = []
        for account in accounts:
            total_col = f"{account}_total"
            if total_col in self.portfolio_history.columns:
                current_value = latest_data[total_col]

                # Calculate account-specific return
                account_returns = self.portfolio_history[
                    f"{account}_return"
                ].dropna()
                if len(account_returns) > 0:
                    total_periods = len(account_returns)
                    year_fraction = total_periods / annual_trading_days
                    total_return = (1 + account_returns).prod() - 1
                    annualized_return = cagr(1, 1 + total_return, year_fraction)
                else:
                    total_return = DEFAULT_RETURN
                    annualized_return = DEFAULT_RETURN

                account_data.append(
                    {
                        "account": account,
                        "current_value": current_value,
                        "total_return": total_return,
                        "annualized_return": annualized_return,
                        "weight_of_portfolio": (
                            current_value / latest_data["total_value"]
                            if latest_data["total_value"] > 0
                            else 0
                        ),
                    }
                )

        return pd.DataFrame(account_data)

    def get_account_holdings(self, account_name: str = None) -> pd.DataFrame:
        """Get current holdings for a specific account or all accounts."""
        if self.portfolio_history is None:
            return pd.DataFrame()

        latest_data = self.portfolio_history.iloc[-1]

        if account_name:
            accounts = [account_name]
        else:
            # Get all accounts
            account_cols = [
                col
                for col in self.portfolio_history.columns
                if col.endswith("_total")
            ]
            accounts = [col.replace("_total", "") for col in account_cols]

        holdings_data = []

        for account in accounts:
            # Find all ticker columns for this account
            ticker_cols = [
                col
                for col in self.portfolio_history.columns
                if col.startswith(f"{account}_") and col.endswith("_shares")
            ]

            for col in ticker_cols:
                ticker = col.replace(f"{account}_", "").replace("_shares", "")
                shares = latest_data[col]

                if shares != 0:  # Only include non-zero holdings
                    value_col = f"{account}_{ticker}_value"
                    price_col = f"{account}_{ticker}_price"

                    value = latest_data.get(value_col, 0)
                    price = latest_data.get(price_col, 0)

                    account_total = latest_data.get(f"{account}_total", 0)
                    weight = value / account_total if account_total > 0 else 0

                    holdings_data.append(
                        {
                            "account": account,
                            "ticker": ticker,
                            "shares": shares,
                            "price": price,
                            "value": value,
                            "weight": weight,
                        }
                    )

        """Export results to CSV files."""
        output_dir = self.config.get_output_path()
        output_path = self._export_history_metrics_to_csv(output_dir)

        print(f"📁 Results exported to: {output_path}")
        return pd.DataFrame(holdings_data)

    def export_to_csv(self, output_dir: Optional[str] = None) -> str:
        """Export results to CSV files."""
        output_path = self._export_history_metrics_to_csv(output_dir)
        print(f"Results exported to: {output_path}")
        return str(output_path)

    def _export_history_metrics_to_csv(
        self, output_dir: Optional[str] = None
    ) -> str:
        """Export metrics and history to csv file"""
        if output_dir is None:
            output_dir = self.config.get_output_path()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export portfolio history
        history_file = output_path / "portfolio_history.csv"
        self.portfolio_history.to_csv(history_file, index=False)

        # Export performance metrics
        metrics_data = []
        if self.portfolio_metrics:
            metrics_data.append({**self.portfolio_metrics, "type": "Portfolio"})
        if self.benchmark_metrics:
            metrics_data.append({**self.benchmark_metrics, "type": "Benchmark"})

        if metrics_data:
            metrics_file = output_path / "performance_metrics.csv"
            pd.DataFrame(metrics_data).to_csv(metrics_file, index=False)

        # Export relative metrics
        relative_data = []
        if self.relative_metrics:
            relative_data.append({**self.relative_metrics, "type": "Relative"})

        if relative_data:
            relative_file = output_path / "relative_metrics.csv"
            pd.DataFrame(relative_data).to_csv(relative_file, index=False)

        return str(output_path)
