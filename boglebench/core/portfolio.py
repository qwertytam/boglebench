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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from alpha_vantage.timeseries import TimeSeries

from ..utils.config import ConfigManager
from ..utils.logging_config import get_logger, setup_logging
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
        setup_logging()  # Initialize after workspace context is set
        self.logger = get_logger("core.portfolio")

        self.config = ConfigManager(config_path)
        self.logger = get_logger("core.portfolio")
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

    def _clean_transaction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate transaction data."""
        # Make a copy to avoid modifying original
        df = df.copy()

        # Convert date column - enforce ISO8601 format
        try:
            # First check if dates are in ISO8601 format (YYYY-MM-DD)
            for i, date_str in enumerate(df["date"]):
                if not self._is_iso8601_date(str(date_str)):
                    raise ValueError(
                        f"Date at row {i} ('{date_str}') is not in ISO8601 format "
                        f"(YYYY-MM-DD). Please use format like '2023-01-15'."
                    )

            df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        except ValueError as e:
            if "is not in ISO8601 format" in str(e):
                raise e  # Re-raise our custom error
            else:
                raise ValueError(f"Error parsing dates: {e}")
        except Exception as e:
            raise ValueError(f"Error parsing dates: {e}")

        # Clean ticker symbols (uppercase, strip whitespace)
        df["ticker"] = df["ticker"].str.upper().str.strip()

        # Add account column if not present (for backward compatibility)
        if "account" not in df.columns:
            df["account"] = "Default"
            print("ℹ️  No 'account' column found. Added default account name.")

        # Clean account names (strip whitespace, title case)
        df["account"] = df["account"].str.strip().str.title()

        # Validate transaction types
        valid_types = ["BUY", "SELL"]
        df["transaction_type"] = df["transaction_type"].str.upper().str.strip()
        invalid_types = df[~df["transaction_type"].isin(valid_types)]
        if not invalid_types.empty:
            invalid_type_list = invalid_types["transaction_type"].unique()
            raise ValueError(
                f"Invalid transaction types found: {invalid_type_list}"
            )

        # Validate numeric fields
        numeric_columns = ["shares", "price_per_share"]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isna().any():
                raise ValueError(f"Invalid numeric values in column: {col}")
            if (df[col] <= 0).any():
                raise ValueError(f"Non-positive values found in column: {col}")

        # Calculate total value for each transaction
        df["total_value"] = df["shares"] * df["price_per_share"]

        # For SELL transactions, make shares negative for easier calculations
        df.loc[df["transaction_type"] == "SELL", "shares"] *= -1
        df.loc[df["transaction_type"] == "SELL", "total_value"] *= -1

        # Sort by date
        df = df.sort_values("date").reset_index(drop=True)

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

    def fetch_market_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
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

        # Determine date range
        if start_date is None:
            # Buffer for calculations
            start_date = self.transactions["date"].min() - timedelta(days=30)
        else:
            start_date = pd.to_datetime(start_date)

        if end_date is None:
            end_date = datetime.now()
        else:
            end_date = pd.to_datetime(end_date)

        # Get list of all tickers (portfolio + benchmark)
        portfolio_tickers = self.transactions["ticker"].unique().tolist()
        benchmark_ticker = self.config.get("settings.benchmark_ticker", "SPY")
        all_tickers = portfolio_tickers + [benchmark_ticker]
        all_tickers = list(set(all_tickers))  # Remove duplicates

        print(f"📊 Fetching market data for {len(all_tickers)} assets...")
        print(f"📅 Date range: {start_date.date()} to {end_date.date()}")

        # Fetch data for each ticker
        market_data = {}
        failed_tickers = []

        for ticker in all_tickers:
            try:
                print(f"  Downloading {ticker}...")

                # Check cache first (if enabled and not forcing refresh)
                cached_data = self._get_cached_data(
                    ticker, start_date, end_date
                )
                if cached_data is not None and not force_refresh:
                    print("Using cached market data")
                    market_data[ticker] = cached_data
                    continue

                # Get Alpha Vantage API key
                api_key = self.config.get("api.alpha_vantage_key")
                if not api_key:
                    raise ValueError(
                        "Alpha Vantage API key required. "
                        "Get free key at https://www.alphavantage.co/support/#api-key"
                    )

                # Download from Alpha Vantage
                ts = TimeSeries(key=api_key, output_format="pandas")
                hist, _ = ts.get_daily_adjusted(
                    symbol=ticker, outputsize="full"
                )

                if hist.empty:
                    failed_tickers.append(ticker)
                    continue

                # Clean up the data - Alpha Vantage has different column names
                hist = hist.rename(
                    columns={
                        "1. open": "Open",
                        "2. high": "High",
                        "3. low": "Low",
                        "4. close": "Close",
                        "6. volume": "Volume",
                    }
                )
                hist = hist[["Open", "High", "Low", "Close", "Volume"]]
                hist.index.name = "date"
                hist = hist.reset_index()

                # Filter date range
                hist["date"] = pd.to_datetime(hist["date"])
                hist = hist[
                    (hist["date"] >= start_date) & (hist["date"] <= end_date)
                ]

                # Cache the data
                self._cache_data(ticker, hist)

                market_data[ticker] = hist

            except Exception as e:
                print(f"  ❌ Failed to download {ticker}: {e}")
                failed_tickers.append(ticker)

        if failed_tickers:
            print(f"⚠️  Failed to download data for: {failed_tickers}")
            if benchmark_ticker in failed_tickers:
                print(
                    f"❌ Warning: Benchmark {benchmark_ticker} data unavailable"
                )

        # Store market data
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

        if not cache_file.exists():
            return None

        try:
            cached_df = pd.read_parquet(cache_file)
            cached_df["date"] = pd.to_datetime(cached_df["date"])

            # Check if cached data covers our date range
            cached_start = cached_df["date"].min()
            cached_end = cached_df["date"].max()

            if cached_start <= start_date and cached_end >= end_date:
                # Filter to requested date range
                mask = (cached_df["date"] >= start_date) & (
                    cached_df["date"] <= end_date
                )
                return cached_df[mask].copy()

        except Exception:
            # If there's any issue with cached data, ignore it
            pass

        return None

    def _cache_data(self, ticker: str, data: pd.DataFrame) -> None:
        """Cache market data to disk."""
        if not self.config.get("settings.cache_market_data", True):
            return

        cache_dir = self.config.get_market_data_path()
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_file = cache_dir / f"{ticker}.parquet"

        try:
            data.to_parquet(cache_file, index=False)
        except Exception as e:
            print(f"⚠️  Warning: Could not cache data for {ticker}: {e}")

    def _get_price_for_date(
        self, ticker: str, target_date: pd.Timestamp
    ) -> float:
        """Get price for ticker on specific date with forward-fill logic."""
        if ticker not in self.market_data:
            raise ValueError(f"No market data available for {ticker}")

        ticker_data = self.market_data[ticker]

        # Try exact date match first
        self.logger.debug("Looking for price of %s on %s", ticker, target_date)
        exact_match = ticker_data[
            ticker_data["date"].dt.date == target_date  # .date()
        ]
        if not exact_match.empty:
            return exact_match["Close"].iloc[0]

        # Forward fill: use most recent price before target date
        self.logger.debug(
            "Forward-filling price for %s on %s", ticker, target_date
        )
        available_data = ticker_data[
            ticker_data["date"].dt.date <= target_date  # .date()
        ]
        if not available_data.empty:
            days_back = (
                target_date.date() - available_data["date"].iloc[-1].date()
            ).days
            if days_back <= 7:  # Only forward-fill up to 7 days
                return available_data["Close"].iloc[-1]
            else:
                print(
                    f"Warning: No recent price data for {ticker} near {target_date}"
                )
                return available_data["Close"].iloc[
                    -1
                ]  # Use it anyway but warn

        # Backward fill: use next available price after target date
        self.logger.debug(
            "Backward-filling price for %s on %s", ticker, target_date
        )
        future_data = ticker_data[
            ticker_data["date"].dt.date > target_date  # .date()
        ]
        if not future_data.empty:
            days_forward = (
                future_data["date"].iloc[0].date() - target_date  # .date()
            ).days
            print(
                f"Warning: Using future price for {ticker} on {target_date} ({days_forward} days forward)"
            )
            return future_data["Close"].iloc[0]

        # If we get here, no data exists at all
        raise ValueError(
            f"No price data available for {ticker} around {target_date}"
        )

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
        self.logger.debug("Calculating full date range for portfolio history")
        start_date = self.transactions["date"].min()
        end_date = max([df["date"].max() for df in self.market_data.values()])

        # Get actual NYSE trading days (excludes weekends AND holidays)
        nyse = mcal.get_calendar("NYSE")
        trading_days = nyse.schedule(start_date=start_date, end_date=end_date)
        date_range = trading_days.index.date  # Convert to date objects

        # Check if all transaction dates are trading days
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
                shares = transaction[
                    "shares"
                ]  # Already negative for SELL transactions

                # Ensure account and ticker exist in our tracking
                if account not in current_holdings:
                    current_holdings[account] = {t: 0.0 for t in tickers}
                if ticker not in current_holdings[account]:
                    current_holdings[account][ticker] = 0.0

                current_holdings[account][ticker] += shares

            # Get market prices for this date
            self.logger.debug(
                "Calculating portfolio value for %s", date
            )  # date.date())
            day_data = {"date": date}
            total_portfolio_value = 0.0
            account_totals = {}

            self.logger.debug("Calculating account-specific positions")
            for account in accounts:
                account_value = 0.0

                for ticker in tickers:
                    shares = current_holdings[account][ticker]

                    try:
                        self.logger.debug(
                            "Getting price for %s on %s", ticker, date
                        )
                        price = self._get_price_for_date(ticker, date)
                    except ValueError as e:
                        self.logger.error(f"Skipping {ticker} on {date}: {e}")
                        price = 0.0  # Only set to 0 if truly no data exists

                    self.logger.debug(
                        "Calculated price for %s on %s: $%.2f",
                        ticker,
                        date,
                        price,
                    )
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
            self.logger.debug("Calculating consolidated positions")
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
                        price = price_data["Close"].iloc[0]
                    else:
                        available_data = ticker_data[
                            ticker_data["date"].dt.date <= date  # date.date()
                        ]
                        if not available_data.empty:
                            price = available_data["Close"].iloc[-1]
                        else:
                            price = 0.0
                else:
                    price = 0.0

                day_data[f"{ticker}_total_shares"] = total_shares
                day_data[f"{ticker}_total_value"] = total_shares * price

            portfolio_data.append(day_data)

        # Convert to DataFrame
        self.logger.debug("Converting portfolio data to DataFrame")
        portfolio_df = pd.DataFrame(portfolio_data)

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
                        ).fillna(0.0)

        # Calculate overall weights
        for ticker in tickers:
            portfolio_df[f"{ticker}_weight"] = (
                portfolio_df[f"{ticker}_total_value"]
                / portfolio_df["total_value"]
            ).fillna(0.0)

        # Calculate daily returns
        portfolio_df["portfolio_return"] = portfolio_df[
            "total_value"
        ].pct_change()

        # Calculate account-specific returns
        for account in accounts:
            account_total_col = f"{account}_total"
            if account_total_col in portfolio_df.columns:
                portfolio_df[f"{account}_return"] = portfolio_df[
                    account_total_col
                ].pct_change()

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
            benchmark_metrics = self._calculate_metrics(
                benchmark_returns, "Benchmark"
            )

        # Calculate relative performance metrics
        relative_metrics = self._calculate_relative_metrics(
            self.portfolio_history["portfolio_return"].dropna(),
            benchmark_returns if self.benchmark_data is not None else None,
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
        benchmark_df["return"] = benchmark_df["Close"].pct_change()

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

        return pd.Series(aligned_returns).dropna()

    def _calculate_metrics(self, returns: pd.Series, name: str) -> Dict:
        """Calculate performance metrics for a return series."""
        if returns.empty:
            return {}

        # Basic statistics
        total_return = (1 + returns).prod() - 1
        annualized_return = (
            1 + returns.mean()
        ) ** 252 - 1  # Assuming daily returns
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility

        # Risk-adjusted metrics
        risk_free_rate = self.config.get("settings.risk_free_rate", 0.02)
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = (
            excess_returns.mean() / returns.std() * np.sqrt(252)
            if returns.std() > 0
            else 0
        )

        # Drawdown analysis
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Additional metrics
        positive_periods = (returns > 0).sum()
        total_periods = len(returns)
        win_rate = positive_periods / total_periods if total_periods > 0 else 0

        return {
            "name": name,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
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

        # Align the series
        min_length = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns.iloc[-min_length:]
        benchmark_returns = benchmark_returns.iloc[-min_length:]

        # Calculate excess returns (alpha)
        excess_returns = portfolio_returns - benchmark_returns

        # Tracking error (volatility of excess returns)
        tracking_error = excess_returns.std() * np.sqrt(252)

        # Information ratio (excess return / tracking error)
        information_ratio = (
            (excess_returns.mean() * 252) / tracking_error
            if tracking_error > 0
            else 0
        )

        # Beta calculation
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

        # Jensen's Alpha (risk-adjusted excess return)
        risk_free_rate = self.config.get("settings.risk_free_rate", 0.02) / 252
        portfolio_excess = portfolio_returns.mean() - risk_free_rate
        benchmark_excess = benchmark_returns.mean() - risk_free_rate
        jensens_alpha = portfolio_excess - (beta * benchmark_excess)
        jensens_alpha_annualized = jensens_alpha * 252

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
                    total_return = (1 + account_returns).prod() - 1
                    annualized_return = (1 + account_returns.mean()) ** 252 - 1
                else:
                    total_return = 0.0
                    annualized_return = 0.0

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
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export portfolio history
        history_file = output_path / "portfolio_history.csv"
        self.portfolio_history.to_csv(history_file, index=False)

        # Export summary metrics
        metrics_data = []
        if self.portfolio_metrics:
            metrics_data.append({**self.portfolio_metrics, "type": "Portfolio"})
        if self.benchmark_metrics:
            metrics_data.append({**self.benchmark_metrics, "type": "Benchmark"})

        if metrics_data:
            metrics_file = output_path / "performance_metrics.csv"
            pd.DataFrame(metrics_data).to_csv(metrics_file, index=False)

        print(f"📁 Results exported to: {output_path}")
        return pd.DataFrame(holdings_data)

    def export_to_csv(self, output_dir: Optional[str] = None) -> str:
        """Export results to CSV files."""
        if output_dir is None:
            output_dir = self.config.get_output_path()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export portfolio history
        history_file = output_path / "portfolio_history.csv"
        self.portfolio_history.to_csv(history_file, index=False)

        # Export summary metrics
        metrics_data = []
        if self.portfolio_metrics:
            metrics_data.append({**self.portfolio_metrics, "type": "Portfolio"})
        if self.benchmark_metrics:
            metrics_data.append({**self.benchmark_metrics, "type": "Benchmark"})

        if metrics_data:
            metrics_file = output_path / "performance_metrics.csv"
            pd.DataFrame(metrics_data).to_csv(metrics_file, index=False)

        print(f"Results exported to: {output_path}")
        return str(output_path)
        return str(output_path)
