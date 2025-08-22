"""
Core BogleBench portfolio analyzer for performance analysis and benchmarking.

This module implements the main BogleBenchAnalyzer class that handles:
- Transaction data loading and processing
- Market data acquisition
- Portfolio construction over time
- Performance metrics calculation
- Benchmark comparison

Following John Bogle's investment principles of simplicity, low costs, and long-term focus.
"""

import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yfinance as yf

from ..utils.config import ConfigManager


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
            config_path: Path to configuration file. If None, uses default locations.
        """
        self.config = ConfigManager(config_path)
        self.transactions = None
        self.market_data = {}
        self.portfolio_history = None
        self.benchmark_data = None
        self.performance_results = None

        # Suppress yfinance warnings for cleaner output
        warnings.filterwarnings("ignore", category=FutureWarning)

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
        optional_columns = ["account"]  # Broker account name

        missing_columns = [
            col for col in required_columns if col not in df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Add account column if not present (for backward compatibility)
        if "account" not in df.columns:
            df["account"] = "Default"
            print("ℹ️  No 'account' column found. Added default account name.")

        # Clean and validate data
        df = self._clean_transaction_data(df)

        # Store processed transactions
        self.transactions = df

        print(
            f"✅ Loaded {len(df)} transactions for {df['ticker'].nunique()} unique assets"
        )
        print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"🏦 Accounts: {', '.join(df['account'].unique())}")
        print(
            f"💰 Total invested: ${df[df['total_value'] > 0]['total_value'].sum():,.2f}"
        )

        return df

    def _clean_transaction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate transaction data."""
        # Make a copy to avoid modifying original
        df = df.copy()

        # Convert date column
        try:
            df["date"] = pd.to_datetime(df["date"])
        except Exception as e:
            raise ValueError(f"Error parsing dates: {e}")

        # Clean ticker symbols (uppercase, strip whitespace)
        df["ticker"] = df["ticker"].str.upper().str.strip()

        # Clean account names (strip whitespace, title case)
        df["account"] = df["account"].str.strip().str.title()

        # Validate transaction types
        valid_types = ["BUY", "SELL"]
        df["transaction_type"] = df["transaction_type"].str.upper().str.strip()
        invalid_types = df[~df["transaction_type"].isin(valid_types)]
        if not invalid_types.empty:
            raise ValueError(
                f"Invalid transaction types found: {invalid_types['transaction_type'].unique()}"
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

    def fetch_market_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch daily market data for all assets in the portfolio plus benchmark.

        Args:
            start_date: Start date for market data (YYYY-MM-DD). If None, uses first transaction date.
            end_date: End date for market data (YYYY-MM-DD). If None, uses today.
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
            start_date = self.transactions["date"].min() - timedelta(
                days=30
            )  # Buffer for calculations
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
                    market_data[ticker] = cached_data
                    continue

                # Download from yfinance
                stock = yf.Ticker(ticker)
                hist = stock.history(
                    start=start_date, end=end_date, auto_adjust=True
                )

                if hist.empty:
                    failed_tickers.append(ticker)
                    continue

                # Clean up the data
                hist = hist[["Open", "High", "Low", "Close", "Volume"]]
                hist.index.name = "date"
                hist = hist.reset_index()

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

        print("🏗️  Building portfolio history...")

        # Get all unique tickers and accounts
        tickers = self.transactions["ticker"].unique()
        accounts = self.transactions["account"].unique()

        # Initialize portfolio tracking by account and ticker
        portfolio_data = []
        # Track holdings by account and ticker: {account: {ticker: shares}}
        current_holdings = {
            account: {ticker: 0.0 for ticker in tickers} for account in accounts
        }

        for date in date_range:
            # Process any transactions on this date
            day_transactions = self.transactions[
                self.transactions["date"].date() == date.date()
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
            day_data = {"date": date}
            total_portfolio_value = 0.0
            account_totals = {}

            for account in accounts:
                account_value = 0.0

                for ticker in tickers:
                    shares = current_holdings[account][ticker]

                    # Get price for this date
                    if ticker in self.market_data:
                        ticker_data = self.market_data[ticker]
                        price_data = ticker_data[
                            ticker_data["date"].dt.date == date.date()
                        ]

                        if not price_data.empty:
                            price = price_data["Close"].iloc[0]
                        else:
                            # Use forward fill if no data for this specific date
                            available_data = ticker_data[
                                ticker_data["date"].dt.date <= date.date()
                            ]
                            if not available_data.empty:
                                price = available_data["Close"].iloc[-1]
                            else:
                                price = 0.0
                    else:
                        price = 0.0

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
                        price = price_data["Close"].iloc[0]
                    else:
                        available_data = ticker_data[
                            ticker_data["date"].dt.date <= date.date()
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

        print(f"✅ Portfolio history built: {len(portfolio_df)} days")
        print(f"🏦 Tracking {len(accounts)} accounts: {', '.join(accounts)}")
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
                benchmark_df["date"].dt.date == date.date()
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

        return pd.DataFrame(holdings_data)
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

        print(f"📁 Results exported to: {output_path}")
        return str(output_path)
