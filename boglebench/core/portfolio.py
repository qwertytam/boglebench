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
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from ..core.attribution import AttributionCalculator
from ..core.brinson_attribution import BrinsonAttributionCalculator
from ..core.composite_benchmark import CompositeBenchmarkBuilder
from ..core.constants import DateAndTimeConstants, Defaults
from ..core.dates import AnalysisPeriod
from ..core.dividend_processor import DividendProcessor
from ..core.history_builder import PortfolioHistoryBuilder
from ..core.market_data import MarketDataProvider
from ..core.metrics import (
    calculate_irr,
    calculate_metrics,
    calculate_relative_metrics,
)
from ..core.results import PerformanceResults
from ..core.transaction_loader import load_validate_transactions
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

        if config_path:
            config_file = Path(config_path).expanduser()
            if config_file.exists():
                WorkspaceContext.discover_workspace(config_file.parent)

        setup_logging()
        self.logger = get_logger("core.portfolio")

        self.transactions = pd.DataFrame()
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.portfolio_history = pd.DataFrame()
        self.benchmark_data = pd.DataFrame()
        self.performance_results = PerformanceResults()
        self.attrib_group_cols: list[str] = []
        self.start_date: Optional[pd.Timestamp] = None
        self.end_date: Optional[pd.Timestamp] = None

        self.config = ConfigManager(config_path)

        # Suppress warnings for cleaner output
        warnings.filterwarnings("ignore", category=FutureWarning)
        self.logger.info("BogleBench analyzer initialized")

    @property
    def config(self) -> ConfigManager:
        """Get the configuration manager."""
        return self._config

    @config.setter
    def config(self, new_config: ConfigManager):
        """Set a new config and re-initialize dependent objects."""
        if not isinstance(new_config, ConfigManager):
            raise TypeError("config must be an instance of ConfigManager")
        self._config = new_config
        self.logger.info("Configuration set. Re-initializing settings...")
        self._initialize_from_config()

    def _initialize_from_config(self):
        """Initialize or update settings from the current config object."""

        api_key = self.config.get(
            "api.alpha_vantage_key", Defaults.DEFAULT_API_KEY
        )
        if isinstance(api_key, dict):
            api_key = api_key.get("value", Defaults.DEFAULT_API_KEY)
        if api_key is None or str(api_key).strip() == "":
            api_key = Defaults.DEFAULT_API_KEY

        cache_dir = self.config.get_market_data_path()

        cache_enabled = self.config.get("settings.cache_market_data", True)
        if isinstance(cache_enabled, dict):
            cache_enabled = cache_enabled.get("value", True)
        if cache_enabled is None:
            cache_enabled = True

        # pylint: disable-next=attribute-defined-outside-init
        self.market_data_provider = MarketDataProvider(
            api_key=api_key,
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
        )

    def load_transactions(
        self, file_path: Optional[Union[str, Path]] = None
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
            file_path = str(self.config.get_transactions_file_path())
        self.logger.info("📄 Loading transactions from: %s", file_path)

        self.transactions, self.attrib_group_cols = load_validate_transactions(
            Path(file_path)
        )
        self.logger.info("✅ Loaded %d transactions", len(self.transactions))

        return self.transactions

    def build_portfolio_history(self) -> pd.DataFrame:
        """
        Build portfolio holdings and values over time.

        Returns:
            DataFrame with daily portfolio holdings, values, and weights
        """
        if self.transactions.empty:
            raise ValueError(
                "Must load transactions first using load_transactions()"
            )

        period = AnalysisPeriod(self.config, self.transactions)
        if not period.start_date or not period.end_date:
            raise ValueError("Could not determine valid start and end dates")

        self.start_date = period.start_date
        self.end_date = period.end_date
        self.logger.info(
            "📅 Analysis period: %s to %s",
            self.start_date,
            self.end_date,
        )
        self._fetch_market_data(period.start_date, period.end_date)

        processor = DividendProcessor(
            self.config,
            self.transactions,
            self.market_data,
            start_date=period.start_date,
            end_date=period.end_date,
        )
        self.transactions = processor.run()

        build = PortfolioHistoryBuilder(
            config=self.config,
            transactions=self.transactions,
            market_data=self.market_data,
            start_date=period.start_date,
            end_date=period.end_date,
        )
        self.portfolio_history = build.build()

        return self.portfolio_history

    def _fetch_market_data(
        self, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> None:
        """Fetch and prepare all required market data.

        Args:
            start_date: Start date for market data
            end_date: End date for market data
        Returns:
            None

        """
        portfolio_symbols = self.transactions["symbol"].unique().tolist()

        benchmark_components = self.config.get_benchmark_components()
        benchmark_symbols = [comp["symbol"] for comp in benchmark_components]

        all_symbols = list(set(portfolio_symbols + benchmark_symbols))
        self.logger.debug(
            "Fetching market data for %d unique symbols: %s",
            len(all_symbols),
            all_symbols,
        )
        self.market_data = self.market_data_provider.get_market_data(
            symbols=all_symbols,
            start_date=start_date,
            end_date=end_date,
        )

        self.logger.info("Building composite benchmark history...")
        builder = CompositeBenchmarkBuilder(
            config=self.config,
            market_data=self.market_data,
            start_date=start_date,
            end_date=end_date,
        )
        self.benchmark_data = builder.build()

        if self.benchmark_data.empty:
            self.logger.warning(
                "Composite benchmark data is empty or not found in market data."
                " No benchmark comparison will be available."
            )

    def calculate_performance(self) -> "PerformanceResults":
        """
        Calculate comprehensive performance metrics for the portfolio and
        benchmark.

        Returns:
            PerformanceResults object containing all metrics and analysis
        """
        if self.portfolio_history.empty:
            self.build_portfolio_history()

        if self.portfolio_history.empty:
            self.logger.error(
                "❌ Portfolio history is empty after build. "
                + "Cannot calculate performance."
            )

        self.logger.info("📊 Calculating performance metrics...")

        # Align benchmark before calculating metrics
        self._align_benchmark_returns()

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
            "analysis.annual_risk_free_rate", Defaults.DEFAULT_RISK_FREE_RATE
        )
        if isinstance(annual_risk_free_rate, Dict):
            annual_risk_free_rate = annual_risk_free_rate.get(
                "value", Defaults.DEFAULT_RISK_FREE_RATE
            )
        if annual_risk_free_rate is None:
            annual_risk_free_rate = Defaults.DEFAULT_RISK_FREE_RATE

        # Calculate portfolio metrics
        portfolio_metrics = {
            "mod_dietz": calculate_metrics(
                self.portfolio_history["portfolio_daily_return_mod_dietz"],
                "Portfolio (Mod. Dietz)",
                annual_trading_days,
                annual_risk_free_rate,
            ),
            "twr": calculate_metrics(
                self.portfolio_history["portfolio_daily_return_twr"],
                "Portfolio (TWR)",
                annual_trading_days,
                annual_risk_free_rate,
            ),
            "irr": {
                "annualized_return": calculate_irr(
                    self.portfolio_history, self.config
                )
            },
        }
        benchmark_metrics = {}
        if "benchmark_returns" in self.portfolio_history.columns:
            benchmark_metrics = calculate_metrics(
                # Skip first day invested as benchmark return is calculated
                # as period-over-period change
                self.portfolio_history["benchmark_returns"][1:],
                "Benchmark",
                annual_trading_days,
                annual_risk_free_rate,
            )

        relative_metrics = {}
        if benchmark_metrics:
            relative_metrics = calculate_relative_metrics(
                # Skip first day invested as benchmark return is calculated
                # as period-over-period change
                self.portfolio_history["portfolio_daily_return_twr"][1:],
                self.portfolio_history["benchmark_returns"][1:],
                annual_trading_days,
                annual_risk_free_rate,
            )

        # Calculate performance attribution
        self.logger.info("📊 Calculating performance attribution...")
        attrib_calculator = AttributionCalculator(
            portfolio_history=self.portfolio_history,
            transactions=self.transactions,
            attrib_group_cols=self.attrib_group_cols,
        )

        # Calculate attribution by holding and account
        holding_attribution = attrib_calculator.calculate(group_by="symbol")
        account_attribution = attrib_calculator.calculate(group_by="account")

        # Calculate for all discovered factor columns
        factor_attributions = {}
        factor_columns = sorted(list(set(self.attrib_group_cols)))
        for factor in factor_columns:
            self.logger.info("Calculating attribution for factor: %s", factor)
            factor_attributions[factor] = attrib_calculator.calculate(
                group_by=factor
            )

        # Calculate Brinson-Fachler attribution if 'sector' is available
        brinson_summary = None
        selection_drilldown = None
        if self.config.get("reporting.show_brinson_attribution", False):
            self.logger.info("📊 Running Brinson attribution analysis...")
            brinson_calculator = BrinsonAttributionCalculator(
                config=self.config,
                portfolio_history=self.portfolio_history,
                transactions=self.transactions,
                market_data=self.market_data,
            )
            group_by = self.config.get(
                "reporting.attribution_group_by", "asset_class"
            )
            if isinstance(group_by, Dict):
                group_by = group_by.get("value", "asset_class")
            if group_by is None or not isinstance(group_by, str):
                group_by = "asset_class"
                self.logger.warning(
                    "Invalid group_by for Brinson attribution. Using default 'asset_class'."
                )
            if self.start_date is None or self.end_date is None:
                raise ValueError("Start date and end date must both be set")

            if group_by not in self.transactions.columns:
                self.logger.error(
                    "Grouping column '%s' not found in transactions. "
                    "Skipping Brinson attribution.",
                    group_by,
                )
            else:
                brinson_summary, selection_drilldown = (
                    brinson_calculator.calculate(
                        group_by, self.start_date, self.end_date
                    )
                )
                self.logger.info("✅ Brinson attribution analysis complete!")

        self.performance_results = PerformanceResults(
            transactions=self.transactions,
            portfolio_metrics=portfolio_metrics,
            benchmark_metrics=benchmark_metrics,
            relative_metrics=relative_metrics,
            portfolio_history=self.portfolio_history,
            holding_attribution=holding_attribution,
            account_attribution=account_attribution,
            factor_attributions=factor_attributions,
            brinson_summary=brinson_summary,
            selection_drilldown=selection_drilldown,
            config=self.config,
        )

        self.logger.info("✅ Performance analysis complete!")
        return self.performance_results

    def _align_benchmark_returns(self) -> None:
        """Align benchmark returns with portfolio dates."""
        if self.benchmark_data.empty:
            self.logger.warning(
                "No benchmark data available to align returns."
            )
            return

        # Convert benchmark data to returns
        benchmark_df = self.benchmark_data.copy()
        benchmark_df["date"] = pd.to_datetime(
            benchmark_df["date"], utc=True
        ).dt.normalize()
        benchmark_df = benchmark_df.sort_values("date")

        # Calculate period on period returns
        # Use adjusted close prices for benchmark returns
        benchmark_df["benchmark_returns"] = benchmark_df[
            "adj_close"
        ].pct_change()

        # Use merge_asof for robust alignment
        self.portfolio_history = pd.merge_asof(
            self.portfolio_history.sort_values("date"),
            benchmark_df[["date", "benchmark_returns"]].sort_values("date"),
            on="date",
            direction="backward",
        )

        self.portfolio_history["benchmark_returns"] = self.portfolio_history[
            "benchmark_returns"
        ].fillna(0)
