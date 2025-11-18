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
from ..core.portfolio_db import PortfolioDatabase
from ..core.results import PerformanceResults
from ..core.symbol_attributes_loader import load_symbol_attributes_from_csv
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

        self.portfolio_db: Optional[PortfolioDatabase] = None
        self.transactions = pd.DataFrame()
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.benchmark_history = pd.DataFrame()
        self.performance_results = PerformanceResults()
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

    def get_database(self) -> Optional[PortfolioDatabase]:
        """Get the normalized portfolio database."""
        return self.portfolio_db

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

        self.transactions = load_validate_transactions(Path(file_path))
        self.logger.info("✅ Loaded %d transactions", len(self.transactions))

        return self.transactions

    def load_symbol_attributes(
        self, csv_path: Optional[str] = None, api_source: Optional[str] = None
    ) -> None:
        """
        Load symbol attributes into database.

        Attributes must be loaded separately from transactions via CSV file or API.
        This should be called after build_portfolio_history() to ensure database exists.

        Args:
            csv_path: Path to attributes CSV file (optional)
            api_source: API source for attributes (optional, not yet implemented)

        Raises:
            ValueError: If portfolio_db not initialized
        """
        if self.portfolio_db is None:
            raise ValueError(
                "Must build portfolio history first using build_portfolio_history()"
            )

        # Use portfolio start date as effective date for attributes
        effective_date = (
            self.start_date if self.start_date else pd.Timestamp.now(tz="UTC")
        )

        if csv_path:
            load_symbol_attributes_from_csv(
                db=self.portfolio_db,
                csv_path=csv_path,
                effective_date=effective_date,
            )
        elif api_source:
            # Future: load from API
            self.logger.warning("API loading not yet implemented")
        else:
            self.logger.info(
                "No attribute source specified. "
                "Attributes are optional but required for attribution analysis."
            )

    def build_portfolio_history(self) -> PortfolioDatabase:
        """
        Build portfolio holdings and values over time.

        Returns:
            PortfolioDatabase: Database with normalized portfolio history
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
        self.portfolio_db = build.build()

        return self.portfolio_db

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
        self.benchmark_history = builder.build()

        if self.benchmark_history.empty:
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
        # Check if database exists and has data
        if self.portfolio_db is None:
            self.logger.debug(
                "Portfolio database not initialized, building now..."
            )
            self.build_portfolio_history()

        if self.portfolio_db is None:
            self.logger.error(
                "❌ Portfolio database is None after build. "
                + "Cannot calculate performance."
            )
            return PerformanceResults()

        # Query database for portfolio data
        portfolio_summary = self.portfolio_db.get_portfolio_summary()
        if portfolio_summary.empty:
            self.logger.error(
                "❌ Portfolio summary is empty. "
                + "Cannot calculate performance."
            )
            return PerformanceResults()

        self.logger.info("📊 Calculating performance metrics...")

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

        # Extract return series from database
        portfolio_mod_dietz_returns = portfolio_summary.set_index("date")[
            "portfolio_mod_dietz_return"
        ].dropna()
        portfolio_twr_returns = portfolio_summary.set_index("date")[
            "portfolio_twr_return"
        ].dropna()

        # Calculate portfolio metrics
        portfolio_metrics = {
            "mod_dietz": calculate_metrics(
                portfolio_mod_dietz_returns,
                "Portfolio (Mod. Dietz)",
                annual_trading_days,
                annual_risk_free_rate,
            ),
            "twr": calculate_metrics(
                portfolio_twr_returns,
                "Portfolio (TWR)",
                annual_trading_days,
                annual_risk_free_rate,
            ),
            "irr": {
                "annualized_return": calculate_irr(
                    portfolio_summary["net_cash_flow"],
                    portfolio_summary["total_value"],
                    self.config,
                )
            },
        }
        benchmark_metrics = {}
        if "benchmark_return" in self.benchmark_history.columns:
            benchmark_metrics = calculate_metrics(
                # Skip first day invested as benchmark return is calculated
                # as period-over-period change
                self.benchmark_history["benchmark_return"][1:],
                "Benchmark",
                annual_trading_days,
                annual_risk_free_rate,
            )

        relative_metrics = {}
        if benchmark_metrics:
            relative_metrics = calculate_relative_metrics(
                # Skip first day invested as benchmark return is calculated
                # as period-over-period change
                portfolio_twr_returns[1:],
                self.benchmark_history["benchmark_return"][1:],
                annual_trading_days,
                annual_risk_free_rate,
            )

        # Calculate performance attribution
        self.logger.info("📊 Calculating performance attribution...")
        attrib_calculator = AttributionCalculator(
            portfolio_db=self.portfolio_db,
        )

        # Calculate attribution by holding and account
        holding_attribution = attrib_calculator.calculate(group_by="symbol")
        account_attribution = attrib_calculator.calculate(group_by="account")

        # Calculate for all discovered factor columns
        factor_attributions = {}
        # Get factor columns from database attributes
        if self.portfolio_db:
            attributes_df = self.portfolio_db.get_symbol_attributes()
            if not attributes_df.empty:
                # Standard attribute columns in database
                factor_columns = [
                    col
                    for col in [
                        "asset_class",
                        "geography",
                        "region",
                        "sector",
                        "style",
                        "market_cap",
                        "fund_type",
                    ]
                    if col in attributes_df.columns
                    and not attributes_df[col].isna().all()
                ]
                for factor in factor_columns:
                    self.logger.info(
                        "Calculating attribution for factor: %s", factor
                    )
                    factor_attributions[factor] = attrib_calculator.calculate(
                        group_by=factor
                    )

        # Calculate Brinson-Fachler attribution
        brinson_summary = None
        selection_drilldown = None
        if (
            self.config.get("analysis.attribution_analysis.enabled", False)
            and self.benchmark_history is not None
            and not self.benchmark_history.empty
        ):
            self.logger.info("📊 Running attribution analysis...")

            if (
                self.config.get(
                    "analysis.attribution_analysis.method", "Brinson"
                )
                == "Brinson"
            ):
                self.logger.info("Calculating Brinson attribution...")
                brinson_calculator = BrinsonAttributionCalculator(
                    benchmark_history=self.benchmark_history,
                    portfolio_db=self.portfolio_db,
                )
                group_by = self.config.get(
                    "analysis.attribution_analysis.transaction_groups",
                    ["asset_class"],
                )
                if isinstance(group_by, Dict):
                    group_by = group_by.get("value", ["asset_class"])
                if group_by is None or not isinstance(group_by, list):
                    group_by = ["asset_class"]
                    self.logger.warning(
                        "Invalid group_by for Brinson attribution. Using default ['asset_class']."
                    )
                if self.start_date is None or self.end_date is None:
                    raise ValueError(
                        "Start date and end date must both be set"
                    )

                # Validate that attributes exist in database
                valid_attributes = [
                    "asset_class",
                    "geography",
                    "region",
                    "sector",
                    "style",
                    "market_cap",
                    "fund_type",
                ]

                for group in group_by:
                    if group not in valid_attributes:
                        self.logger.error(
                            "Grouping attribute '%s' not valid. Must be one of: %s. "
                            "Skipping Brinson attribution.",
                            group,
                            ", ".join(valid_attributes),
                        )
                    else:
                        brinson_summary = {}
                        selection_drilldown = {}
                        for group in group_by:
                            self.logger.info(
                                " - Grouping by attribute: %s", group
                            )
                            (
                                brinson_summary[group],
                                selection_drilldown[group],
                            ) = brinson_calculator.calculate(group)
                            self.logger.info(
                                "✅ Brinson attribution analysis complete!"
                            )

        self.performance_results = PerformanceResults(
            transactions=self.transactions,
            portfolio_metrics=portfolio_metrics,
            benchmark_metrics=benchmark_metrics,
            relative_metrics=relative_metrics,
            holding_attribution=holding_attribution,
            account_attribution=account_attribution,
            factor_attributions=factor_attributions,
            benchmark_history=self.benchmark_history,
            brinson_summary=brinson_summary,
            selection_drilldown=selection_drilldown,
            config=self.config,
            portfolio_db=self.portfolio_db,
        )

        self.logger.info("✅ Performance analysis complete!")
        return self.performance_results
