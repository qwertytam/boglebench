"""
Performance results container and reporting.

This module provides a container class for portfolio performance analysis results
and methods for generating summary reports, accessing detailed metrics, and
exporting data for further analysis or visualization.
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..utils.config import ConfigManager
from ..utils.logging_config import get_logger
from .portfolio_db import PortfolioDatabase


class PerformanceResults:
    """Container for portfolio performance analysis results."""

    def __init__(
        self,
        transactions: Optional[pd.DataFrame] = None,
        portfolio_metrics: Optional[Dict] = None,
        benchmark_metrics: Optional[Dict] = None,
        relative_metrics: Optional[Dict] = None,
        portfolio_history: Optional[pd.DataFrame] = None,
        benchmark_history: Optional[pd.DataFrame] = None,
        holding_attribution: Optional[pd.DataFrame] = None,
        account_attribution: Optional[pd.DataFrame] = None,
        factor_attributions: Optional[Dict] = None,
        brinson_summary: Optional[Dict[str, pd.DataFrame]] = None,
        selection_drilldown: Optional[
            Dict[str, Dict[str, pd.DataFrame]]
        ] = None,
        config: Optional[ConfigManager] = None,
        portfolio_db: Optional[PortfolioDatabase] = None,  # NEW
    ):
        """
        Initialize the PerformanceResults container.

        Args:
            transactions: DataFrame containing transaction data
            portfolio_metrics: Dictionary of portfolio performance metrics
            benchmark_metrics: Dictionary of benchmark performance metrics
            relative_metrics: Dictionary of relative performance metrics
            portfolio_history: DataFrame with historical portfolio data
            benchmark_history: DataFrame with historical benchmark data
            holding_attribution: DataFrame with attribution by holding
            account_attribution: DataFrame with attribution by account
            factor_attributions: Dictionary of attribution by factors
            brinson_summary: Dictionary of Brinson attribution summaries
            selection_drilldown: Detailed selection attribution by groups
            config: ConfigManager instance
            portfolio_db: PortfolioDatabase instance
        """
        self.transactions = transactions
        self.portfolio_metrics = portfolio_metrics
        self.benchmark_metrics = benchmark_metrics
        self.relative_metrics = relative_metrics
        self.portfolio_history = portfolio_history
        self.benchmark_history = benchmark_history
        self.holding_attribution = holding_attribution
        self.account_attribution = account_attribution
        self.factor_attributions = factor_attributions
        self.brinson_summary = brinson_summary
        self.selection_drilldown = selection_drilldown
        self.config = config
        self.portfolio_db = portfolio_db  # NEW

        self.logger = get_logger("core.results")

    def summary(self) -> str:
        """Generate a summary report of the performance analysis."""
        lines = []
        lines.append("\n")
        lines.append("=" * 80)
        lines.append("🎯 BOGLEBENCH PERFORMANCE ANALYSIS")
        lines.append("   'Stay the course' - John C. Bogle")
        lines.append("=" * 80)

        # Portfolio metrics
        if self.portfolio_metrics:
            p = self.portfolio_metrics
            lines.append("\n📊 PORTFOLIO PERFORMANCE\n")
            lines.append(
                "  Return Methods       Mod. Dietz     TWR        IRR"
            )
            lines.append(
                f"  Total Return:        "
                f"{p['mod_dietz']['total_return']:>+8.2%}    "
                f"{p['twr']['total_return']:>+8.2%}"
            )
            lines.append(
                f"  Annualized Return:   "
                f"{p['mod_dietz']['annualized_return']:>+8.2%}    "
                f"{p['twr']['annualized_return']:>+8.2%}   "
                f"{p['irr']['annualized_return']:>+8.2%}"
            )
            lines.append(
                f"  Volatility:          "
                f"{p['mod_dietz']['volatility']:>+8.2%}    "
                f"{p['twr']['volatility']:>+8.2%}"
            )
            lines.append(
                f"  Sharpe Ratio:        "
                f"{p['mod_dietz']['sharpe_ratio']:>+8.3f}    "
                f"{p['twr']['sharpe_ratio']:>+8.3f}"
            )
            lines.append(
                f"  Max Drawdown:        "
                f"{p['mod_dietz']['max_drawdown']:>+8.2%}    "
                f"{p['twr']['max_drawdown']:>+8.2%}"
            )

        # Benchmark metrics
        if self.benchmark_metrics:
            b = self.benchmark_metrics
            lines.append("\n📈 BENCHMARK PERFORMANCE\n")
            lines.append(f"  Total Return:        {b['total_return']:>+8.2%}")
            lines.append(
                f"  Annualized Return:   {b['annualized_return']:>+8.2%}"
            )
            lines.append(f"  Volatility:          {b['volatility']:>+8.2%}")
            lines.append(f"  Sharpe Ratio:        {b['sharpe_ratio']:>+8.3f}")
            lines.append(f"  Max Drawdown:        {b['max_drawdown']:>+8.2%}")

        # Relative metrics
        if self.relative_metrics:
            r = self.relative_metrics
            lines.append("\n📊 RELATIVE PERFORMANCE\n")
            lines.append(f"  Excess Return:       {r['excess_return']:>+8.2%}")
            lines.append(
                f"  Tracking Error:      {r['tracking_error']:>+8.2%}"
            )
            lines.append(
                f"  Information Ratio:   {r['information_ratio']:>+8.3f}"
            )
            lines.append(f"  Beta:                {r['beta']:>+8.3f}")
            lines.append(f"  Alpha:               {r['alpha']:>+8.2%}")
            lines.append(f"  Correlation:         {r['correlation']:>+8.3f}")

        # Attribution summaries
        if (
            self.holding_attribution is not None
            and not self.holding_attribution.empty
        ):
            lines.append("\n📋 HOLDING ATTRIBUTION (Top 5 Contributors)\n")
            top_holdings = self.holding_attribution.nlargest(
                5, "Contribution to Portfolio Return"
            )
            for _, row in top_holdings.iterrows():
                lines.append(
                    f"  {row.name:8s}  Weight: {row['Avg. Weight']:>6.1%}  "
                    f"Return: {row['Return (TWR)']:>+7.2%}  "
                    f"Contribution: {row['Contribution to Portfolio Return']:>+7.2%}"
                )

        if (
            self.account_attribution is not None
            and not self.account_attribution.empty
        ):
            lines.append("\n🏦 ACCOUNT ATTRIBUTION\n")
            for _, row in self.account_attribution.iterrows():
                lines.append(
                    f"  {row.name:15s}  Weight: {row['Avg. Weight']:>6.1%}  "
                    f"Return: {row['Return (TWR)']:>+7.2%}  "
                    f"Contribution: {row['Contribution to Portfolio Return']:>+7.2%}"
                )

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)

    def get_portfolio_returns(self) -> pd.Series:
        """
        Get portfolio return series.

        Returns:
            Series with date index and daily returns
        """
        # Try database first
        if self.portfolio_db is not None:
            df = self.portfolio_db.get_portfolio_summary()
            if not df.empty:
                return df.set_index("date")[
                    "portfolio_mod_dietz_return"
                ].dropna()

        # Fallback to DataFrame
        if self.portfolio_history is None:
            return pd.Series(dtype=float)

        return self.portfolio_history[
            "portfolio_daily_return_mod_dietz"
        ].dropna()

    def get_cumulative_returns(self) -> pd.Series:
        """
        Get cumulative portfolio returns.

        Returns:
            Series with date index and cumulative returns
        """
        returns = self.get_portfolio_returns()
        returns = (1 + returns).cumprod()
        return returns - 1

    def get_account_summary(self) -> pd.DataFrame:
        """
        Get summary of all accounts.

        Returns:
            DataFrame with account, current_value, weight_of_portfolio
        """
        # Try database first
        if self.portfolio_db is not None:
            df = self.portfolio_db.get_account_data()
            if df.empty:
                return pd.DataFrame()

            latest_date = df["date"].max()
            latest = df[df["date"] == latest_date]

            summary = latest[["account", "total_value", "weight"]].rename(
                columns={
                    "total_value": "current_value",
                    "weight": "weight_of_portfolio",
                }
            )

            return summary.sort_values("current_value", ascending=False)

        # Fallback to DataFrame
        if self.portfolio_history is None or self.portfolio_history.empty:
            return pd.DataFrame()

        latest_data = self.portfolio_history.iloc[-1]
        accounts = self._extract_accounts_from_columns(
            self.portfolio_history.columns
        )

        account_data = []
        for account in accounts:
            total_value_col = f"{account}_total_value"
            weight_col = f"{account}_weight"

            if (
                total_value_col in self.portfolio_history.columns
                and weight_col in self.portfolio_history.columns
            ):
                account_data.append(
                    {
                        "account": account,
                        "current_value": latest_data[total_value_col],
                        "weight_of_portfolio": latest_data[weight_col],
                    }
                )

        return pd.DataFrame(account_data).sort_values(
            "current_value", ascending=False
        )

    def get_account_holdings(
        self, account_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get current holdings for an account.

        Args:
            account_name: Name of account (if None, returns all accounts)

        Returns:
            DataFrame with account, symbol, shares, value, weight
        """
        # Try database first
        if self.portfolio_db is not None:
            holdings_df = self.portfolio_db.get_latest_holdings(
                account=account_name
            )
            if not holdings_df.empty:
                holdings_df = holdings_df[
                    ["account", "symbol", "quantity", "value", "weight"]
                ].rename(columns={"quantity": "shares"})
            return holdings_df

        # Fallback to DataFrame
        if self.portfolio_history is None:
            return pd.DataFrame()

        latest_data = self.portfolio_history.iloc[-1]

        accounts = [account_name] if account_name else []
        holdings_data = []

        for account in accounts:
            # Find all symbol columns for this account
            symbol_cols = [
                col
                for col in self.portfolio_history.columns
                if col.startswith(f"{account}_") and col.endswith("_quantity")
            ]

            for col in symbol_cols:
                symbol = col.replace(f"{account}_", "").replace(
                    "_quantity", ""
                )
                shares = latest_data[col]

                if shares != 0:  # Only include non-zero holdings
                    value_col = f"{account}_{symbol}_value"
                    weight_col = f"{account}_{symbol}_weight"

                    holdings_data.append(
                        {
                            "account": account,
                            "symbol": symbol,
                            "shares": shares,
                            "value": (
                                latest_data[value_col]
                                if value_col in self.portfolio_history.columns
                                else 0
                            ),
                            "weight": (
                                latest_data[weight_col]
                                if weight_col in self.portfolio_history.columns
                                else 0
                            ),
                        }
                    )

        df = pd.DataFrame(holdings_data)
        if not df.empty:
            df = df.sort_values("value", ascending=False)
        return df

    def get_holdings_history(
        self,
        account: Optional[str] = None,
        symbol: Optional[str] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Get holdings history over time.

        Args:
            account: Filter by account name
            symbol: Filter by symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            DataFrame with date, account, symbol, quantity, value, weight
        """
        # Try database first
        if self.portfolio_db is not None:
            return self.portfolio_db.get_holdings(
                account=account,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
            )

        # Fallback to DataFrame (complex extraction)
        if self.portfolio_history is None or self.portfolio_history.empty:
            return pd.DataFrame()

        # This is complex for wide format - recommend using database
        self.logger.warning(
            "Holdings history from wide DataFrame is limited. "
            "Consider using portfolio_db.get_holdings() for full functionality."
        )
        return pd.DataFrame()

    def get_allocation_by_attribute(
        self,
        attribute: str,
        date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Get portfolio allocation by attribute (geography, sector, etc.).

        Args:
            attribute: Attribute name ('geography', 'sector', 'style', etc.)
            date: Specific date (default: latest)

        Returns:
            DataFrame with category, total_weight, total_value
        """
        if self.portfolio_db is None:
            self.logger.warning(
                "Allocation by attribute requires database. "
                "Ensure portfolio_db is set."
            )
            return pd.DataFrame()

        return self.portfolio_db.get_allocation_by_attribute(
            attribute=attribute,
            date=date,
        )

    def get_allocation_time_series(
        self,
        attribute: str,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Get portfolio allocation over time by attribute.

        Args:
            attribute: Attribute name ('geography', 'sector', etc.)
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with date, category, total_weight, total_value
        """
        if self.portfolio_db is None:
            self.logger.warning(
                "Allocation time series requires database. "
                "Ensure portfolio_db is set."
            )
            return pd.DataFrame()

        return self.portfolio_db.get_allocation_time_series(
            attribute=attribute,
            start_date=start_date,
            end_date=end_date,
        )

    def get_performance_by_attribute(
        self,
        attribute: str,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Get performance metrics grouped by attribute.

        Args:
            attribute: Attribute name ('geography', 'sector', etc.)
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            DataFrame with category, avg_weight, avg_daily_return, volatility
        """
        if self.portfolio_db is None:
            self.logger.warning(
                "Performance by attribute requires database. "
                "Ensure portfolio_db is set."
            )
            return pd.DataFrame()

        return self.portfolio_db.get_performance_by_attribute(
            attribute=attribute,
            start_date=start_date,
            end_date=end_date,
        )

    def get_symbol_data(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Get symbol-level data over time.

        Args:
            symbol: Filter by symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with date, symbol, price, total_quantity, total_value, weight, returns
        """
        if self.portfolio_db is not None:
            return self.portfolio_db.get_symbol_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
            )

        # Fallback to DataFrame (limited)
        if self.portfolio_history is None or self.portfolio_history.empty:
            return pd.DataFrame()

        self.logger.warning(
            "Symbol data from wide DataFrame is limited. "
            "Consider using portfolio_db.get_symbol_data() for full functionality."
        )
        return pd.DataFrame()

    def _extract_accounts_from_columns(self, columns) -> list:
        """Extract account names from column names."""
        accounts = set()
        for col in columns:
            if "_total_value" in col:
                account = col.replace("_total_value", "")
                # Exclude symbol-level columns
                if account not in ["total"]:
                    accounts.add(account)
        return sorted(list(accounts))

    def export_to_csv(
        self,
        output_dir: Optional[str] = None,
        prefix: str = "boglebench",
    ) -> Path:
        """
        Export results to CSV files.

        Args:
            output_dir: Directory to export files to. If None, uses config.
            prefix: Prefix for output files

        Returns:
            Path: Directory containing exported files
        """
        if output_dir is None:
            if self.config:
                output_dir = str(self.config.get_output_path())
            else:
                output_dir = "."

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        # Export portfolio history
        if self.portfolio_db is not None:
            # Export from database (normalized format)
            self._export_from_database(output_path, prefix, timestamp)
        elif self.portfolio_history is not None:
            # Export legacy DataFrame
            file_path = (
                output_path / f"{prefix}_portfolio_history_{timestamp}.csv"
            )
            self.portfolio_history.to_csv(file_path, index=False)
            self.logger.info("Exported portfolio history to: %s", file_path)

        # Export metrics
        if self.portfolio_metrics:
            metrics_data = []
            for method, metrics in self.portfolio_metrics.items():
                for metric, value in metrics.items():
                    metrics_data.append(
                        {
                            "method": method,
                            "metric": metric,
                            "value": value,
                        }
                    )
            metrics_df = pd.DataFrame(metrics_data)
            file_path = output_path / f"{prefix}_metrics_{timestamp}.csv"
            metrics_df.to_csv(file_path, index=False)
            self.logger.info("Exported metrics to: %s", file_path)

        # Export attributions
        if (
            self.holding_attribution is not None
            and not self.holding_attribution.empty
        ):
            file_path = (
                output_path / f"{prefix}_holding_attribution_{timestamp}.csv"
            )
            self.holding_attribution.to_csv(file_path)
            self.logger.info("Exported holding attribution to: %s", file_path)

        if (
            self.account_attribution is not None
            and not self.account_attribution.empty
        ):
            file_path = (
                output_path / f"{prefix}_account_attribution_{timestamp}.csv"
            )
            self.account_attribution.to_csv(file_path)
            self.logger.info("Exported account attribution to: %s", file_path)

        self.logger.info("📁 Results exported to: %s", output_path)
        return output_path

    def _export_from_database(
        self, output_path: Path, prefix: str, timestamp: str
    ) -> None:
        """Export data from normalized database."""
        if self.portfolio_db is None:
            return

        # Export portfolio summary
        portfolio_summary = self.portfolio_db.get_portfolio_summary()
        if not portfolio_summary.empty:
            file_path = (
                output_path / f"{prefix}_portfolio_summary_{timestamp}.csv"
            )
            portfolio_summary.to_csv(file_path, index=False)
            self.logger.info("Exported portfolio summary to: %s", file_path)

        # Export account data
        account_data = self.portfolio_db.get_account_data()
        if not account_data.empty:
            file_path = output_path / f"{prefix}_account_data_{timestamp}.csv"
            account_data.to_csv(file_path, index=False)
            self.logger.info("Exported account data to: %s", file_path)

        # Export holdings
        holdings = self.portfolio_db.get_holdings()
        if not holdings.empty:
            file_path = output_path / f"{prefix}_holdings_{timestamp}.csv"
            holdings.to_csv(file_path, index=False)
            self.logger.info("Exported holdings to: %s", file_path)

        # Export symbol data
        symbol_data = self.portfolio_db.get_symbol_data()
        if not symbol_data.empty:
            file_path = output_path / f"{prefix}_symbol_data_{timestamp}.csv"
            symbol_data.to_csv(file_path, index=False)
            self.logger.info("Exported symbol data to: %s", file_path)

        # Export symbol attributes (if any)
        symbol_attributes = self.portfolio_db.get_symbol_attributes()
        if not symbol_attributes.empty:
            file_path = (
                output_path / f"{prefix}_symbol_attributes_{timestamp}.csv"
            )
            symbol_attributes.to_csv(file_path, index=False)
            self.logger.info("Exported symbol attributes to: %s", file_path)

    def print_database_stats(self) -> None:
        """Print database statistics (if using database)."""
        if self.portfolio_db is not None:
            self.portfolio_db.print_stats()
        else:
            self.logger.warning(
                "No database available. Using legacy DataFrame."
            )
