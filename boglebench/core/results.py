"""Results container and reporting for portfolio performance analysis."""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..utils.config import ConfigManager
from ..utils.logging_config import get_logger
from ..utils.tools import cagr
from .constants import Defaults


class PerformanceResults:
    """Container for portfolio performance analysis results."""

    def __init__(
        self,
        portfolio_metrics: Optional[Dict] = None,
        benchmark_metrics: Optional[Dict] = None,
        relative_metrics: Optional[Dict] = None,
        portfolio_history: Optional[pd.DataFrame] = None,
        config: Optional[ConfigManager] = None,
    ):
        self.portfolio_metrics = portfolio_metrics
        self.benchmark_metrics = benchmark_metrics
        self.relative_metrics = relative_metrics
        self.portfolio_history = portfolio_history
        self.config = config

        self.logger = get_logger("core.results")

    def summary(self) -> str:
        """Generate a summary report of the performance analysis."""
        lines = []
        lines.append("=" * 60)
        lines.append("🎯 BOGLEBENCH PERFORMANCE ANALYSIS")
        lines.append("   'Stay the course' - John C. Bogle")
        lines.append("=" * 60)

        # Portfolio metrics.   f"{3­:0=­+5}­"
        if self.portfolio_metrics:
            p = self.portfolio_metrics
            lines.append("\n📊 PORTFOLIO PERFORMANCE")
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
                f"\n  Max Drawdown:        "
                f"{p['mod_dietz']['max_drawdown']:>+8.2%}"
            )
            lines.append(
                f"  Win Rate:            {p['mod_dietz']['win_rate']:>+8.2%}"
            )

        # Benchmark metrics
        if self.benchmark_metrics:
            b = self.benchmark_metrics
            benchmark_name = self.config.get(
                "settings.benchmark_ticker", "Benchmark"
            )
            lines.append(f"\n📈 {benchmark_name} PERFORMANCE")
            lines.append(f"  Total Return:        {b['total_return']:>+8.2%}")
            lines.append(
                f"  Annualized Return:   {b['annualized_return']:>+8.2%}"
            )
            lines.append(f"  Volatility:          {b['volatility']:>+8.2%}")
            lines.append(f"  Sharpe Ratio:        {b['sharpe_ratio']:>+8.3f}")
            lines.append(f"  Max Drawdown:        {b['max_drawdown']:>+8.2%}")

        # Relative performance
        if self.relative_metrics:
            r = self.relative_metrics
            lines.append("\n🎯 RELATIVE PERFORMANCE (Using TWR)")
            lines.append(
                f"  Tracking Error:      {r['tracking_error']:>+8.2%}"
            )
            lines.append(
                f"  Information Ratio:   {r['information_ratio']:>+8.3f}"
            )
            lines.append(f"  Beta:                {r['beta']:>+8.3f}")
            lines.append(f"  Jensen's Alpha:      {r['jensens_alpha']:>+8.2%}")
            lines.append(f"  Correlation:         {r['correlation']:>+8.3f}")

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
        return self.portfolio_history[
            "portfolio_daily_return_mod_dietz"
        ].dropna()

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
                "settings.annual_trading_days", Defaults.DAYS_IN_TRADING_YEAR
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
                    f"{account}_mod_dietz_return"
                ].dropna()
                if len(account_returns) > 0:
                    total_periods = len(account_returns)
                    year_fraction = total_periods / annual_trading_days
                    total_return = (1 + account_returns).prod() - 1
                    annualized_return = cagr(
                        1, 1 + total_return, year_fraction
                    )
                else:
                    total_return = Defaults.ZERO_RETURN
                    annualized_return = Defaults.ZERO_RETURN

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
                            "quantity": shares,
                            "price": price,
                            "value": value,
                            "weight": weight,
                        }
                    )

        output_dir = self.config.get_output_path()
        output_path = self._export_history_metrics_to_csv(output_dir)

        self.logger.info("📁 Results exported to: %s", output_path)
        return pd.DataFrame(holdings_data)

    def export_to_csv(self, output_dir: Optional[str] = None) -> str:
        """Export results to CSV files."""
        output_path = self._export_history_metrics_to_csv(output_dir)
        self.logger.info("📁 Results exported to: %s", output_path)
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
            metrics_data.append(
                {
                    **self.portfolio_metrics,
                    "type": "Portfolio",
                }
            )
        if self.benchmark_metrics:
            metrics_data.append(
                {**self.benchmark_metrics, "type": "Benchmark"}
            )

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
