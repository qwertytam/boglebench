"""
Visualization module for BogleBench portfolio analysis.

Create comprehensive charts and visualizations following Bogle's principles
of simple, clear communication of investment performance.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..core.portfolio import PerformanceResults


class BogleBenchCharts:
    """Create charts and visualizations for portfolio analysis."""

    def __init__(self, results: PerformanceResults):
        """
        Initialize chart generator with performance results.

        Args:
            results: PerformanceResults object from analysis
        """
        self.results = results
        self.style_setup()

    def style_setup(self):
        """Set up chart styling in Bogle's spirit: clean and simple."""
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        # Custom color palette inspired by Vanguard's simple approach
        self.colors = {
            "portfolio": "#1f77b4",  # Blue
            "benchmark": "#ff7f0e",  # Orange
            "positive": "#2ca02c",  # Green
            "negative": "#d62728",  # Red
            "neutral": "#7f7f7f",  # Gray
        }

    def create_performance_dashboard(
        self, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive performance dashboard.

        Args:
            save_path: Path to save the chart. If None, displays only.

        Returns:
            matplotlib Figure object
        """

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "BogleBench Performance Dashboard\n"
            '"Stay the course" - John C. Bogle',
            fontsize=16,
            fontweight="bold",
        )

        # 1. Portfolio Value Growth
        self._plot_portfolio_growth(axes[0, 0])

        # 2. Cumulative Returns Comparison
        self._plot_cumulative_returns(axes[0, 1])

        # 3. Account Allocation
        self._plot_account_allocation(axes[0, 2])

        # 4. Asset Allocation
        self._plot_asset_allocation(axes[1, 0])

        # 5. Rolling Returns
        self._plot_rolling_returns(axes[1, 1])

        # 6. Risk-Return Scatter
        self._plot_risk_return(axes[1, 2])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Dashboard saved to: {save_path}")

        return fig

    def _plot_portfolio_growth(self, ax):
        """Plot portfolio value over time."""
        portfolio_history = self.results.portfolio_history

        ax.plot(
            portfolio_history["date"],
            portfolio_history["total_value"],
            color=self.colors["portfolio"],
            linewidth=2,
            label="Portfolio Value",
        )

        ax.set_title("Portfolio Value Growth", fontweight="bold")
        ax.set_ylabel("Portfolio Value ($)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f"${x:,.0f}")
        )

    def _plot_cumulative_returns(self, ax):
        """Plot cumulative returns vs benchmark."""
        portfolio_returns = self.results.get_cumulative_returns()

        ax.plot(
            portfolio_returns.index,
            portfolio_returns.values * 100,
            color=self.colors["portfolio"],
            linewidth=2,
            label="Portfolio",
        )

        # Add benchmark if available
        if self.results.benchmark_metrics:
            # Calculate benchmark cumulative returns
            # This would need benchmark return data
            pass

        ax.set_title("Cumulative Returns", fontweight="bold")
        ax.set_ylabel("Cumulative Return (%)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_account_allocation(self, ax):
        """Plot allocation by account."""
        account_summary = self.results.get_account_summary()

        if not account_summary.empty:
            wedges, texts, autotexts = ax.pie(
                account_summary["current_value"],
                labels=account_summary["account"],
                autopct="%1.1f%%",
                startangle=90,
                colors=sns.color_palette("husl", len(account_summary)),
            )

            ax.set_title("Allocation by Account", fontweight="bold")
        else:
            ax.text(
                0.5,
                0.5,
                "No account data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Allocation by Account", fontweight="bold")

    def _plot_asset_allocation(self, ax):
        """Plot allocation by asset."""
        holdings = self.results.get_account_holdings()

        if not holdings.empty:
            # Aggregate by symbol across all accounts
            asset_allocation = holdings.groupby("symbol")["value"].sum()

            wedges, texts, autotexts = ax.pie(
                asset_allocation.values,
                labels=asset_allocation.index,
                autopct="%1.1f%%",
                startangle=90,
                colors=sns.color_palette("husl", len(asset_allocation)),
            )

            ax.set_title("Asset Allocation", fontweight="bold")
        else:
            ax.text(
                0.5,
                0.5,
                "No holdings data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Asset Allocation", fontweight="bold")

    def _plot_rolling_returns(self, ax):
        """Plot rolling 12-month returns."""
        portfolio_returns = self.results.get_portfolio_returns()

        if len(portfolio_returns) >= 252:  # Need at least 1 year
            # Calculate 12-month rolling returns
            rolling_returns = (
                portfolio_returns.rolling(252)
                .apply(lambda x: (1 + x).prod() - 1, raw=False)
                .dropna()
            )

            ax.plot(
                rolling_returns.index,
                rolling_returns.values * 100,
                color=self.colors["portfolio"],
                linewidth=2,
            )

            ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
            ax.set_title("Rolling 12-Month Returns", fontweight="bold")
            ax.set_ylabel("Return (%)")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5,
                0.5,
                "Insufficient data for rolling returns",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Rolling 12-Month Returns", fontweight="bold")

    def _plot_risk_return(self, ax):
        """Plot risk vs return scatter."""
        portfolio_metrics = self.results.portfolio_metrics
        benchmark_metrics = self.results.benchmark_metrics

        # Portfolio point
        if portfolio_metrics:
            ax.scatter(
                portfolio_metrics["twr"]["volatility"] * 100,
                portfolio_metrics["twr"]["annualized_return"] * 100,
                s=200,
                color=self.colors["portfolio"],
                label="Portfolio",
                alpha=0.8,
            )

        # Benchmark point
        if benchmark_metrics:
            ax.scatter(
                benchmark_metrics["volatility"] * 100,
                benchmark_metrics["annualized_return"] * 100,
                s=200,
                color=self.colors["benchmark"],
                label="Benchmark",
                alpha=0.8,
            )

        ax.set_xlabel("Risk (Volatility %)")
        ax.set_ylabel("Return (%)")
        ax.set_title("Risk vs Return", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

    def create_account_comparison(
        self, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create detailed account comparison charts."""
        account_summary = self.results.get_account_summary()

        if account_summary.empty:
            print("No account data available for comparison")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            "Account Performance Comparison", fontsize=14, fontweight="bold"
        )

        # 1. Account Values
        axes[0, 0].bar(
            account_summary["account"],
            account_summary["current_value"],
            color=sns.color_palette("husl", len(account_summary)),
        )
        axes[0, 0].set_title("Current Account Values")
        axes[0, 0].set_ylabel("Value ($)")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # 2. Account Returns
        axes[0, 1].bar(
            account_summary["account"],
            account_summary["total_return"] * 100,
            color=sns.color_palette("husl", len(account_summary)),
        )
        axes[0, 1].set_title("Total Returns by Account")
        axes[0, 1].set_ylabel("Return (%)")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # 3. Portfolio Weight
        axes[1, 0].pie(
            account_summary["current_value"],
            labels=account_summary["account"],
            autopct="%1.1f%%",
            startangle=90,
        )
        axes[1, 0].set_title("Portfolio Weight by Account")

        # 4. Performance Metrics Table
        axes[1, 1].axis("tight")
        axes[1, 1].axis("off")

        # Create performance table
        table_data = account_summary[
            ["account", "current_value", "total_return", "annualized_return"]
        ].copy()
        table_data["current_value"] = table_data["current_value"].apply(
            lambda x: f"${x:,.0f}"
        )
        table_data["total_return"] = table_data["total_return"].apply(
            lambda x: f"{x:.2%}"
        )
        table_data["annualized_return"] = table_data[
            "annualized_return"
        ].apply(lambda x: f"{x:.2%}")

        table = axes[1, 1].table(
            cellText=table_data.values,
            colLabels=["Account", "Value", "Total Return", "Annual Return"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title("Performance Summary")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"📊 Account comparison saved to: {save_path}")

        return fig
