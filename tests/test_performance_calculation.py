"""Tests for performance calculation in BogleBenchAnalyzer."""

import tempfile
from pathlib import Path

import numpy as np
import numpy_financial as npf
import pandas as pd
import pytest

from boglebench.core.portfolio import BogleBenchAnalyzer, PerformanceResults
from boglebench.utils.config import ConfigManager


class TestPerformanceCalculation:
    """Test suite for performance calculation."""

    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create config
            config = ConfigManager()
            config.config["data"]["base_path"] = str(config_dir)

            # Create directories
            (config_dir / "transactions").mkdir()
            (config_dir / "market_data").mkdir()
            (config_dir / "output").mkdir()

            # Setting to 1.0 for ease of comparing total returns
            config.config["advanced"]["performance"][
                "period_cash_flow_weight"
            ] = 1.0

            config.config["analysis"]["start_date"] = "2023-06-05"
            config.config["analysis"]["end_date"] = "2023-06-16"

            yield config

    @pytest.fixture
    def test_data_dir(self):
        """Path to test data directory."""
        return Path(__file__).parent / "test_data"

    def test_calculate_performance_simple_case(
        self, test_data_dir, temp_config
    ):
        """Test performance calculation with simple one-week scenario."""
        # Create analyzer
        analyzer = BogleBenchAnalyzer()
        analyzer.config = temp_config
        analyzer.start_date = temp_config.get("analysis.start_date")
        analyzer.end_date = temp_config.get("analysis.end_date")

        # Simple market data for AAPL and SPY over one week
        aapl_mkt_data_path = test_data_dir / "AAPL_market_data_pytest.csv"
        aapl_mkt_data = pd.read_csv(aapl_mkt_data_path, parse_dates=["date"])

        spy_mkt_data_path = test_data_dir / "SPY_market_data_pytest.csv"
        spy_mkt_data = pd.read_csv(spy_mkt_data_path, parse_dates=["date"])

        simple_market_data = {
            "AAPL": aapl_mkt_data,
            "SPY": spy_mkt_data,
        }

        # Transactions: Buy 100 AAPL on day 1 at $180, hold through week
        csv_path = test_data_dir / "single_purchase_pytest.csv"
        analyzer.load_transactions(csv_path)

        # Mock market data instead of fetching
        analyzer.market_data = simple_market_data
        analyzer.benchmark_data = simple_market_data["SPY"].copy()

        # Build portfolio and calculate performance
        analyzer.build_portfolio_history()
        results = analyzer.calculate_performance()

        # Verify results structure
        assert isinstance(results, PerformanceResults)
        assert results.portfolio_metrics is not None
        assert results.benchmark_metrics is not None
        assert results.relative_metrics is not None
        assert results.config is not None
        assert results.portfolio_history is not None

        # Test portfolio metrics calculations
        portfolio_mod_dietz_metrics = results.portfolio_metrics["mod_dietz"]
        assert "total_return" in portfolio_mod_dietz_metrics
        assert "annualized_return" in portfolio_mod_dietz_metrics
        assert "volatility" in portfolio_mod_dietz_metrics
        assert "sharpe_ratio" in portfolio_mod_dietz_metrics
        assert "max_drawdown" in portfolio_mod_dietz_metrics
        assert "win_rate" in portfolio_mod_dietz_metrics

        assert "irr" in results.portfolio_metrics

        portfolio_twr_metrics = results.portfolio_metrics["twr"]
        assert "total_return" in portfolio_twr_metrics
        assert "annualized_return" in portfolio_twr_metrics
        assert "volatility" in portfolio_twr_metrics
        assert "sharpe_ratio" in portfolio_twr_metrics
        assert "max_drawdown" in portfolio_twr_metrics
        assert "win_rate" in portfolio_twr_metrics

        # Check initial and final portfolio values
        purchase_price = 180.00
        start_price = 179.58  # price at close on first trading day
        final_price = 184.92  # price at close on final trading day

        purchase_qty = 100
        final_qty = 100

        initial_expected = purchase_qty * start_price
        final_expected = final_qty * final_price

        # Verify expected calculations for one week
        # Portfolio: Buy at 180 -> End at 184.92
        expected_total_return = (final_price - purchase_price) / purchase_price
        accuracy = 0.001 / 100  # 0.001% accuracy

        expected_daily_returns = np.array(
            [
                179.58 / purchase_price - 1,
                179.21 / 179.58 - 1,
                177.82 / 179.21 - 1,
                180.57 / 177.82 - 1,
                180.96 / 180.57 - 1,
                183.79 / 180.96 - 1,
                183.31 / 183.79 - 1,
                183.95 / 183.31 - 1,
                186.01 / 183.95 - 1,
                final_price / 186.01 - 1,
            ]
        )

        # Verify portfolio history was built correctly
        portfolio_history = results.portfolio_history

        assert len(portfolio_history) == 10  # 10 trading days

        expected_columns = [
            "date",
            "Test_Account_AAPL_shares",
            "Test_Account_AAPL_price",
            "Test_Account_AAPL_value",
            "Test_Account_total",
            "total_value",
            "AAPL_total_shares",
            "AAPL_total_value",
            "net_cash_flow",
            "weighted_cash_flow",
            "market_value_change",
            "market_value_return",
            "Test_Account_AAPL_weight",
            "AAPL_weight",
            "investment_cash_flow",
            "income_cash_flow",
            "Test_Account_cash_flow",
            "Test_Account_weighted_cash_flow",
            "cash_flow_impact",
            "portfolio_daily_return_mod_dietz",
            "portfolio_daily_return_twr",
            "Test_Account_mod_dietz_return",
            "Test_Account_mod_twr_return",
            "Benchmark_Returns",
        ]

        for col in expected_columns:
            assert col in portfolio_history.columns

        for col in portfolio_history.columns:
            assert col in expected_columns

        assert (
            abs(
                portfolio_mod_dietz_metrics["total_return"]
                - expected_total_return
            )
            < accuracy
        )

        # Initial and final values
        initial_value = portfolio_history["total_value"].iloc[0]
        final_value = portfolio_history["total_value"].iloc[-1]

        assert abs(initial_value - initial_expected) < accuracy
        assert abs(final_value - final_expected) < accuracy

        # Check metrics output
        annual_trading_days = int(
            results.config.get("settings.annual_trading_days", 252)
        )

        # For market return calculations, we use the number of return days,
        # which is one less than the number of portfolio history entries
        # because returns are calculated between days.
        # However, we are calculating total return over the entire period,
        # so we use the full length of portfolio history here.
        return_days = len(portfolio_history)

        # Annualized return
        expected_annualized_return = (1 + expected_total_return) ** (
            annual_trading_days / return_days
        ) - 1

        assert (
            abs(
                portfolio_mod_dietz_metrics["annualized_return"]
                - expected_annualized_return
            )
            < accuracy
        )

        expected_daily_mean_returns = np.mean(expected_daily_returns)

        # Volatility
        expected_volatility = np.std(
            expected_daily_returns, ddof=1
        )  # Sample stddev so ddof=1
        expected_annual_volatility = expected_volatility * np.sqrt(
            annual_trading_days
        )

        assert (
            abs(
                portfolio_mod_dietz_metrics["volatility"]
                - expected_annual_volatility
            )
            < accuracy
        )

        # IRR returns
        irr_daily_cash_flows = np.array(
            [
                purchase_price * purchase_qty * -1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                final_expected,
            ]
        )
        expected_irr = npf.irr(irr_daily_cash_flows)
        expected_annual_irr = (1 + expected_irr) ** annual_trading_days - 1

        assert abs(
            results.portfolio_metrics["irr"]["annualized_return"]
            - expected_annual_irr
        ) < (accuracy)

        # TWR total return
        expected_daily_twr_returns = expected_daily_returns.copy()
        expected_daily_twr_returns[0] = 0  # First return is zero for TWR
        expected_total_twr_return = (1 + expected_daily_twr_returns).prod() - 1
        expected_daily_mean_twr_returns = np.mean(expected_daily_twr_returns)

        assert (
            abs(
                portfolio_twr_metrics["total_return"]
                - expected_total_twr_return
            )
            < accuracy
        )

        expected_twr_volatility = np.std(
            expected_daily_twr_returns, ddof=1
        )  # Sample stddev so ddof=1
        expected_annual_twr_volatility = expected_twr_volatility * np.sqrt(
            annual_trading_days
        )
        assert (
            abs(
                portfolio_twr_metrics["volatility"]
                - expected_annual_twr_volatility
            )
            < accuracy
        )

        risk_free_rate = temp_config.get("settings.risk_free_rate", 0.02)
        daily_risk_free_rate = (1 + risk_free_rate) ** (
            1 / annual_trading_days
        ) - 1
        expected_sharpe_ratio = (
            expected_daily_mean_returns - daily_risk_free_rate
        ) / expected_volatility
        expected_annual_sharpe_ratio = expected_sharpe_ratio * np.sqrt(
            annual_trading_days
        )
        assert abs(
            portfolio_mod_dietz_metrics["sharpe_ratio"]
            - expected_annual_sharpe_ratio
        ) < (accuracy)

        expected_sharpe_twr_ratio = (
            expected_daily_mean_twr_returns - daily_risk_free_rate
        ) / expected_twr_volatility
        expected_annual_sharpe_twr_ratio = expected_sharpe_twr_ratio * np.sqrt(
            annual_trading_days
        )
        assert abs(
            portfolio_twr_metrics["sharpe_ratio"]
            - expected_annual_sharpe_twr_ratio
        ) < (accuracy)

        # Max drawdown; use dataframe for cummax function
        wealth = pd.DataFrame((1 + expected_daily_returns).cumprod())
        draw_down = wealth / wealth.cummax() - 1
        expected_max_drawdown = draw_down.min().values[0]
        assert (
            abs(
                portfolio_mod_dietz_metrics["max_drawdown"]
                - expected_max_drawdown
            )
            < accuracy
        )

        assert (
            abs(portfolio_twr_metrics["max_drawdown"] - expected_max_drawdown)
            < accuracy
        )

    def test_benchmark_comparison(self, test_data_dir, temp_config):
        """Test benchmark comparison calculations with simple one-week
        scenario."""
        # Create analyzer
        analyzer = BogleBenchAnalyzer()
        analyzer.config = temp_config
        analyzer.start_date = temp_config.get("analysis.start_date")
        analyzer.end_date = temp_config.get("analysis.end_date")

        # Simple market data for AAPL and SPY over one week
        aapl_mkt_data_path = test_data_dir / "AAPL_market_data_pytest.csv"
        aapl_mkt_data = pd.read_csv(aapl_mkt_data_path, parse_dates=["date"])

        spy_mkt_data_path = test_data_dir / "SPY_market_data_pytest.csv"
        spy_mkt_data = pd.read_csv(spy_mkt_data_path, parse_dates=["date"])

        simple_market_data = {
            "AAPL": aapl_mkt_data,
            "SPY": spy_mkt_data,
        }

        # Transactions: Buy 100 AAPL on day 1 at $180, hold through week
        csv_path = test_data_dir / "single_purchase_pytest.csv"
        analyzer.load_transactions(csv_path)

        # Mock market data instead of fetching
        analyzer.market_data = simple_market_data
        analyzer.benchmark_data = simple_market_data["SPY"].copy()

        results = analyzer.calculate_performance()

        # Check benchmark metrics
        benchmark_metrics = results.benchmark_metrics
        assert "total_return" in benchmark_metrics

        # SPY adj close: 414.49 -> 428.07
        expected_benchmark_return = (428.07 - 414.49) / 414.49
        accuracy = 0.001 / 100  # 0.001% accuracy
        assert (
            abs(benchmark_metrics["total_return"] - expected_benchmark_return)
            < accuracy
        )

        # Check relative metrics
        relative_metrics = results.relative_metrics
        assert "tracking_error" in relative_metrics
        assert "information_ratio" in relative_metrics
        assert "beta" in relative_metrics
        assert "jensens_alpha" in relative_metrics
        assert "correlation" in relative_metrics

        # Check metrics output
        annual_trading_days = int(
            results.config.get("settings.annual_trading_days", 252)
        )
        return_days = len(results.portfolio_history) - 1
        expected_annualized_return = (1 + expected_benchmark_return) ** (
            annual_trading_days / return_days
        ) - 1
        assert (
            abs(
                benchmark_metrics["annualized_return"]
                - expected_annualized_return
            )
            < accuracy
        )

        # First return is zero as benchmark returns are based on market returns
        # Market returns are period to period change in close prices
        # So the first holding period is zero
        expected_bm_daily_returns = np.array(
            [
                0,
                415.39 / 414.49 - 1,
                413.95 / 415.39 - 1,
                416.46 / 413.95 - 1,
                417.2 / 416.46 - 1,
                420.99 / 417.2 - 1,
                423.76 / 420.99 - 1,
                424.27 / 423.76 - 1,
                429.53 / 424.27 - 1,
                428.07 / 429.53 - 1,
            ]
        )
        expected_daily_mean_returns = np.mean(expected_bm_daily_returns[1:])
        expected_volatility = np.std(
            expected_bm_daily_returns[1:], ddof=1
        )  # Sample stddev so ddof=1
        expected_annual_volatility = expected_volatility * np.sqrt(
            annual_trading_days
        )

        assert (
            abs(benchmark_metrics["volatility"] - expected_annual_volatility)
            < accuracy
        )

        risk_free_rate = temp_config.get("settings.risk_free_rate", 0.02)
        daily_risk_free_rate = (1 + risk_free_rate) ** (1 / 252) - 1
        expected_sharpe_ratio = (
            expected_daily_mean_returns - daily_risk_free_rate
        ) / expected_volatility
        expected_annual_sharpe_ratio = expected_sharpe_ratio * np.sqrt(
            annual_trading_days
        )
        assert abs(
            benchmark_metrics["sharpe_ratio"] - expected_annual_sharpe_ratio
        ) < (
            accuracy * 1
        )  # Sharpe ratio can be larger, adjust accuracy if required

        # Max drawdown; use dataframe for cummax function
        wealth = pd.DataFrame((1 + expected_bm_daily_returns[1:]).cumprod())
        draw_down = wealth / wealth.cummax() - 1
        expected_max_drawdown = draw_down.min().values[0]
        assert (
            abs(benchmark_metrics["max_drawdown"] - expected_max_drawdown)
            < accuracy
        )

        expected_asset_daily_returns = np.array(
            [
                179.58 / 180 - 1,
                179.21 / 179.58 - 1,
                177.82 / 179.21 - 1,
                180.57 / 177.82 - 1,
                180.96 / 180.57 - 1,
                183.79 / 180.96 - 1,
                183.31 / 183.79 - 1,
                183.95 / 183.31 - 1,
                186.01 / 183.95 - 1,
                184.92 / 186.01 - 1,
            ]
        )

        # Tracking error
        expected_excess_returns = (
            expected_asset_daily_returns[1:] - expected_bm_daily_returns[1:]
        )
        expected_tracking_error = np.std(expected_excess_returns, ddof=1)
        expected_annual_tracking_error = expected_tracking_error * np.sqrt(
            annual_trading_days
        )
        assert (
            abs(
                relative_metrics["tracking_error"]
                - expected_annual_tracking_error
            )
            < accuracy
        )

        # Information ratio
        expected_info_ratio = (
            np.mean(expected_excess_returns)
            / expected_tracking_error
            * np.sqrt(annual_trading_days)
        )
        assert (
            abs(relative_metrics["information_ratio"] - expected_info_ratio)
            < accuracy
        )

        # Beta
        covariance_matrix = np.cov(
            expected_asset_daily_returns[1:],
            expected_bm_daily_returns[1:],
            ddof=1,
        )  # Sample covariance so ddof=1
        expected_beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
        assert abs(relative_metrics["beta"] - expected_beta) < accuracy

        # Jensen's Alpha
        daily_risk_free_rate = (1 + risk_free_rate) ** (
            1 / annual_trading_days
        ) - 1
        expected_jensens_alpha = (
            np.mean(expected_asset_daily_returns[1:])
            - daily_risk_free_rate
            - expected_beta
            * (np.mean(expected_bm_daily_returns[1:]) - daily_risk_free_rate)
        ) * annual_trading_days  # Annualize
        assert (
            abs(relative_metrics["jensens_alpha"] - expected_jensens_alpha)
            < accuracy
        )

    def test_performance_with_no_benchmark(self, test_data_dir, temp_config):
        """Test performance calculation when benchmark data is unavailable."""
        # Create analyzer
        analyzer = BogleBenchAnalyzer()
        analyzer.config = temp_config
        analyzer.start_date = temp_config.get("analysis.start_date")
        analyzer.end_date = temp_config.get("analysis.end_date")

        # Simple market data for AAPL and SPY over one week
        aapl_mkt_data_path = test_data_dir / "AAPL_market_data_pytest.csv"
        aapl_mkt_data = pd.read_csv(aapl_mkt_data_path, parse_dates=["date"])

        spy_mkt_data_path = test_data_dir / "SPY_market_data_pytest.csv"
        spy_mkt_data = pd.read_csv(spy_mkt_data_path, parse_dates=["date"])

        simple_market_data = {
            "AAPL": aapl_mkt_data,
            "SPY": spy_mkt_data,
        }

        # Transactions: Buy 100 AAPL on day 1 at $180, hold through week
        csv_path = test_data_dir / "single_purchase_pytest.csv"
        analyzer.load_transactions(csv_path)

        # Only provide portfolio data, no benchmark
        analyzer.market_data = {"AAPL": simple_market_data["AAPL"]}
        analyzer.benchmark_data = pd.DataFrame()  # Empty benchmark

        results = analyzer.calculate_performance()

        # Should still have portfolio metrics
        assert results.portfolio_metrics is not None
        # But no benchmark metrics
        assert results.benchmark_metrics == {}
        assert results.relative_metrics == {}

    def test_summary_generation(self, temp_config, test_data_dir):
        """Test that summary report is generated correctly."""
        # Create analyzer
        analyzer = BogleBenchAnalyzer()
        analyzer.config = temp_config
        analyzer.start_date = temp_config.get("analysis.start_date")
        analyzer.end_date = temp_config.get("analysis.end_date")

        # Simple market data for AAPL and SPY over one week
        aapl_mkt_data_path = test_data_dir / "AAPL_market_data_pytest.csv"
        aapl_mkt_data = pd.read_csv(aapl_mkt_data_path, parse_dates=["date"])

        spy_mkt_data_path = test_data_dir / "SPY_market_data_pytest.csv"
        spy_mkt_data = pd.read_csv(spy_mkt_data_path, parse_dates=["date"])

        simple_market_data = {
            "AAPL": aapl_mkt_data,
            "SPY": spy_mkt_data,
        }

        # Transactions: Buy 100 AAPL on day 1 at $180, hold through week
        csv_path = test_data_dir / "single_purchase_pytest.csv"
        analyzer.load_transactions(csv_path)

        # Mock market data instead of fetching
        analyzer.market_data = simple_market_data
        analyzer.benchmark_data = simple_market_data["SPY"].copy()

        results = analyzer.calculate_performance()
        summary = results.summary()

        # Check summary contains expected sections
        assert "BOGLEBENCH PERFORMANCE ANALYSIS" in summary
        assert "John C. Bogle" in summary
        assert "PORTFOLIO PERFORMANCE" in summary
        assert "SPY PERFORMANCE" in summary  # Benchmark section
        assert "RELATIVE PERFORMANCE" in summary
        assert "Total Return:" in summary
        assert "Sharpe Ratio:" in summary
