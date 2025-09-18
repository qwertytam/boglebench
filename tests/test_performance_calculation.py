"""Tests for performance calculation in BogleBenchAnalyzer."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import numpy_financial as npf  # type: ignore
import pandas as pd
import pytest
from alpha_vantage.timeseries import TimeSeries  # type: ignore

from boglebench.core.portfolio import BogleBenchAnalyzer, PerformanceResults
from boglebench.utils.config import ConfigManager


def scenario_simple_case():
    """Simple scenario: Buy 100 AAPL on day 1 at $180, hold through week."""
    # Transactions: Buy 100 AAPL on day 1 at $180, hold through week
    purchase_price = 180.00
    final_price = 184.92  # price at close on final trading day
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

    market_data_files = {
        "AAPL": "AAPL_market_data_pytest.csv",
        "SPY": "SPY_market_data_pytest.csv",
    }

    return (
        "simple_case",
        "single_purchase_pytest.csv",
        expected_daily_returns,
        market_data_files,
    )


def scenario_benchmark_comparison():
    """Scenario with benchmark comparison: Buy 100 AAPL on day 1 at $180,
    hold through week."""

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

    market_data_files = {
        "AAPL": "AAPL_market_data_pytest.csv",
        "SPY": "SPY_market_data_pytest.csv",
    }

    return (
        "benchmark_comparison",
        "single_purchase_pytest.csv",
        expected_asset_daily_returns,
        market_data_files,
    )


def scenario_no_benchmark():
    """Scenario with no benchmark data: Buy 100 AAPL on day 1 at $180,
    hold through week."""

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

    market_data_files = {
        "AAPL": "AAPL_market_data_pytest.csv",
        # No benchmark data
    }

    return (
        "no_benchmark",
        "single_purchase_pytest.csv",
        expected_asset_daily_returns,
        market_data_files,
    )


def scenario_summary_generation():
    """Scenario for testing summary generation: Buy 100 AAPL on day 1 at $180,
    hold through week."""

    market_data_files = {
        "AAPL": "AAPL_market_data_pytest.csv",
        "SPY": "SPY_market_data_pytest.csv",
    }

    return (
        "summary_generation",
        "single_purchase_pytest.csv",
        None,
        market_data_files,
    )


class TestPerformanceCalculation:
    """Test suite for performance calculation."""

    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            (config_dir / "transactions").mkdir(exist_ok=True)
            (config_dir / "market_data").mkdir(exist_ok=True)
            (config_dir / "output").mkdir(exist_ok=True)

            config = ConfigManager()
            config.config["data"]["base_path"] = str(config_dir)

            config.config["analysis"]["start_date"] = "2023-06-05"
            config.config["analysis"]["end_date"] = "2023-06-16"

            config.config["settings"]["cache_market_data"] = True
            config.config["settings"]["force_refresh_market_data"] = False

            config.config["benchmark"]["components"] = [
                {"symbol": "SPY", "weight": 1.0}
            ]

            # Setting to 1.0 for ease of comparing total returns
            config.config["advanced"]["performance"][
                "modified_dietz_periodic_cash_flow_weight"
            ] = 1.0

            yield config

    @pytest.fixture
    def test_data_dir(self):
        """Path to test data directory."""
        return Path(__file__).parent / "test_data"

    @pytest.fixture(
        params=[
            scenario_simple_case(),
            scenario_benchmark_comparison(),
            scenario_no_benchmark(),
            scenario_summary_generation(),
        ],
        ids=lambda x: x[0],  # Use scenario name for test ID
    )
    def scenario_analyzer(
        self, request, temp_config, test_data_dir, monkeypatch
    ):
        """Fixture to set up BogleBenchAnalyzer for a given dividend
        scenario."""
        (
            scenario_name,
            transactions_file_name,
            expected_daily_returns,
            market_data_files,
        ) = request.param

        transactions_source = test_data_dir / transactions_file_name

        temp_data_path = temp_config.get_data_path()
        transactions_path = temp_data_path / "transactions"
        transactions_file_path = transactions_path / transactions_file_name
        shutil.copyfile(transactions_source, transactions_file_path)

        market_data_dict = {}
        for ticker, filename in market_data_files.items():
            market_data_path = test_data_dir / filename
            market_data_dict[ticker] = pd.read_csv(
                market_data_path, parse_dates=["date"]
            )
        market_data_path = temp_config.get_market_data_path()
        for ticker, df in market_data_dict.items():
            df["date"] = pd.to_datetime(
                df["date"], errors="coerce", format="%Y-%m-%d", utc=True
            )
            df.to_parquet(market_data_path / f"{ticker}.parquet", index=False)

        output_path = temp_config.get_output_path()

        monkeypatch.setattr(
            ConfigManager, "get_data_path", lambda self, config: temp_data_path
        )

        monkeypatch.setattr(
            ConfigManager,
            "get_transactions_file_path",
            lambda self: transactions_file_path,
        )

        monkeypatch.setattr(
            ConfigManager,
            "get_market_data_path",
            lambda self: market_data_path,
        )

        monkeypatch.setattr(
            ConfigManager,
            "get_output_path",
            lambda self: output_path,
        )

        monkeypatch.setattr(
            TimeSeries,
            "get_daily_adjusted",
            lambda self, symbol, outputsize: pd.DataFrame(),
        )

        analyzer = BogleBenchAnalyzer()
        analyzer.config = temp_config

        yield analyzer, scenario_name, transactions_file_path, expected_daily_returns

    def test_calculate_performance_scenarios(
        self, temp_config, scenario_analyzer
    ):
        """Test performance calculation with simple one-week scenario."""
        analyzer, scenario_name, transactions_file, expected_daily_returns = (
            scenario_analyzer
        )

        # Load data, build portfolio and calculate performance
        analyzer.load_transactions(transactions_file)
        analyzer.build_portfolio_history()
        results = analyzer.calculate_performance()

        # Verify results structure
        assert isinstance(results, PerformanceResults)

        portfolio_metrics = results.portfolio_metrics
        assert portfolio_metrics != {}
        assert portfolio_metrics is not None

        benchmark_metrics = results.benchmark_metrics
        relative_metrics = results.relative_metrics
        if scenario_name == "no_benchmark":
            # But no benchmark metrics
            assert benchmark_metrics == {}
            assert relative_metrics == {}
            return

        assert benchmark_metrics is not None
        assert benchmark_metrics != {}
        assert relative_metrics is not None
        assert relative_metrics != {}
        assert results.config is not None
        assert results.portfolio_history is not None

        # Test portfolio metrics calculations
        portfolio_mod_dietz_metrics = portfolio_metrics["mod_dietz"]
        assert "total_return" in portfolio_mod_dietz_metrics
        assert "annualized_return" in portfolio_mod_dietz_metrics
        assert "volatility" in portfolio_mod_dietz_metrics
        assert "sharpe_ratio" in portfolio_mod_dietz_metrics
        assert "max_drawdown" in portfolio_mod_dietz_metrics
        assert "win_rate" in portfolio_mod_dietz_metrics

        assert "irr" in portfolio_metrics

        portfolio_twr_metrics = portfolio_metrics["twr"]
        assert "total_return" in portfolio_twr_metrics
        assert "annualized_return" in portfolio_twr_metrics
        assert "volatility" in portfolio_twr_metrics
        assert "sharpe_ratio" in portfolio_twr_metrics
        assert "max_drawdown" in portfolio_twr_metrics
        assert "win_rate" in portfolio_twr_metrics

        annual_trading_days = results.config.get(
            "advanced.performance.annualization_factor", 252
        )
        if isinstance(annual_trading_days, dict):
            annual_trading_days = annual_trading_days.get("value", 252)
        if annual_trading_days is None or annual_trading_days <= 0:
            annual_trading_days = 252

        risk_free_rate = temp_config.get("analysis.risk_free_rate", 0.02)
        daily_risk_free_rate = (1 + risk_free_rate) ** (
            1 / annual_trading_days
        ) - 1

        accuracy = 0.001 / 100  # 0.001% accuracy

        if scenario_name == "simple_case":
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
            expected_total_return = (
                final_price - purchase_price
            ) / purchase_price

            # Verify portfolio history was built correctly
            portfolio_history = results.portfolio_history

            assert len(portfolio_history) == 10  # 10 trading days

            expected_columns = [
                "date",
                "Test_Account_AAPL_shares",
                "Test_Account_AAPL_value",
                "Test_Account_total",
                "total_value",
                "AAPL_total_shares",
                "AAPL_total_value",
                "AAPL_price",
                "investment_cash_flow",
                "income_cash_flow",
                "net_cash_flow",
                "Test_Account_cash_flow",
                "portfolio_daily_return_mod_dietz",
                "portfolio_daily_return_twr",
                "Test_Account_mod_dietz_return",
                "Test_Account_twr_return",
                "benchmark_returns",
                "market_value_change",
                "market_value_return",
                "Test_Account_AAPL_weight",
                "Test_Account_weight",
                "AAPL_weight",
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
                portfolio_metrics["irr"]["annualized_return"]
                - expected_annual_irr
            ) < (accuracy)

            # TWR total return
            expected_daily_twr_returns = expected_daily_returns.copy()
            expected_daily_twr_returns[0] = 0  # First return is zero for TWR
            expected_total_twr_return = (
                1 + expected_daily_twr_returns
            ).prod() - 1
            expected_daily_mean_twr_returns = np.mean(
                expected_daily_twr_returns
            )

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
            expected_annual_sharpe_twr_ratio = (
                expected_sharpe_twr_ratio * np.sqrt(annual_trading_days)
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
                abs(
                    portfolio_twr_metrics["max_drawdown"]
                    - expected_max_drawdown
                )
                < accuracy
            )

        elif scenario_name == "benchmark_comparison":
            # SPY adj close: 414.49 -> 428.07
            expected_benchmark_returns = (428.07 - 414.49) / 414.49
            accuracy = 0.001 / 100  # 0.001% accuracy
            assert (
                abs(
                    benchmark_metrics["total_return"]
                    - expected_benchmark_returns
                )
                < accuracy
            )

            # Check relative metrics
            assert "tracking_error" in relative_metrics
            assert "information_ratio" in relative_metrics
            assert "beta" in relative_metrics
            assert "jensens_alpha" in relative_metrics
            assert "correlation" in relative_metrics

            # Check metrics output
            return_days = len(results.portfolio_history) - 1
            expected_annualized_return = (1 + expected_benchmark_returns) ** (
                annual_trading_days / return_days
            ) - 1
            assert (
                abs(
                    benchmark_metrics["annualized_return"]
                    - expected_annualized_return
                )
                < accuracy
            )

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

            expected_daily_mean_returns = np.mean(
                expected_bm_daily_returns[1:]
            )
            expected_volatility = np.std(
                expected_bm_daily_returns[1:], ddof=1
            )  # Sample stddev so ddof=1
            expected_annual_volatility = expected_volatility * np.sqrt(
                annual_trading_days
            )

            assert (
                abs(
                    benchmark_metrics["volatility"]
                    - expected_annual_volatility
                )
                < accuracy
            )

            expected_sharpe_ratio = (
                expected_daily_mean_returns - daily_risk_free_rate
            ) / expected_volatility
            expected_annual_sharpe_ratio = expected_sharpe_ratio * np.sqrt(
                annual_trading_days
            )
            assert (
                abs(
                    benchmark_metrics["sharpe_ratio"]
                    - expected_annual_sharpe_ratio
                )
                < accuracy
            )

            # Max drawdown; use dataframe for cummax function
            wealth = pd.DataFrame(
                (1 + expected_bm_daily_returns[1:]).cumprod()
            )
            draw_down = wealth / wealth.cummax() - 1
            expected_max_drawdown = draw_down.min().values[0]
            assert (
                abs(benchmark_metrics["max_drawdown"] - expected_max_drawdown)
                < accuracy
            )

            # Tracking error
            expected_excess_returns = (
                expected_daily_returns[1:] - expected_bm_daily_returns[1:]
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
                abs(
                    relative_metrics["information_ratio"] - expected_info_ratio
                )
                < accuracy
            )

            # Beta
            covariance_matrix = np.cov(
                expected_daily_returns[1:],
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
                np.mean(expected_daily_returns[1:])
                - daily_risk_free_rate
                - expected_beta
                * (
                    np.mean(expected_bm_daily_returns[1:])
                    - daily_risk_free_rate
                )
            ) * annual_trading_days  # Annualize
            assert (
                abs(relative_metrics["jensens_alpha"] - expected_jensens_alpha)
                < accuracy
            )

        elif scenario_name == "summary_generation":
            summary = results.summary()

            # Check summary contains expected sections
            assert "BOGLEBENCH PERFORMANCE ANALYSIS" in summary
            assert "John C. Bogle" in summary
            assert "PORTFOLIO PERFORMANCE" in summary
            assert "SPY PERFORMANCE" in summary  # Benchmark section
            assert "RELATIVE PERFORMANCE" in summary
            assert "Total Return:" in summary
            assert "Sharpe Ratio:" in summary

        else:
            pytest.fail(f"Unknown scenario name: {scenario_name}")
            pytest.fail(f"Unknown scenario name: {scenario_name}")
