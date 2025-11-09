"""Test suite for performance calculations with multiple transactions."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from boglebench.core.portfolio import BogleBenchAnalyzer
from boglebench.utils.config import ConfigManager


class TestMultiTransactionPerformance:
    """Test suite for performance calculations with multiple transactions."""

    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            (config_dir / "transactions").mkdir()
            (config_dir / "market_data").mkdir()
            (config_dir / "output").mkdir()

            config = ConfigManager()
            config.config["data"]["base_path"] = str(config_dir)

            config.config["analysis"]["start_date"] = "2023-06-05"
            config.config["analysis"]["end_date"] = "2023-06-16"

            config.config["benchmark"]["name"] = "SPY"

            yield config

    @pytest.fixture
    def test_data_dir(self):
        """Path to test data directory."""
        return Path(__file__).parent / "test_data"

    @pytest.fixture
    def market_data(self, test_data_dir):
        """Simple market data for AAPL and SPY over one week."""
        # Simple market data for AAPL and SPY over one week
        aapl_mkt_data_path = test_data_dir / "AAPL_market_data_pytest.csv"
        aapl_mkt_data = pd.read_csv(aapl_mkt_data_path, parse_dates=["date"])

        msft_mkt_data_path = test_data_dir / "MSFT_market_data_pytest.csv"
        msft_mkt_data = pd.read_csv(msft_mkt_data_path, parse_dates=["date"])

        spy_mkt_data_path = test_data_dir / "SPY_market_data_pytest.csv"
        spy_mkt_data = pd.read_csv(spy_mkt_data_path, parse_dates=["date"])

        market_data_dict = {
            "AAPL": aapl_mkt_data,
            "MSFT": msft_mkt_data,
            "SPY": spy_mkt_data,
        }

        yield market_data_dict

    @pytest.fixture
    def scenario_analyzer(self, market_data, temp_config, monkeypatch):
        """Fixture to set up BogleBenchAnalyzer for a given dividend
        scenario."""

        # Save transactions to csv
        temp_data_path = temp_config.get_data_path()
        transactions_file_path = (
            temp_data_path / "transactions" / "transactions.csv"
        )

        market_data_path = temp_config.get_market_data_path()
        for symbol, df in market_data.items():
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
            df.to_parquet(market_data_path / f"{symbol}.parquet", index=False)

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
            ConfigManager,
            "get_benchmark_components",
            lambda self: [{"symbol": "SPY", "weight": 1.0}],
        )

        analyzer = BogleBenchAnalyzer()
        analyzer.config = temp_config

        yield analyzer

    def test_portfolio_position_tracking(
        self, test_data_dir, scenario_analyzer
    ):
        """Test that positions are tracked correctly through multiple
        transactions."""
        # Create analyzer
        analyzer = scenario_analyzer

        # Transactions
        csv_path = test_data_dir / "double_transactions_pytest.csv"
        analyzer.load_transactions(csv_path)

        # Build portfolio and calculate performance
        analyzer.build_portfolio_history()
        results = analyzer.calculate_performance()

        # Build portfolio history
        portfolio_history = results.portfolio_history
        portfolio_history["date"] = pd.to_datetime(portfolio_history["date"])

        # June 5: 100 AAPL shares at $180
        expected_quantity_aapl = 100
        expected_close_price_aapl = 179.58
        expected_close_value_aapl = (
            expected_quantity_aapl * expected_close_price_aapl
        )

        june_5_data = portfolio_history[
            portfolio_history["date"].dt.date
            == pd.Timestamp("2023-06-05").date()
        ]
        assert not june_5_data.empty, "No data found for June 5"
        assert (
            june_5_data.iloc[0]["Test_Account_AAPL_quantity"]
            == expected_quantity_aapl
        )
        assert (
            abs(
                june_5_data.iloc[0]["Test_Account_AAPL_value"]
                - expected_close_value_aapl
            )
            < 1
        )
        assert june_5_data.iloc[0]["Test_Account_MSFT_quantity"] == 0

        # June 8: 100 AAPL + 50 MSFT
        june_8_data = portfolio_history[
            portfolio_history["date"].dt.date
            == pd.Timestamp("2023-06-08").date()
        ]
        expected_quantity_aapl = 100
        expected_close_price_aapl = 180.57

        expected_quantity_msft = 50
        expected_close_price_msft = 325.26

        assert not june_8_data.empty, "No data found for June 8"
        assert (
            june_8_data.iloc[0]["Test_Account_AAPL_quantity"]
            == expected_quantity_aapl
        )
        assert (
            june_8_data.iloc[0]["Test_Account_MSFT_quantity"]
            == expected_quantity_msft
        )

        expected_total = (
            expected_quantity_aapl * expected_close_price_aapl
            + expected_quantity_msft * expected_close_price_msft
        )
        assert abs(june_8_data.iloc[0]["total_value"] - expected_total) < 1

        # June 12: 0 AAPL + 25 MSFT (after sales)
        june_12_data = portfolio_history[
            portfolio_history["date"].dt.date
            == pd.Timestamp("2023-06-12").date()
        ]

        expected_quantity_aapl = 0
        expected_quantity_msft = 25
        expected_close_price_msft = 331.85

        assert not june_12_data.empty, "No data found for June 12"
        assert (
            june_12_data.iloc[0]["Test_Account_AAPL_quantity"]
            == expected_quantity_aapl
        )
        assert (
            june_12_data.iloc[0]["Test_Account_MSFT_quantity"]
            == expected_quantity_msft
        )
        # Only 25 MSFT shares
        expected_close_value_msft = (
            expected_quantity_msft * expected_close_price_msft
        )
        assert (
            abs(
                june_12_data.iloc[0]["total_value"] - expected_close_value_msft
            )
            < 1
        )

    def test_realized_gains_calculation(
        self, test_data_dir, scenario_analyzer
    ):
        """Test performance calculation with realized gains from sales."""
        analyzer = scenario_analyzer

        # Setting to 1.0 for ease of comparing total returns
        analyzer.config.config["advanced"]["performance"][
            "modified_dietz_periodic_cash_flow_weight"
        ] = 1.0

        # Transactions
        csv_path = test_data_dir / "double_transactions_pytest.csv"
        analyzer.load_transactions(csv_path)

        analyzer.build_portfolio_history()
        results = analyzer.calculate_performance()

        # Verify performance metrics are calculated
        portfolio_metrics = results.portfolio_metrics
        assert "total_return" in portfolio_metrics["mod_dietz"]
        assert "win_rate" in portfolio_metrics["mod_dietz"]
        assert "total_return" in portfolio_metrics["twr"]
        assert "win_rate" in portfolio_metrics["twr"]

        # Check that win rate reflects profitable transactions
        # Both AAPL and MSFT sales were profitable, so win rate should be high
        assert portfolio_metrics["mod_dietz"]["win_rate"] > 0.5

        # Verify portfolio returns include realized gains
        portfolio_returns = results.get_portfolio_returns()
        assert len(portfolio_returns) > 0

        # On sale dates, there should be significant portfolio value changes
        portfolio_history = results.portfolio_history
        portfolio_history["date"] = pd.to_datetime(portfolio_history["date"])

        # Check for value changes on transaction dates
        june_12_returns = portfolio_history[
            portfolio_history["date"].dt.date
            == pd.Timestamp("2023-06-12").date()
        ]["portfolio_daily_return_mod_dietz"]

        if not june_12_returns.empty:
            # Should see portfolio composition change on sale date
            assert not pd.isna(june_12_returns.iloc[0])

    def test_partial_position_sales(self, test_data_dir, scenario_analyzer):
        """Test handling of partial position sales (MSFT case)."""
        analyzer = scenario_analyzer

        # Setting to 1.0 for ease of comparing total returns
        analyzer.config.config["advanced"]["performance"][
            "modified_dietz_periodic_cash_flow_weight"
        ] = 1.0

        # Transactions
        csv_path = test_data_dir / "double_transactions_pytest.csv"
        analyzer.load_transactions(csv_path)

        # Build portfolio and calculate performance
        analyzer.build_portfolio_history()
        results = analyzer.calculate_performance()
        portfolio_history = results.portfolio_history

        portfolio_history["date"] = pd.to_datetime(portfolio_history["date"])

        # Before MSFT purchase (June 7): 0 MSFT shares
        june_7_data = portfolio_history[
            portfolio_history["date"].dt.date
            == pd.Timestamp("2023-06-07").date()
        ]
        if not june_7_data.empty:
            assert june_7_data.iloc[0]["Test_Account_MSFT_quantity"] == 0

        # After MSFT purchase, before sale (June 9): 50 MSFT shares
        june_9_data = portfolio_history[
            portfolio_history["date"].dt.date
            == pd.Timestamp("2023-06-09").date()
        ]
        if not june_9_data.empty:
            assert june_9_data.iloc[0]["Test_Account_MSFT_quantity"] == 50

        # After partial sale (June 13): 25 MSFT shares remaining
        june_13_data = portfolio_history[
            portfolio_history["date"].dt.date
            == pd.Timestamp("2023-06-13").date()
        ]
        if not june_13_data.empty:
            assert june_13_data.iloc[0]["Test_Account_MSFT_quantity"] == 25

    def test_transaction_data_validation(
        self, test_data_dir, scenario_analyzer
    ):
        """Test that transaction data is properly validated and processed."""
        analyzer = scenario_analyzer

        # Setting to 1.0 for ease of comparing total returns
        analyzer.config.config["advanced"]["performance"][
            "modified_dietz_periodic_cash_flow_weight"
        ] = 1.0

        # Transactions
        csv_path = test_data_dir / "double_transactions_pytest.csv"
        analyzer.load_transactions(csv_path)
        transactions = analyzer.transactions

        # Verify transaction processing
        assert len(transactions) == 4

        # Check that SELL transactions have negative shares
        sell_transactions = transactions[
            transactions["transaction_type"] == "SELL"
        ]
        assert len(sell_transactions) == 2
        assert all(sell_transactions["quantity"] < 0)

        # Check that BUY transactions have positive shares
        buy_transactions = transactions[
            transactions["transaction_type"] == "BUY"
        ]
        assert len(buy_transactions) == 2
        assert all(buy_transactions["quantity"] > 0)

        # Verify total values are calculated correctly
        assert "total_value" in transactions.columns
        for _, row in transactions.iterrows():
            expected_total = row["quantity"] * row["value_per_share"]
            assert abs(row["total_value"] - expected_total) < 0.01

    def test_complex_performance_summary(
        self, temp_config, test_data_dir, scenario_analyzer
    ):
        """Test comprehensive performance summary with multiple
        transactions."""
        analyzer = scenario_analyzer

        # Setting to 1.0 for ease of comparing total returns
        analyzer.config.config["advanced"]["performance"][
            "modified_dietz_periodic_cash_flow_weight"
        ] = 1.0

        # Transactions
        csv_path = test_data_dir / "double_transactions_pytest.csv"
        analyzer.load_transactions(csv_path)

        # Build portfolio and calculate performance
        analyzer.build_portfolio_history()
        results = analyzer.calculate_performance()
        summary = results.summary()
        print(summary)

        expected_periods = 10

        # dollars; initial AAPL purchase, then buy MSFT, then sell some of both
        expected_asset_cash_flows = np.array(
            [
                18000,
                0,
                0,
                16250,
                0,
                -26700,
                0,
                0,
                0,
                0,
            ]
        )
        expected_asset_end_values = np.array(
            [
                17958,
                17921,
                17782,
                34320,
                34435.5,
                8296.25,
                8357.25,
                8433.5,
                8702.5,
                8558.25,
            ]
        )

        expected_asset_beg_values = np.zeros(expected_periods)
        expected_asset_beg_values[1:] = expected_asset_end_values[:-1]

        cash_flow_weight = temp_config.get(
            "advanced.performance.modified_dietz_periodic_cash_flow_weight",
            0.5,
        )
        expected_weighted_cfs = expected_asset_cash_flows * cash_flow_weight
        expected_asset_daily_returns_mod_dietz_numerator = (
            expected_asset_end_values
            - expected_asset_beg_values
            - expected_asset_cash_flows
        )

        # Verify portfolio history was built correctly
        portfolio_history = results.portfolio_history
        assert len(portfolio_history) == 10  # 10 trading days
        assert "total_value" in portfolio_history.columns
        assert "portfolio_daily_return_mod_dietz" in portfolio_history.columns
        assert "portfolio_daily_return_twr" in portfolio_history.columns

        # Verify benchmark history was built correctly
        benchmark_history = results.benchmark_history
        assert len(benchmark_history) == 10  # 10 trading days
        assert "adj_close" in benchmark_history.columns

        # Verify returns
        accuracy = 0.001 / 100  # 0.001% accuracy

        # Modified Dietz return
        expected_asset_daily_mod_dietz_returns = (
            expected_asset_daily_returns_mod_dietz_numerator
            / (expected_asset_beg_values + expected_weighted_cfs)
        )
        expected_asset_total_mod_dietz_return = float(
            (1 + expected_asset_daily_mod_dietz_returns).prod() - 1
        )
        portfolio_mod_dietz_metrics = results.portfolio_metrics["mod_dietz"]

        assert (
            abs(
                portfolio_mod_dietz_metrics["total_return"]
                - expected_asset_total_mod_dietz_return
            )
            < accuracy
        )

        # TWR return
        expected_asset_beg_values[0] = 1  # Avoid div by zero on first day
        expected_asset_daily_twr_returns = (
            (expected_asset_end_values - expected_asset_cash_flows)
            / expected_asset_beg_values
        ) - 1

        expected_asset_daily_twr_returns[0] = 0  # No return on first day
        expected_asset_total_twr_return = float(
            (1 + expected_asset_daily_twr_returns).prod() - 1
        )
        portfolio_twr_metrics = results.portfolio_metrics["twr"]

        assert (
            abs(
                portfolio_twr_metrics["total_return"]
                - expected_asset_total_twr_return
            )
            < accuracy
        )

        annual_trading_days = int(
            results.config.get(
                "advanced.performance.annualization_factor", 252
            )
        )
        # Annualized return
        return_days = len(portfolio_history)
        expected_annualized_asset_mod_dietz_return = (
            1 + expected_asset_total_mod_dietz_return
        ) ** (annual_trading_days / return_days) - 1
        assert (
            abs(
                portfolio_mod_dietz_metrics["annualized_return"]
                - expected_annualized_asset_mod_dietz_return
            )
            < accuracy
        )

        expected_annualized_asset_twr_return = (
            1 + expected_asset_total_twr_return
        ) ** (annual_trading_days / return_days) - 1
        assert (
            abs(
                portfolio_twr_metrics["annualized_return"]
                - expected_annualized_asset_twr_return
            )
            < accuracy
        )

        # Verify Volatility
        expected_asset_mod_dietz_volatility = np.std(
            expected_asset_daily_mod_dietz_returns, ddof=1
        )  # Sample stddev so ddof=1
        expected_annual_asset_mod_dietz_volatility = (
            expected_asset_mod_dietz_volatility * np.sqrt(annual_trading_days)
        )

        assert (
            abs(
                portfolio_mod_dietz_metrics["volatility"]
                - expected_annual_asset_mod_dietz_volatility
            )
            < accuracy
        )

        expected_asset_twr_volatility = np.std(
            expected_asset_daily_twr_returns, ddof=1
        )  # Sample stddev so ddof=1
        expected_annual_asset_twr_volatility = (
            expected_asset_twr_volatility * np.sqrt(annual_trading_days)
        )
        assert (
            abs(
                portfolio_twr_metrics["volatility"]
                - expected_annual_asset_twr_volatility
            )
            < accuracy
        )

        # Verify Sharpe Ratio
        expected_asset_daily_mean_mod_dietz_returns = np.mean(
            expected_asset_daily_mod_dietz_returns
        )
        risk_free_rate = temp_config.get("analysis.risk_free_rate", 0.02)
        daily_risk_free_rate = (1 + risk_free_rate) ** (
            1 / annual_trading_days
        ) - 1
        expected_asset_mod_dietz_sharpe_ratio = (
            expected_asset_daily_mean_mod_dietz_returns - daily_risk_free_rate
        ) / expected_asset_mod_dietz_volatility
        expected_annual_asset_mod_dietz_sharpe_ratio = (
            expected_asset_mod_dietz_sharpe_ratio
            * np.sqrt(annual_trading_days)
        )
        assert abs(
            portfolio_mod_dietz_metrics["sharpe_ratio"]
            - expected_annual_asset_mod_dietz_sharpe_ratio
        ) < (accuracy)

        expected_asset_daily_mean_twr_returns = np.mean(
            expected_asset_daily_twr_returns
        )
        expected_asset_twr_sharpe_ratio = (
            expected_asset_daily_mean_twr_returns - daily_risk_free_rate
        ) / expected_asset_twr_volatility
        expected_annual_asset_twr_sharpe_ratio = (
            expected_asset_twr_sharpe_ratio * np.sqrt(annual_trading_days)
        )
        assert abs(
            portfolio_twr_metrics["sharpe_ratio"]
            - expected_annual_asset_twr_sharpe_ratio
        ) < (accuracy)

        # Max drawdown; use dataframe for cummax function
        cum_asset_wealth = pd.DataFrame(
            (1 + expected_asset_daily_mod_dietz_returns).cumprod()
        )
        asset_draw_down = cum_asset_wealth / cum_asset_wealth.cummax() - 1
        expected_max_asset_drawdown = asset_draw_down.min().values[0]
        assert (
            abs(
                portfolio_mod_dietz_metrics["max_drawdown"]
                - expected_max_asset_drawdown
            )
            < accuracy
        )

        assert (
            abs(
                portfolio_twr_metrics["max_drawdown"]
                - expected_max_asset_drawdown
            )
            < accuracy
        )

        # Verify comprehensive summary content
        assert "BOGLEBENCH PERFORMANCE ANALYSIS" in summary
        assert "PORTFOLIO PERFORMANCE" in summary
        assert "'SPY' PERFORMANCE" in summary
        assert "RELATIVE PERFORMANCE" in summary
        assert "Total Return:" in summary
        assert "Volatility:" in summary
        assert "Sharpe Ratio:" in summary

        # Should contain valid percentage values (not inf or nan)
        assert "inf%" not in summary
        assert "nan%" not in summary

        # Check relative metrics
        relative_metrics = results.relative_metrics
        assert "tracking_error" in relative_metrics
        assert "information_ratio" in relative_metrics
        assert "beta" in relative_metrics
        assert "jensens_alpha" in relative_metrics
        assert "correlation" in relative_metrics
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

        assert benchmark_history["benchmark_return"].notna().all()
        assert (
            abs(
                benchmark_history["benchmark_return"]
                - expected_bm_daily_returns
            ).max()
            < accuracy
        )

        # Tracking error
        expected_excess_returns = (
            expected_asset_daily_twr_returns[1:]
            - expected_bm_daily_returns[1:]
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
            expected_asset_daily_twr_returns[1:],
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
            np.mean(expected_asset_daily_twr_returns[1:])
            - daily_risk_free_rate
            - expected_beta
            * (np.mean(expected_bm_daily_returns[1:]) - daily_risk_free_rate)
        ) * annual_trading_days  # Annualize

        assert (
            abs(relative_metrics["jensens_alpha"] - expected_jensens_alpha)
            < accuracy
        )
