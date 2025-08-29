# In tests/test_multi_transaction_performance.py
import tempfile
from pathlib import Path

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

            # Create config
            config = ConfigManager()
            config.config["data"]["base_path"] = str(config_dir)

            # Create directories
            (config_dir / "transactions").mkdir()
            (config_dir / "market_data").mkdir()
            (config_dir / "output").mkdir()

            yield config

    @pytest.fixture
    def test_data_dir(self):
        """Path to test data directory."""
        return Path(__file__).parent / "test_data"

    def test_portfolio_position_tracking(self, temp_config, test_data_dir):
        """Test that positions are tracked correctly through multiple
        transactions."""
        # Create analyzer
        analyzer = BogleBenchAnalyzer()
        analyzer.config = temp_config

        # Simple market data for AAPL, MSFT and SPY over two weeks
        aapl_mkt_data_path = test_data_dir / "AAPL_market_data_pytest.csv"
        aapl_mkt_data = pd.read_csv(aapl_mkt_data_path, parse_dates=["date"])

        msft_mkt_data_path = test_data_dir / "MSFT_market_data_pytest.csv"
        msft_mkt_data = pd.read_csv(msft_mkt_data_path, parse_dates=["date"])

        spy_mkt_data_path = test_data_dir / "SPY_market_data_pytest.csv"
        spy_mkt_data = pd.read_csv(spy_mkt_data_path, parse_dates=["date"])

        simple_market_data = {
            "AAPL": aapl_mkt_data,
            "MSFT": msft_mkt_data,
            "SPY": spy_mkt_data,
        }

        # Transactions
        csv_path = test_data_dir / "double_transactions_pytest.csv"
        analyzer.load_transactions(csv_path)

        # Mock market data instead of fetching
        analyzer.market_data = simple_market_data
        analyzer.benchmark_data = simple_market_data["SPY"].copy()

        # Build portfolio history
        portfolio_history = analyzer.build_portfolio_history()
        portfolio_history["date"] = pd.to_datetime(portfolio_history["date"])

        # June 5: 100 AAPL shares at $180
        june_5_data = portfolio_history[
            portfolio_history["date"].dt.date
            == pd.Timestamp("2023-06-05").date()
        ]
        assert not june_5_data.empty, "No data found for June 5"
        assert june_5_data.iloc[0]["Test_Account_AAPL_shares"] == 100
        assert abs(june_5_data.iloc[0]["Test_Account_AAPL_value"] - 18000) < 1
        assert june_5_data.iloc[0]["Test_Account_MSFT_shares"] == 0

        # June 8: 100 AAPL + 50 MSFT
        june_8_data = portfolio_history[
            portfolio_history["date"].dt.date
            == pd.Timestamp("2023-06-08").date()
        ]
        assert not june_8_data.empty, "No data found for June 8"
        assert june_8_data.iloc[0]["Test_Account_AAPL_shares"] == 100
        assert june_8_data.iloc[0]["Test_Account_MSFT_shares"] == 50
        # AAPL at $185 + MSFT at $260
        expected_total = 100 * 185 + 50 * 260
        assert abs(june_8_data.iloc[0]["total_value"] - expected_total) < 1

        # June 12: 0 AAPL + 25 MSFT (after sales)
        june_12_data = portfolio_history[
            portfolio_history["date"].dt.date
            == pd.Timestamp("2023-06-12").date()
        ]
        assert not june_12_data.empty, "No data found for June 12"
        assert june_12_data.iloc[0]["Test_Account_AAPL_shares"] == 0
        assert june_12_data.iloc[0]["Test_Account_MSFT_shares"] == 25
        # Only 25 MSFT shares at $265
        expected_remaining = 25 * 265
        assert abs(june_12_data.iloc[0]["total_value"] - expected_remaining) < 1

    def test_realized_gains_calculation(self, temp_config, test_data_dir):
        """Test performance calculation with realized gains from sales."""
        # Create analyzer
        analyzer = BogleBenchAnalyzer()
        analyzer.config = temp_config

        # Simple market data for AAPL, MSFT and SPY over two weeks
        aapl_mkt_data_path = test_data_dir / "AAPL_market_data_pytest.csv"
        aapl_mkt_data = pd.read_csv(aapl_mkt_data_path, parse_dates=["date"])

        msft_mkt_data_path = test_data_dir / "MSFT_market_data_pytest.csv"
        msft_mkt_data = pd.read_csv(msft_mkt_data_path, parse_dates=["date"])

        spy_mkt_data_path = test_data_dir / "SPY_market_data_pytest.csv"
        spy_mkt_data = pd.read_csv(spy_mkt_data_path, parse_dates=["date"])

        simple_market_data = {
            "AAPL": aapl_mkt_data,
            "MSFT": msft_mkt_data,
            "SPY": spy_mkt_data,
        }

        # Transactions
        csv_path = test_data_dir / "double_transactions_pytest.csv"
        analyzer.load_transactions(csv_path)

        # Mock market data instead of fetching
        analyzer.market_data = simple_market_data
        analyzer.benchmark_data = simple_market_data["SPY"].copy()

        results = analyzer.calculate_performance()

        # Verify performance metrics are calculated
        portfolio_metrics = results.portfolio_metrics
        assert "total_return" in portfolio_metrics
        assert "win_rate" in portfolio_metrics

        # Check that win rate reflects profitable transactions
        # Both AAPL and MSFT sales were profitable, so win rate should be high
        assert portfolio_metrics["win_rate"] > 0.5

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
        ]["portfolio_return"]

        if not june_12_returns.empty:
            # Should see portfolio composition change on sale date
            assert not pd.isna(june_12_returns.iloc[0])

    def test_partial_position_sales(self, temp_config, test_data_dir):
        """Test handling of partial position sales (MSFT case)."""
        # Create analyzer
        analyzer = BogleBenchAnalyzer()
        analyzer.config = temp_config

        # Simple market data for AAPL, MSFT and SPY over two weeks
        aapl_mkt_data_path = test_data_dir / "AAPL_market_data_pytest.csv"
        aapl_mkt_data = pd.read_csv(aapl_mkt_data_path, parse_dates=["date"])

        msft_mkt_data_path = test_data_dir / "MSFT_market_data_pytest.csv"
        msft_mkt_data = pd.read_csv(msft_mkt_data_path, parse_dates=["date"])

        spy_mkt_data_path = test_data_dir / "SPY_market_data_pytest.csv"
        spy_mkt_data = pd.read_csv(spy_mkt_data_path, parse_dates=["date"])

        simple_market_data = {
            "AAPL": aapl_mkt_data,
            "MSFT": msft_mkt_data,
            "SPY": spy_mkt_data,
        }

        # Transactions
        csv_path = test_data_dir / "double_transactions_pytest.csv"
        analyzer.load_transactions(csv_path)

        # Mock market data instead of fetching
        analyzer.market_data = simple_market_data
        analyzer.benchmark_data = simple_market_data["SPY"].copy()

        portfolio_history = analyzer.build_portfolio_history()
        portfolio_history["date"] = pd.to_datetime(portfolio_history["date"])

        # Before MSFT purchase (June 7): 0 MSFT shares
        june_7_data = portfolio_history[
            portfolio_history["date"].dt.date
            == pd.Timestamp("2023-06-07").date()
        ]
        if not june_7_data.empty:
            assert june_7_data.iloc[0]["Test_Account_MSFT_shares"] == 0

        # After MSFT purchase, before sale (June 9): 50 MSFT shares
        june_9_data = portfolio_history[
            portfolio_history["date"].dt.date
            == pd.Timestamp("2023-06-09").date()
        ]
        if not june_9_data.empty:
            assert june_9_data.iloc[0]["Test_Account_MSFT_shares"] == 50

        # After partial sale (June 13): 25 MSFT shares remaining
        june_13_data = portfolio_history[
            portfolio_history["date"].dt.date
            == pd.Timestamp("2023-06-13").date()
        ]
        if not june_13_data.empty:
            assert june_13_data.iloc[0]["Test_Account_MSFT_shares"] == 25

    def test_transaction_data_validation(self, temp_config, test_data_dir):
        """Test that transaction data is properly validated and processed."""
        # Create analyzer
        analyzer = BogleBenchAnalyzer()
        analyzer.config = temp_config

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
        assert all(sell_transactions["shares"] < 0)

        # Check that BUY transactions have positive shares
        buy_transactions = transactions[
            transactions["transaction_type"] == "BUY"
        ]
        assert len(buy_transactions) == 2
        assert all(buy_transactions["shares"] > 0)

        # Verify total values are calculated correctly
        assert "total_value" in transactions.columns
        for _, row in transactions.iterrows():
            expected_total = row["shares"] * row["price_per_share"]
            assert abs(row["total_value"] - expected_total) < 0.01

    def test_complex_performance_summary(self, temp_config, test_data_dir):
        """Test comprehensive performance summary with multiple
        transactions."""
        # Create analyzer
        analyzer = BogleBenchAnalyzer()
        analyzer.config = temp_config

        # Simple market data for AAPL, MSFT and SPY over two weeks
        aapl_mkt_data_path = test_data_dir / "AAPL_market_data_pytest.csv"
        aapl_mkt_data = pd.read_csv(aapl_mkt_data_path, parse_dates=["date"])

        msft_mkt_data_path = test_data_dir / "MSFT_market_data_pytest.csv"
        msft_mkt_data = pd.read_csv(msft_mkt_data_path, parse_dates=["date"])

        spy_mkt_data_path = test_data_dir / "SPY_market_data_pytest.csv"
        spy_mkt_data = pd.read_csv(spy_mkt_data_path, parse_dates=["date"])

        simple_market_data = {
            "AAPL": aapl_mkt_data,
            "MSFT": msft_mkt_data,
            "SPY": spy_mkt_data,
        }

        # Transactions
        csv_path = test_data_dir / "double_transactions_pytest.csv"
        analyzer.load_transactions(csv_path)

        # Mock market data instead of fetching
        analyzer.market_data = simple_market_data
        analyzer.benchmark_data = simple_market_data["SPY"].copy()

        results = analyzer.calculate_performance()
        summary = results.summary()

        print("Portfolio Metrics:\n%s\n", results.portfolio_metrics)
        print("Benchmark Metrics:\n%s\n", results.benchmark_metrics)
        print("Relative Metrics:\n%s\n", results.relative_metrics)

        # Verify comprehensive summary content
        assert "BOGLEBENCH PERFORMANCE ANALYSIS" in summary
        assert "PORTFOLIO PERFORMANCE" in summary
        assert "SPY PERFORMANCE" in summary
        assert "RELATIVE PERFORMANCE" in summary
        assert "Total Return:" in summary
        assert "Volatility:" in summary
        assert "Sharpe Ratio:" in summary

        # Should contain valid percentage values (not inf or nan)
        assert "inf%" not in summary
        assert "nan%" not in summary
