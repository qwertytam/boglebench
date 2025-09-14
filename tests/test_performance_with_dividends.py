"""
Tests for performance calculation under various dividend scenarios.

This suite verifies that the end-to-end performance metrics are calculated
correctly when dividends are paid in cash, fully reinvested, or partially
reinvested.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from boglebench.core.market_data import MarketDataProvider
from boglebench.core.portfolio import BogleBenchAnalyzer
from boglebench.core.results import PerformanceResults
from boglebench.utils.config import ConfigManager


def create_mock_market_data(workspace: Path, market_data_dict: dict):
    """Creates mock market data parquet files for testing."""
    market_data_dir = workspace / "market_data"
    market_data_dir.mkdir(exist_ok=True)

    for ticker, data in market_data_dict.items():
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        # Add required columns if they don't exist
        for col in [
            "close",
            "adj_close",
            "split_coefficient",
        ]:
            if col not in df.columns:
                if col == "adj_close":
                    df[col] = df["close"]
                else:
                    df[col] = 0

        df.to_parquet(market_data_dir / f"{ticker}.parquet")


class TestPerformanceWithDividends:
    """Test suite for performance calculations involving dividends."""

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

            # Updating default settings
            config.config["settings"]["benchmark_ticker"] = "SPY"
            config.config["settings"]["cache_market_data"] = False
            config.config["api"]["alpha_vantage_key"] = "DUMMY_KEY"
            config.config["data"]["transactions_file"] = "transactions.csv"

            config.config["analysis"]["start_date"] = "2023-01-03"
            config.config["analysis"]["default_end_date"] = "2023-01-05"

            yield config

    def _setup_scenario(
        self, temp_config, transactions_data, market_data_dict
    ):
        """
        Helper to set up a test scenario by creating data files and the analyzer.
        It also mocks the market data fetching to use local files.
        """
        # Patch the analyzer's fetch method to prevent live API calls
        with patch.object(
            MarketDataProvider,
            "get_market_data",
            return_value=market_data_dict,
        ) as mock_fetch:
            analyzer = BogleBenchAnalyzer()
            analyzer.config = temp_config

            # Write transactions to CSV
            workspace = analyzer.config.get_data_path()
            transactions_file = workspace / "transactions.csv"
            pd.DataFrame(transactions_data).to_csv(
                transactions_file, index=False
            )

            # Create mock market data files
            create_mock_market_data(workspace, market_data_dict)

            # Manually assign mocked data since we are bypassing the fetch logic
            analyzer.market_data = {
                ticker: pd.read_parquet(
                    workspace / "market_data" / f"{ticker}.parquet"
                )
                for ticker in market_data_dict
            }
            analyzer.benchmark_data = analyzer.market_data["SPY"]

            yield analyzer, mock_fetch

    def test_performance_no_dividends(self, temp_config):
        """Test performance calculation with a simple BUY and no dividends."""
        transactions = [
            {
                "date": "2023-01-03",
                "ticker": "VOO",
                "transaction_type": "BUY",
                "quantity": 10,
                "value_per_share": 100.00,
                "total_value": 1000.00,
                "account": "Taxable",
            }
        ]
        market_data = {
            "VOO": {
                "date": ["2023-01-03", "2023-01-04", "2023-01-05"],
                "close": [100.0, 101.0, 102.0],
                "dividend": [0.0, 0.0, 0.0],
            },
            "SPY": {
                "date": ["2023-01-03", "2023-01-04", "2023-01-05"],
                "close": [400.0, 401.0, 402.0],
                "dividend": [0.0, 0.0, 0.0],
            },
        }

        for analyzer, _ in self._setup_scenario(
            temp_config, transactions, market_data
        ):
            analyzer.load_transactions()
            analyzer.build_portfolio_history()
            results = analyzer.calculate_performance()

            assert results is not None
            final_day = results.portfolio_history.iloc[-1]
            assert final_day["Taxable_VOO_shares"] == 10
            assert (
                final_day["total_value"] == 10 * 102.0
            )  # 10 shares * final price

            # Verify results structure
            assert isinstance(results, PerformanceResults)
            assert results.portfolio_metrics is not None
            assert results.benchmark_metrics is not None
            assert results.relative_metrics is not None
            assert results.config is not None
            assert results.portfolio_history is not None

            # Test portfolio metrics calculations
            portfolio_mod_dietz_metrics = results.portfolio_metrics[
                "mod_dietz"
            ]
            assert "total_return" in portfolio_mod_dietz_metrics
            assert "annualized_return" in portfolio_mod_dietz_metrics
            assert "volatility" in portfolio_mod_dietz_metrics
            assert "sharpe_ratio" in portfolio_mod_dietz_metrics
            assert "max_drawdown" in portfolio_mod_dietz_metrics
            assert "win_rate" in portfolio_mod_dietz_metrics

            portfolio_twr_metrics = results.portfolio_metrics["twr"]
            assert "total_return" in portfolio_twr_metrics
            assert "annualized_return" in portfolio_twr_metrics
            assert "volatility" in portfolio_twr_metrics
            assert "sharpe_ratio" in portfolio_twr_metrics
            assert "max_drawdown" in portfolio_twr_metrics
            assert "win_rate" in portfolio_twr_metrics

            # Verify expected calculations
            # Portfolio: Buy at 100 -> End at 102, no dividends
            expected_total_return = (102.00 - 100.00) / 100.00
            accuracy = 0.001 / 100  # 0.001% accuracy)

            assert (
                abs(
                    portfolio_mod_dietz_metrics["total_return"]
                    - expected_total_return
                )
                < accuracy
            )

            # Verify portfolio history was built correctly
            portfolio_history = results.portfolio_history
            # assert len(portfolio_history) == 3  # 3 trading days
            assert "total_value" in portfolio_history.columns
            assert (
                "portfolio_daily_return_mod_dietz" in portfolio_history.columns
            )

            # Check initial and final portfolio values
            initial_value = portfolio_history["total_value"].iloc[0]
            final_value = portfolio_history["total_value"].iloc[-1]

            purchase_price = 100.00
            start_price = 100.00  # price at close on first trading day
            final_price = 102.00  # price at close on final trading day

            purchase_qty = 10
            final_qty = 10

            initial_expected = purchase_qty * start_price
            final_expected = final_qty * final_price

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

            expected_daily_returns = np.array(
                [
                    purchase_price / purchase_price - 1,
                    101 / purchase_price - 1,
                    final_price / 101 - 1,
                ]
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
            ) < (
                accuracy * 1
            )  # Sharpe ratio can be larger, adjust accuracy if required

    def test_performance_with_cash_dividend(self, temp_config):
        """Test performance with a cash dividend payment."""
        transactions = [
            {
                "date": "2023-01-03",
                "ticker": "VTI",
                "transaction_type": "BUY",
                "quantity": 100,
                "value_per_share": 50.0,
                "total_value": 5000.0,
                "account": "IRA",
            },
            {
                "date": "2023-01-05",
                "ticker": "VTI",
                "transaction_type": "DIVIDEND",
                "quantity": 0,  # Cash dividend, no shares involved
                "value_per_share": 0,  # Not used for cash dividend
                "total_value": 75.0,
                "account": "IRA",
            },
        ]
        market_data = {
            "VTI": {
                "date": ["2023-01-03", "2023-01-04", "2023-01-05"],
                "close": [50.0, 51.0, 50.5],
                "dividend": [0.0, 0.0, 0.75],
            },
            "SPY": {
                "date": ["2023-01-03", "2023-01-04", "2023-01-05"],
                "close": [400.0, 401.0, 402.0],
                "dividend": [0.0, 0.0, 0.0],
            },
        }

        for analyzer, _ in self._setup_scenario(
            temp_config, transactions, market_data
        ):
            analyzer.load_transactions()
            analyzer.build_portfolio_history()
            results = analyzer.calculate_performance()

            assert results is not None
            assert results.portfolio_history is not None

            final_day = results.portfolio_history.iloc[-1]
            dividend_day = results.portfolio_history[
                results.portfolio_history["date"].dt.day == 5
            ].iloc[0]

            assert final_day["Ira_VTI_shares"] == 100  # Shares don't change
            assert (
                dividend_day["net_cash_flow"] == -75.0
            )  # Cash flow from dividend
            assert (
                final_day["total_value"] == 100 * 50.5
            )  # Final value = shares * final price

            # Verify results structure
            assert isinstance(results, PerformanceResults)
            assert results.portfolio_metrics is not None
            assert results.benchmark_metrics is not None
            assert results.relative_metrics is not None
            assert results.config is not None
            assert results.portfolio_history is not None

            # Test portfolio metrics calculations
            portfolio_mod_dietz_metrics = results.portfolio_metrics[
                "mod_dietz"
            ]
            assert "total_return" in portfolio_mod_dietz_metrics
            assert "annualized_return" in portfolio_mod_dietz_metrics
            assert "volatility" in portfolio_mod_dietz_metrics
            assert "sharpe_ratio" in portfolio_mod_dietz_metrics
            assert "max_drawdown" in portfolio_mod_dietz_metrics
            assert "win_rate" in portfolio_mod_dietz_metrics

            # Verify expected calculations
            accuracy = 0.001 / 100  # 0.001% accuracy

            # Verify portfolio history was built correctly
            portfolio_history = results.portfolio_history
            # assert len(portfolio_history) == 3  # 3 trading days
            assert "total_value" in portfolio_history.columns
            assert (
                "portfolio_daily_return_mod_dietz" in portfolio_history.columns
            )

            # Check initial and final portfolio values
            initial_value = portfolio_history["total_value"].iloc[0]
            final_value = portfolio_history["total_value"].iloc[-1]

            purchase_price = 50.00
            start_price = 50.00  # price at close on first trading day
            final_price = 50.50  # price at close on final trading day

            purchase_qty = 100
            final_qty = 100

            initial_expected_value = purchase_qty * start_price
            final_expected_value = final_qty * final_price

            assert abs(initial_value - initial_expected_value) < accuracy
            assert abs(final_value - final_expected_value) < accuracy

            # Check metrics output
            annual_trading_days = int(
                results.config.get("settings.annual_trading_days", 252)
            )

            # For market return calculations, we use the number of return days,
            # which is one less than the number of portfolio history entries
            # because returns are calculated between days.
            # However, we are calculating total return over the entire period,
            # so we use the full length of portfolio history here.

            # Cash portion of dividend; negative as cash flow out of
            # the portfolio
            dividend_amount = -75.0
            cash_flow_weight = 0.5
            previous_day_close = 51.0

            final_return_numerator = (
                final_value - previous_day_close * final_qty - dividend_amount
            )

            final_return_denominator = (
                previous_day_close * final_qty
                + cash_flow_weight * dividend_amount
            )

            expected_daily_returns = np.array(
                [
                    purchase_price / purchase_price - 1,
                    previous_day_close / purchase_price - 1,
                    final_return_numerator / final_return_denominator,
                ]
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
            ) < (
                accuracy * 1
            )  # Sharpe ratio can be larger, adjust accuracy if required

    def test_performance_with_full_reinvestment(self, temp_config):
        """Test performance with a fully reinvested dividend."""
        transactions = [
            {
                "date": "2023-01-03",
                "ticker": "VXUS",
                "transaction_type": "BUY",
                "quantity": 100,
                "value_per_share": 40.0,
                "total_value": 4000.0,
                "account": "Taxable",
            },
            {
                "date": "2023-01-05",
                "ticker": "VXUS",
                "transaction_type": "DIVIDEND",
                "quantity": 0,  # Cash dividend, no shares involved
                "value_per_share": 0,  # Not used for cash dividend
                "total_value": 82.0,
                "account": "Taxable",
            },
            {
                "date": "2023-01-05",
                "ticker": "VXUS",
                "transaction_type": "DIVIDEND_REINVEST",
                "quantity": 2.0,
                "value_per_share": 41.0,
                "total_value": -82.0,  # Reinvest are given as negative values
                "account": "Taxable",
            },
        ]
        market_data = {
            "VXUS": {
                "date": ["2023-01-03", "2023-01-04", "2023-01-05"],
                "close": [40.0, 40.5, 41.0],
                "dividend": [0.0, 0.0, 0.82],
            },
            "SPY": {
                "date": ["2023-01-03", "2023-01-04", "2023-01-05"],
                "close": [400.0, 401.0, 402.0],
                "dividend": [0.0, 0.0, 0.0],
            },
        }

        for analyzer, _ in self._setup_scenario(
            temp_config, transactions, market_data
        ):
            analyzer.load_transactions()
            analyzer.build_portfolio_history()
            results = analyzer.calculate_performance()

            assert results is not None
            final_day = results.portfolio_history.iloc[-1]
            dividend_day = results.portfolio_history[
                results.portfolio_history["date"].dt.day == 5
            ].iloc[0]

            assert (
                final_day["Taxable_VXUS_shares"] == 102
            )  # 100 initial + 2 reinvested
            assert (
                dividend_day["net_cash_flow"] == 0.0
            )  # Full reinvestment is a net zero cash flow
            assert (
                final_day["total_value"] == 102 * 41.0
            )  # Final value = final shares * final price

            # Verify results structure
            assert isinstance(results, PerformanceResults)
            assert results.portfolio_metrics is not None
            assert results.benchmark_metrics is not None
            assert results.relative_metrics is not None
            assert results.config is not None
            assert results.portfolio_history is not None

            # Test portfolio metrics calculations
            portfolio_mod_dietz_metrics = results.portfolio_metrics[
                "mod_dietz"
            ]
            assert "total_return" in portfolio_mod_dietz_metrics
            assert "annualized_return" in portfolio_mod_dietz_metrics
            assert "volatility" in portfolio_mod_dietz_metrics
            assert "sharpe_ratio" in portfolio_mod_dietz_metrics
            assert "max_drawdown" in portfolio_mod_dietz_metrics
            assert "win_rate" in portfolio_mod_dietz_metrics

            # Verify expected calculations
            accuracy = 0.001 / 100  # 0.001% accuracy

            # Verify portfolio history was built correctly
            portfolio_history = results.portfolio_history
            # assert len(portfolio_history) == 3  # 3 trading days
            assert "total_value" in portfolio_history.columns
            assert (
                "portfolio_daily_return_mod_dietz" in portfolio_history.columns
            )

            # Check initial and final portfolio values
            initial_value = portfolio_history["total_value"].iloc[0]
            final_value = portfolio_history["total_value"].iloc[-1]

            purchase_price = 40.00
            start_price = 40.00  # price at close on first trading day
            final_price = 41.00  # price at close on final trading day

            purchase_qty = 100
            final_qty = 100
            dividend_qty = 2  # Shares bought with reinvested dividend
            final_qty += dividend_qty

            initial_expected_value = purchase_qty * start_price
            final_expected_value = final_qty * final_price

            assert abs(initial_value - initial_expected_value) < accuracy
            assert abs(final_value - final_expected_value) < accuracy

            # Check metrics output
            annual_trading_days = int(
                results.config.get("settings.annual_trading_days", 252)
            )

            # For market return calculations, we use the number of return days,
            # which is one less than the number of portfolio history entries
            # because returns are calculated between days.
            # However, we are calculating total return over the entire period,
            # so we use the full length of portfolio history here.

            dividend_amount = 0
            cash_flow_weight = 0.5
            previous_day_close = 40.5

            final_return_numerator = (
                final_value
                - previous_day_close * purchase_qty
                - dividend_amount
            )

            final_return_denominator = (
                previous_day_close * purchase_qty
                + cash_flow_weight * dividend_amount
            )

            expected_daily_returns = np.array(
                [
                    purchase_price / purchase_price - 1,
                    previous_day_close / purchase_price - 1,
                    final_return_numerator / final_return_denominator,
                ]
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
            ) < (
                accuracy * 1
            )  # Sharpe ratio can be larger, adjust accuracy if required

    def test_performance_with_partial_reinvestment(self, temp_config):
        """Test performance with a dividend that is part cash, part reinvested."""

        purchase_price = 75.00
        start_price = 75.00  # price at close on first trading day
        final_price = 75.50  # price at close on final trading day

        purchase_qty = 100
        final_qty = 100

        total_purchase_value = purchase_qty * purchase_price

        total_div_amount = 125.50  # Total dividend received
        total_div_per_share = total_div_amount / purchase_qty
        # Split into cash and reinvested portions
        total_div_reinvest = final_price  # Reinvest enough to buy 1 share
        drp_quantity = total_div_reinvest / final_price  # Should be 1 share
        reinvest_div_per_share = total_div_reinvest / purchase_qty

        transactions = [
            {
                "date": "2023-01-03",
                "ticker": "BND",
                "transaction_type": "BUY",
                "quantity": purchase_qty,
                "value_per_share": purchase_price,
                "total_value": total_purchase_value,
                "account": "Roth",
            },
            {
                "date": "2023-01-05",
                "ticker": "BND",
                "transaction_type": "DIVIDEND",
                "quantity": 0,  # Cash dividend, no shares involved
                "value_per_share": 0,  # Not used for cash dividend
                "total_value": total_div_amount,
                "account": "Roth",
            },
            {
                "date": "2023-01-05",
                "ticker": "BND",
                "transaction_type": "DIVIDEND_REINVEST",
                "quantity": drp_quantity,
                "value_per_share": reinvest_div_per_share,
                "total_value": -total_div_reinvest,  # Reinvest is negative
                "account": "Roth",
            },
        ]
        market_data = {
            "BND": {
                "date": ["2023-01-03", "2023-01-04", "2023-01-05"],
                "close": [start_price, 75.2, final_price],
                "dividend": [
                    0.0,
                    0.0,
                    total_div_per_share,
                ],
            },
            "SPY": {
                "date": ["2023-01-03", "2023-01-04", "2023-01-05"],
                "close": [400.0, 401.0, 402.0],
                "dividend": [0.0, 0.0, 0.0],
            },
        }

        for analyzer, _ in self._setup_scenario(
            temp_config, transactions, market_data
        ):
            analyzer.load_transactions()
            analyzer.build_portfolio_history()
            results = analyzer.calculate_performance()

            assert results is not None
            final_day = results.portfolio_history.iloc[-1]
            dividend_day = results.portfolio_history[
                results.portfolio_history["date"].dt.day == 5
            ].iloc[0]

            # 100 initial + 1 reinvested
            assert final_day["Roth_BND_shares"] == 101

            # Net of total cash portion less reinvested amount
            # Dividend is a net cash outflow of $50, so negative cash flow
            assert dividend_day["net_cash_flow"] == -50.0

            assert final_day["total_value"] == 101 * 75.5

            # Verify results structure
            assert isinstance(results, PerformanceResults)
            assert results.portfolio_metrics is not None
            assert results.benchmark_metrics is not None
            assert results.relative_metrics is not None
            assert results.config is not None
            assert results.portfolio_history is not None

            # Test portfolio metrics calculations
            portfolio_mod_dietz_metrics = results.portfolio_metrics[
                "mod_dietz"
            ]
            assert "total_return" in portfolio_mod_dietz_metrics
            assert "annualized_return" in portfolio_mod_dietz_metrics
            assert "volatility" in portfolio_mod_dietz_metrics
            assert "sharpe_ratio" in portfolio_mod_dietz_metrics
            assert "max_drawdown" in portfolio_mod_dietz_metrics
            assert "win_rate" in portfolio_mod_dietz_metrics

            # Verify expected calculations
            accuracy = 0.001 / 100  # 0.001% accuracy

            # Verify portfolio history was built correctly
            portfolio_history = results.portfolio_history
            # assert len(portfolio_history) == 3  # 3 trading days
            assert "total_value" in portfolio_history.columns
            assert (
                "portfolio_daily_return_mod_dietz" in portfolio_history.columns
            )

            # Check initial and final portfolio values
            initial_value = portfolio_history["total_value"].iloc[0]
            final_value = portfolio_history["total_value"].iloc[-1]

            final_qty += drp_quantity

            initial_expected_value = purchase_qty * start_price
            final_expected_value = final_qty * final_price

            assert abs(initial_value - initial_expected_value) < accuracy
            assert abs(final_value - final_expected_value) < accuracy

            # Check metrics output
            annual_trading_days = int(
                results.config.get("settings.annual_trading_days", 252)
            )

            # For market return calculations, we use the number of return days,
            # which is one less than the number of portfolio history entries
            # because returns are calculated between days.
            # However, we are calculating total return over the entire period,
            # so we use the full length of portfolio history here.

            # Cash portion of dividend; negative as cash flow out of the
            # portfolio
            cash_dividend_amount = -50.0
            cash_flow_weight = 0.5
            previous_day_close = 75.2

            final_return_numerator = (
                final_value
                - previous_day_close * purchase_qty
                - cash_dividend_amount
            )

            final_return_denominator = (
                previous_day_close * purchase_qty
                + cash_flow_weight * cash_dividend_amount
            )

            expected_daily_returns = np.array(
                [
                    purchase_price / purchase_price - 1,
                    previous_day_close / purchase_price - 1,
                    final_return_numerator / final_return_denominator,
                ]
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
            ) < (
                accuracy * 1
            )  # Sharpe ratio can be larger, adjust accuracy if required
