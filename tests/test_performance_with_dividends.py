"""
Tests for performance calculation under various dividend scenarios.

This suite verifies that the end-to-end performance metrics are calculated
correctly when dividends are paid in cash, fully reinvested, or partially
reinvested.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from boglebench.core.portfolio import BogleBenchAnalyzer
from boglebench.core.results import PerformanceResults
from boglebench.utils.config import ConfigManager

SPY_MARKET_DATA = pd.DataFrame(
    {
        "date": pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"]),
        "close": [400.0, 401.0, 402.0],
        "adj_close": [400.0, 401.0, 402.0],
        "dividend": [0.0, 0.0, 0.0],
        "split_coefficient": [0.0, 0.0, 0.0],
    }
)


def scenario_no_dividends():
    """Simple scenario with a single BUY and no dividends."""
    transactions = pd.DataFrame(
        [
            {
                "date": "2023-01-03",
                "symbol": "VOO",
                "transaction_type": "BUY",
                "quantity": 10,
                "value_per_share": 100.00,
                "total_value": 1000.00,
                "account": "Taxable",
            }
        ]
    )
    market_data = {
        "VOO": pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2023-01-03", "2023-01-04", "2023-01-05"]
                ),
                "close": [100.0, 101.0, 102.0],
                "adj_close": [100.0, 101.0, 102.0],
                "dividend": [0.0, 0.0, 0.0],
                "split_coefficient": [0.0, 0.0, 0.0],
            }
        ),
        "SPY": SPY_MARKET_DATA,
    }

    return "no_dividends", transactions, market_data


def scenario_cash_dividends():
    """Simple scenario with a single BUY and cash dividends."""
    transactions = pd.DataFrame(
        [
            {
                "date": "2023-01-03",
                "symbol": "VTI",
                "transaction_type": "BUY",
                "quantity": 100,
                "value_per_share": 50.0,
                "total_value": 5000.0,
                "account": "IRA",
            },
            {
                "date": "2023-01-05",
                "symbol": "VTI",
                "transaction_type": "DIVIDEND",
                "quantity": 0,  # Cash dividend, no shares involved
                "value_per_share": 0,  # Not used for cash dividend
                "total_value": 75.0,
                "account": "IRA",
            },
        ]
    )
    market_data = {
        "VTI": pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2023-01-03", "2023-01-04", "2023-01-05"]
                ),
                "close": [50.0, 51.0, 50.5],
                "adj_close": [50.0, 51.0, 50.5],
                "dividend": [0.0, 0.0, 0.75],
                "split_coefficient": [0.0, 0.0, 0.0],
            }
        ),
        "SPY": SPY_MARKET_DATA,
    }

    return "cash_dividends", transactions, market_data


def scenario_full_reinvestment():
    """Simple scenario with a single BUY and fully reinvested dividends."""
    transactions = pd.DataFrame(
        [
            {
                "date": "2023-01-03",
                "symbol": "VXUS",
                "transaction_type": "BUY",
                "quantity": 100,
                "value_per_share": 40.0,
                "total_value": 4000.0,
                "account": "Taxable",
            },
            {
                "date": "2023-01-05",
                "symbol": "VXUS",
                "transaction_type": "DIVIDEND",
                "quantity": 0,  # Cash dividend, no shares involved
                "value_per_share": 0,  # Not used for cash dividend
                "total_value": 82.0,
                "account": "Taxable",
            },
            {
                "date": "2023-01-05",
                "symbol": "VXUS",
                "transaction_type": "DIVIDEND_REINVEST",
                "quantity": 2.0,
                "value_per_share": 41.0,
                "total_value": -82.0,  # Reinvest are given as negative values
                "account": "Taxable",
            },
        ]
    )
    market_data = {
        "VXUS": pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2023-01-03", "2023-01-04", "2023-01-05"]
                ),
                "close": [40.0, 40.5, 41.0],
                "adj_close": [40.0, 40.5, 41.0],
                "dividend": [0.0, 0.0, 0.82],
                "split_coefficient": [0.0, 0.0, 0.0],
            }
        ),
        "SPY": SPY_MARKET_DATA,
    }

    return "full_reinvestment", transactions, market_data


def scenario_partial_reinvestment():
    """Simple scenario with a single BUY and partially reinvested dividends."""
    transactions = pd.DataFrame(
        [
            {
                "date": "2023-01-03",
                "symbol": "BND",
                "transaction_type": "BUY",
                "quantity": 100,
                "value_per_share": 75,
                "total_value": 100 * 75,
                "account": "Roth",
            },
            {
                "date": "2023-01-05",
                "symbol": "BND",
                "transaction_type": "DIVIDEND",
                "quantity": 0,  # Cash dividend, no shares involved
                "value_per_share": 0,  # Not used for cash dividend
                "total_value": 125.50,
                "account": "Roth",
            },
            {
                "date": "2023-01-05",
                "symbol": "BND",
                "transaction_type": "DIVIDEND_REINVEST",
                "quantity": 1.0,  # Reinvest enough to buy 1 share
                "value_per_share": 0.755,
                "total_value": -75.50,  # Reinvest is negative
                "account": "Roth",
            },
        ]
    )
    market_data = {
        "BND": pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2023-01-03", "2023-01-04", "2023-01-05"]
                ),
                "close": [75, 75.2, 75.5],
                "adj_close": [75, 75.2, 75.5],
                "dividend": [
                    0.0,
                    0.0,
                    1.255,
                ],
                "split_coefficient": [0.0, 0.0, 0.0],
            }
        ),
        "SPY": SPY_MARKET_DATA,
    }

    return "partial_reinvestment", transactions, market_data


class TestPerformanceWithDividends:
    """Test suite for performance calculations involving dividends."""

    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            (config_dir / "input").mkdir()
            (config_dir / "market_data").mkdir()
            (config_dir / "output").mkdir()

            config = ConfigManager()
            config.config["data"]["base_path"] = str(config_dir)
            config.config["database"]["db_path"] = ":memory:"

            # Updating default settings
            config.config["settings"]["cache_market_data"] = True
            config.config["settings"]["force_refresh_market_data"] = False

            config.config["analysis"]["start_date"] = "2023-01-03"
            config.config["analysis"]["end_date"] = "2023-01-05"

            yield config

    @pytest.fixture(
        params=[
            scenario_no_dividends(),
            scenario_cash_dividends(),
            scenario_full_reinvestment(),
            scenario_partial_reinvestment(),
        ],
        ids=lambda x: x[0],  # Use scenario name for test ID
    )
    def scenario_analyzer(self, request, temp_config, monkeypatch):
        """Fixture to set up BogleBenchAnalyzer for a given dividend
        scenario."""
        scenario_name, transactions_df, market_data_dict = request.param

        # Save transactions to csv
        temp_data_path = temp_config.get_data_path()
        transactions_file_path = temp_data_path / "input" / "transactions.csv"
        transactions_df.to_csv(transactions_file_path, index=False)

        market_data_path = temp_config.get_market_data_path()
        for symbol, df in market_data_dict.items():
            df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
            df.to_parquet(market_data_path / f"{symbol}.parquet", index=False)

        output_path = temp_config.get_output_path()

        monkeypatch.setattr(
            ConfigManager,
            "get_data_path",
            lambda self, subdir=None: (
                temp_data_path / subdir if subdir else temp_data_path
            ),
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

        yield analyzer, scenario_name, transactions_file_path

    def test_performance_dividends_scenarios(self, scenario_analyzer):
        """Test performance calculation across various dividend scenarios."""
        analyzer, scenario_name, transactions_file = scenario_analyzer

        # --- Main Workflow ---
        analyzer.load_transactions(transactions_file)
        results = analyzer.calculate_performance()

        accuracy = 0.001 / 100  # 0.001% accuracy)
        annual_trading_days = int(
            results.config.get(
                "advanced.performance.annualization_factor", 252
            )
        )
        risk_free_rate = analyzer.config.config.get(
            "analysis.risk_free_rate", 0.02
        )
        daily_risk_free_rate = (1 + risk_free_rate) ** (
            1 / annual_trading_days
        ) - 1

        # --- Assertions ---
        assert results is not None

        # Verify results structure
        assert isinstance(results, PerformanceResults)
        assert results.portfolio_metrics is not None
        assert results.benchmark_metrics is not None
        assert results.relative_metrics is not None
        assert results.config is not None
        assert results.portfolio_db is not None

        # Verify portfolio metrics structure
        portfolio_mod_dietz_metrics = results.portfolio_metrics["mod_dietz"]
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

        # Verify portfolio history was built correctly
        portfolio_summary = results.portfolio_db.get_portfolio_summary()
        assert len(portfolio_summary) == 3  # 3 trading days
        assert "total_value" in portfolio_summary.columns
        assert "portfolio_mod_dietz_return" in portfolio_summary.columns

        initial_value = portfolio_summary["total_value"].iloc[0]
        final_value = portfolio_summary["total_value"].iloc[-1]
        final_day_holdings = results.portfolio_db.get_latest_holdings(
            include_zero=True
        )
        _, end_date = results.portfolio_db.get_date_range()

        if scenario_name == "no_dividends":
            sym = "VOO"
            final_day_holding_symbol = final_day_holdings[
                final_day_holdings["symbol"] == sym
            ].iloc[0]
            assert final_day_holding_symbol["quantity"] == 10
            assert final_day_holding_symbol["value"] == 10 * 102.0

            final_day_symbol = results.portfolio_db.get_symbol_data(
                symbol=sym, start_date=end_date, end_date=end_date
            )
            assert final_day_symbol["price"].iloc[0] == 102

            # Check initial and final portfolio values
            purchase_price = 100.00
            start_price = 100.00  # price at close on first trading day
            final_price = 102.00  # price at close on final trading day

            purchase_qty = 10
            final_qty = 10

            initial_expected = purchase_qty * start_price
            final_expected = final_qty * final_price

            assert abs(initial_value - initial_expected) < accuracy
            assert abs(final_value - final_expected) < accuracy

            # Verify expected calculations
            # Portfolio: Buy at 100 -> End at 102, no dividends
            expected_total_return = (102.00 - 100.00) / 100.00
            assert (
                abs(
                    portfolio_mod_dietz_metrics["total_return"]
                    - expected_total_return
                )
                < accuracy
            )

            # For market return calculations, we use the number of return days,
            # which is one less than the number of portfolio history entries
            # because returns are calculated between days.
            # However, we are calculating total return over the entire period,
            # so we use the full length of portfolio summary here.
            return_days = len(portfolio_summary)

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

            expected_sharpe_ratio = (
                expected_daily_mean_returns - daily_risk_free_rate
            ) / expected_volatility
            expected_annual_sharpe_ratio = expected_sharpe_ratio * np.sqrt(
                annual_trading_days
            )
            assert (
                abs(
                    portfolio_mod_dietz_metrics["sharpe_ratio"]
                    - expected_annual_sharpe_ratio
                )
                < accuracy
            )

        elif scenario_name == "cash_dividends":
            acct = "Ira"
            final_day_holding_acct = final_day_holdings[
                final_day_holdings["account"] == acct
            ].iloc[0]
            assert (
                final_day_holding_acct["quantity"] == 100
            )  # Shares don't change
            assert (
                final_day_holding_acct["value"] == 100 * 50.5
            )  # Final value = shares * final price

            final_day_account = results.portfolio_db.get_account_data(
                account=acct, start_date=end_date, end_date=end_date
            )
            assert (
                final_day_account["cash_flow"].iloc[0] == -75.0
            )  # Cash flow from dividend

            # Check initial and final portfolio values
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

            expected_sharpe_ratio = (
                expected_daily_mean_returns - daily_risk_free_rate
            ) / expected_volatility
            expected_annual_sharpe_ratio = expected_sharpe_ratio * np.sqrt(
                annual_trading_days
            )
            assert (
                abs(
                    portfolio_mod_dietz_metrics["sharpe_ratio"]
                    - expected_annual_sharpe_ratio
                )
                < accuracy
            )

        elif scenario_name == "full_reinvestment":
            final_day = results.portfolio_db.get_holdings(
                account="Taxable",
                symbol="VXUS",
                date=end_date,
                include_zero=True,
            ).iloc[0]
            assert final_day["quantity"] == 102  # 100 initial + 2 reinvested
            assert (
                final_day["value"] == 102 * 41.0
            )  # Final value = final shares * final price

            # Check initial and final portfolio values
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

            expected_sharpe_ratio = (
                expected_daily_mean_returns - daily_risk_free_rate
            ) / expected_volatility
            expected_annual_sharpe_ratio = expected_sharpe_ratio * np.sqrt(
                annual_trading_days
            )
            assert (
                abs(
                    portfolio_mod_dietz_metrics["sharpe_ratio"]
                    - expected_annual_sharpe_ratio
                )
                < accuracy
            )

        elif scenario_name == "partial_reinvestment":
            final_day = results.portfolio_db.get_holdings(
                account="Roth",
                symbol="BND",
                date=end_date,
                include_zero=True,
            ).iloc[0]
            assert final_day["quantity"] == 101  # 100 initial + 1 reinvested
            assert (
                final_day["value"] == 101 * 75.5
            )  # Final value = final shares * final price

            dividend_day = results.portfolio_db.get_cash_flows(
                accounts=["Roth"], start_date=end_date, end_date=end_date
            ).iloc[0]
            assert dividend_day["cash_flow"] == -50.0

            # Check initial and final portfolio values
            purchase_price = 75.00
            start_price = 75.00  # price at close on first trading day
            final_price = 75.50  # price at close on final trading day

            purchase_qty = 100
            final_qty = 100
            dividend_qty = 1  # Shares bought with reinvested dividend
            final_qty += dividend_qty

            initial_expected_value = purchase_qty * start_price
            final_expected_value = final_qty * final_price

            assert abs(initial_value - initial_expected_value) < accuracy
            assert abs(final_value - final_expected_value) < accuracy

            # Check metrics output
            # Cash portion of dividend; negative as cash flow out of
            # the portfolio
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

        else:
            raise ValueError(f"Unknown scenario: {scenario_name}")
