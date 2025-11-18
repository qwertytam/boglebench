"""
Comprehensive integration tests for various dividend payment and reinvestment
scenarios.

These tests verify the end-to-end impact of different dividend workflows on
portfolio holdings, cash flow, and performance metrics.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from boglebench.core.portfolio import BogleBenchAnalyzer
from boglebench.utils.config import ConfigManager

# --- Scenario Data Generators ---

SPY_MARKET_DATA = pd.DataFrame(
    {
        "date": pd.to_datetime(
            ["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
        ),
        "close": [400.0, 401.0, 402.0, 403.0],
        "adj_close": [400.0, 401.0, 402.0, 403.0],
        "dividend": [0.0, 0.0, 0.0, 0.0],
        "split_coefficient": [0.0, 0.0, 0.0, 0.0],
    }
)


def scenario_single_stock_cash_dividend():
    """
    Scenario: Buy one stock, receive a full cash dividend.
    - Verifies correct cash flow and that share count remains unchanged.
    """
    transactions = pd.DataFrame(
        [
            {
                "date": "2023-01-02",
                "symbol": "TICKA",
                "transaction_type": "BUY",
                "quantity": 100,
                "value_per_share": 10.00,
                "total_value": 1000.00,
                "account": "Taxable",
            },
            {
                "date": "2023-01-04",
                "symbol": "TICKA",
                "transaction_type": "DIVIDEND",
                "quantity": 0,  # Cash dividend, no shares involved
                "value_per_share": 0,  # Not used for cash dividend
                "total_value": 50.00,  # $0.50/share dividend
                "account": "Taxable",
            },
        ]
    )
    market_data = {
        "TICKA": pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
                ),
                "close": [10.0, 10.1, 10.2, 10.3],
                "adj_close": [10.0, 10.1, 10.2, 10.3],
                "dividend": [0.0, 0.0, 0.50, 0.0],
                "split_coefficient": [0.0, 0.0, 0.0, 0.0],
            }
        ),
        "SPY": SPY_MARKET_DATA,
    }
    return "cash_dividend", transactions, market_data


def scenario_single_stock_full_reinvestment():
    """
    Scenario: Buy one stock, receive a fully reinvested dividend.
    - Verifies that share count increases correctly.
    """
    transactions = pd.DataFrame(
        [
            {
                "date": "2023-01-02",
                "symbol": "TICKB",
                "transaction_type": "BUY",
                "quantity": 100,
                "value_per_share": 20.00,
                "total_value": 2000.00,
                "account": "IRA",
            },
            {
                "date": "2023-01-04",
                "symbol": "TICKB",
                "transaction_type": "DIVIDEND",
                "quantity": 0,  # Cash dividend, no shares involved
                "value_per_share": 0,  # Not used for cash dividend
                "total_value": 50.00,  # $0.50 dividend per share * 100 shares
                "account": "IRA",
            },
            {
                "date": "2023-01-04",
                "symbol": "TICKB",
                "transaction_type": "DIVIDEND_REINVEST",
                "quantity": 2.5,  # $50 dividend / $20 share price = 2.5 shares
                "value_per_share": 20.0,
                "total_value": 50.00,
                "account": "IRA",
            },
        ]
    )
    market_data = {
        "TICKB": pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
                ),
                "close": [20.0, 20.1, 20.0, 20.2],
                "adj_close": [20.0, 20.1, 20.0, 20.2],
                "dividend": [0.0, 0.0, 0.50, 0.0],
                "split_coefficient": [0.0, 0.0, 0.0, 0.0],
            }
        ),
        "SPY": SPY_MARKET_DATA,
    }
    return "full_reinvest", transactions, market_data


def scenario_single_stock_partial_reinvestment():
    """
    Scenario: Buy one stock, receive a dividend partially in cash, partially
    reinvested.
    - This is modeled as two separate transactions on the same day.
    - Verifies share count increases and cash flow is correctly recorded.
    """
    transactions = pd.DataFrame(
        [
            {
                "date": "2023-01-02",
                "symbol": "TICKC",
                "transaction_type": "BUY",
                "quantity": 100,
                "value_per_share": 30.00,
                "total_value": 3000.00,
                "account": "Taxable",
            },
            {
                "date": "2023-01-04",
                "symbol": "TICKC",
                "transaction_type": "DIVIDEND",  # $100 total value
                "quantity": 0,  # Cash dividend, no shares involved
                "value_per_share": 0,  # Not used for cash dividend
                "total_value": 100.00,
                "account": "Taxable",
            },
            {
                "date": "2023-01-04",
                "symbol": "TICKC",
                # $75 reinvested portion
                "transaction_type": "DIVIDEND_REINVEST",
                "quantity": 2.5,  # $75 / $30 share price
                "value_per_share": 30.0,
                "total_value": 75.00,
                "account": "Taxable",
            },
        ]
    )
    market_data = {
        "TICKC": pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
                ),
                "close": [30.0, 30.1, 30.0, 30.2],
                "adj_close": [30.0, 30.1, 30.0, 30.2],
                "dividend": [0.0, 0.0, 1.00, 0.0],  # $1/share total dividend
                "split_coefficient": [0.0, 0.0, 0.0, 0.0],
            }
        ),
        "SPY": SPY_MARKET_DATA,
    }
    return "partial_reinvest", transactions, market_data


def scenario_dividend_after_partial_sale():
    """
    Scenario: Buy stock, sell some, then receive a dividend.
    - Verifies dividend is calculated on the correct remaining number of
    shares.
    """
    transactions = pd.DataFrame(
        [
            {
                "date": "2023-01-02",
                "symbol": "TICKD",
                "transaction_type": "BUY",
                "quantity": 200,
                "value_per_share": 10.00,
                "total_value": 2000.00,
                "account": "Taxable",
            },
            {
                "date": "2023-01-03",
                "symbol": "TICKD",
                "transaction_type": "SELL",
                "quantity": 50,  # Sell 50 shares
                "value_per_share": 10.10,
                "total_value": 505.00,
                "account": "Taxable",
            },
            {
                "date": "2023-01-05",
                "symbol": "TICKD",
                "transaction_type": "DIVIDEND",
                "quantity": 0,  # Cash dividend, no shares involved
                "value_per_share": 0,  # Not used for cash dividend
                "total_value": 75.00,  # $0.50/share on remaining 150 shares
                "account": "Taxable",
            },
        ]
    )
    market_data = {
        "TICKD": pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
                ),
                "close": [10.0, 10.1, 10.2, 10.3],
                "adj_close": [10.0, 10.1, 10.2, 10.3],
                "dividend": [0.0, 0.0, 0.0, 0.50],
                "split_coefficient": [0.0, 0.0, 0.0, 0.0],
            }
        ),
        "SPY": SPY_MARKET_DATA,
    }
    return "partial_sale", transactions, market_data


class TestDividendScenarios:
    """A test class for various end-to-end dividend scenarios."""

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
            config.config["database"]["db_path"] = ":memory:"
            config.config["analysis"]["start_date"] = "2023-01-02"
            config.config["analysis"]["end_date"] = "2023-01-05"

            config.config["settings"]["cache_market_data"] = True
            config.config["settings"]["force_refresh_market_data"] = False

            # Setting to 1.0 for ease of comparing total returns
            config.config["advanced"]["performance"][
                "modified_dietz_periodic_cash_flow_weight"
            ] = 1.0

            yield config

    @pytest.fixture(
        params=[
            scenario_single_stock_cash_dividend(),
            scenario_single_stock_full_reinvestment(),
            scenario_single_stock_partial_reinvestment(),
            scenario_dividend_after_partial_sale(),
        ],
        ids=lambda x: x[0],  # Use scenario name for test ID
    )
    def scenario_analyzer(self, request, temp_config, monkeypatch):
        """Fixture to set up BogleBenchAnalyzer for a given dividend
        scenario."""
        scenario_name, transactions_df, market_data_dict = request.param

        # Save transactions to csv
        temp_data_path = temp_config.get_data_path()
        transactions_file_path = (
            temp_data_path / "transactions" / "transactions.csv"
        )
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

    def test_dividend_scenarios(self, scenario_analyzer):
        """
        Runs a dividend scenario and performs assertions based on the scenario
        type.
        """
        analyzer, scenario_name, transactions_file = scenario_analyzer

        # --- Main Workflow ---
        analyzer.load_transactions(transactions_file)
        analyzer.build_portfolio_history()
        results = analyzer.calculate_performance()
        portfolio_db = results.portfolio_db

        # --- Assertions ---
        assert results is not None
        assert "total_return" in results.portfolio_metrics["mod_dietz"]
        assert "total_return" in results.portfolio_metrics["twr"]
        assert "total_return" in results.benchmark_metrics

        if scenario_name == "cash_dividend":
            account = "Taxable"
            symbol = "TICKA"
            date = pd.to_datetime("2023-01-04", utc=True)
            holdings_df = portfolio_db.get_holdings(
                account=account, symbol=symbol, date=date
            )
            assert not holdings_df.empty
            assert holdings_df["quantity"].iloc[0] == 100

            # Cash dividend is a negative cash flow
            cash_flow_df = portfolio_db.get_cash_flows(
                symbols=[symbol], accounts=[account], date=date
            )
            assert cash_flow_df["cash_flow"].sum() == -50.00

        elif scenario_name == "full_reinvest":
            account = "Ira"
            symbol = "TICKB"
            date = pd.to_datetime("2023-01-04", utc=True)
            holdings_df = portfolio_db.get_holdings(
                account=account, symbol=symbol, date=date
            )
            assert not holdings_df.empty
            assert holdings_df["quantity"].iloc[0] == 102.5

            cash_flow_df = portfolio_db.get_cash_flows(
                symbols=[symbol], accounts=[account], date=date
            )
            assert cash_flow_df["cash_flow"].sum() == 0.00

        elif scenario_name == "partial_reinvest":
            account = "Taxable"
            symbol = "TICKC"
            date = pd.to_datetime("2023-01-04", utc=True)
            holdings_df = portfolio_db.get_holdings(
                account=account, symbol=symbol, date=date
            )
            assert not holdings_df.empty
            assert holdings_df["quantity"].iloc[0] == 102.5

            # Total cash flow is the sum of the cash and reinvested portions
            cash_flow_df = portfolio_db.get_cash_flows(
                symbols=[symbol], accounts=[account], date=date
            )
            assert (
                cash_flow_df["cash_flow"].sum() == -25.00
            )  # 25 cash + 75 reinvest

        elif scenario_name == "partial_sale":
            account = "Taxable"
            symbol = "TICKD"
            date = pd.to_datetime("2023-01-05", utc=True)
            holdings_df = portfolio_db.get_holdings(
                account=account, symbol=symbol, date=date
            )
            assert not holdings_df.empty
            assert holdings_df["quantity"].iloc[0] == 150

            cash_flow_df = portfolio_db.get_cash_flows(
                symbols=[symbol], accounts=[account], date=date
            )
            assert cash_flow_df["cash_flow"].sum() == -75.00
