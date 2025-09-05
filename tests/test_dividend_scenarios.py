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

# def create_test_config(workspace: Path) -> Path:
#     """Creates a standard config.yaml file in the test workspace."""
#     config_file = workspace / "config.yaml"
#     config_file.write_text(
#         """
# settings:
#   benchmark_ticker: "SPY"
#   cache_market_data: false
#   annual_trading_days: 252
#   risk_free_rate: 0.02
# paths:
#   transactions: "transactions.csv"
# api:
#   alpha_vantage_key: "DUMMY_KEY"
# advanced:
#   performance:
#     period_cash_flow_weight: 0.5
# """
#     )
#     return config_file


# --- Scenario Data Generators ---


def scenario_single_stock_cash_dividend():
    """
    Scenario: Buy one stock, receive a full cash dividend.
    - Verifies correct cash flow and that share count remains unchanged.
    """
    transactions = pd.DataFrame(
        [
            {
                "date": "2023-01-02",
                "ticker": "TICKA",
                "transaction_type": "BUY",
                "shares": 100,
                "price_per_share": 10.00,
                "amount": 1000.00,
                "account": "Taxable",
            },
            {
                "date": "2023-01-04",
                "ticker": "TICKA",
                "transaction_type": "DIVIDEND",
                "shares": 0,
                "price_per_share": 0,
                "amount": 50.00,  # $0.50/share dividend
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
                "dividend": [0.0, 0.0, 0.50, 0.0],
            }
        ),
        "SPY": pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
                ),
                "close": [400.0, 401.0, 402.0, 403.0],
                "dividend": [0.0, 0.0, 0.0, 0.0],
            }
        ),
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
                "ticker": "TICKB",
                "transaction_type": "BUY",
                "shares": 100,
                "price_per_share": 20.00,
                "amount": 2000.00,
                "account": "IRA",
            },
            {
                "date": "2023-01-04",
                "ticker": "TICKB",
                "transaction_type": "DIVIDEND_REINVEST",
                "shares": 2.5,  # $50 dividend / $20 share price = 2.5 shares
                "price_per_share": 20.0,
                "amount": 50.00,
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
                "dividend": [0.0, 0.0, 0.50, 0.0],
            }
        ),
        "SPY": pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
                ),
                "close": [400.0, 401.0, 402.0, 403.0],
                "dividend": [0.0, 0.0, 0.0, 0.0],
            }
        ),
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
                "ticker": "TICKC",
                "transaction_type": "BUY",
                "shares": 100,
                "price_per_share": 30.00,
                "amount": 3000.00,
                "account": "Taxable",
            },
            {
                "date": "2023-01-04",
                "ticker": "TICKC",
                "transaction_type": "DIVIDEND",  # $25 cash portion
                "shares": 0,
                "price_per_share": 0,
                "amount": 25.00,
                "account": "Taxable",
            },
            {
                "date": "2023-01-04",
                "ticker": "TICKC",
                # $75 reinvested portion
                "transaction_type": "DIVIDEND_REINVEST",
                "shares": 2.5,  # $75 / $30 share price
                "price_per_share": 30.0,
                "amount": 75.00,
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
                "dividend": [0.0, 0.0, 1.00, 0.0],  # $1/share total dividend
            }
        ),
        "SPY": pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
                ),
                "close": [400.0, 401.0, 402.0, 403.0],
                "dividend": [0.0, 0.0, 0.0, 0.0],
            }
        ),
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
                "ticker": "TICKD",
                "transaction_type": "BUY",
                "shares": 200,
                "price_per_share": 10.00,
                "amount": 2000.00,
                "account": "Taxable",
            },
            {
                "date": "2023-01-03",
                "ticker": "TICKD",
                "transaction_type": "SELL",
                "shares": 50,  # Sell 50 shares
                "price_per_share": 10.10,
                "amount": 505.00,
                "account": "Taxable",
            },
            {
                "date": "2023-01-05",
                "ticker": "TICKD",
                "transaction_type": "DIVIDEND",
                "shares": 0,
                "price_per_share": 0,
                "amount": 75.00,  # $0.50/share on remaining 150 shares
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
                "dividend": [0.0, 0.0, 0.0, 0.50],
            }
        ),
        "SPY": pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
                ),
                "close": [400.0, 401.0, 402.0, 403.0],
                "dividend": [0.0, 0.0, 0.0, 0.0],
            }
        ),
    }
    return "partial_sale", transactions, market_data


class TestDividendScenarios:
    """A test class for various end-to-end dividend scenarios."""

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

            yield config

    @pytest.fixture
    def test_data_dir(self):
        """Path to test data directory."""
        return Path(__file__).parent / "test_data"

    @pytest.fixture(
        params=[
            scenario_single_stock_cash_dividend(),
            scenario_single_stock_full_reinvestment(),
            scenario_single_stock_partial_reinvestment(),
            scenario_dividend_after_partial_sale(),
        ],
        ids=lambda x: x[0],  # Use scenario name for test ID
    )
    def scenario_analyzer(self, request, temp_config):
        """Fixture to set up BogleBenchAnalyzer for a given dividend
        scenario."""
        _, transactions_df, market_data_dict = request.param

        # Create analyzer
        analyzer = BogleBenchAnalyzer()
        analyzer.config = temp_config

        # Save transactions to csv
        workspace = analyzer.config.get_data_path()
        transactions_file = workspace / "transactions.csv"
        transactions_df.to_csv(transactions_file, index=False)

        # Mock the market data fetching
        analyzer.market_data = market_data_dict
        for df in analyzer.market_data.values():
            df["date"] = pd.to_datetime(df["date"], utc=True)
            for col in [
                "open",
                "high",
                "low",
                "adj_close",
                "volume",
                "split_coefficient",
            ]:
                if col not in df.columns:
                    if col == "adj_close":
                        df[col] = df["close"]
                    else:
                        df[col] = 0
        analyzer.benchmark_data = analyzer.market_data["SPY"]

        # Yield analyzer and scenario name
        yield analyzer, request.param[0], transactions_file

    def test_dividend_scenarios(self, scenario_analyzer):
        """
        Runs a dividend scenario and performs assertions based on the scenario
        type.
        """
        analyzer, scenario_name, transactions_file = scenario_analyzer

        # --- Main Workflow ---
        analyzer.load_transactions(transactions_file)
        portfolio_df = analyzer.build_portfolio_history()
        results = analyzer.calculate_performance()

        # --- Assertions ---
        assert results is not None
        assert "total_return" in results.portfolio_metrics
        assert "total_return" in results.benchmark_metrics

        final_day = portfolio_df.iloc[-1]

        if scenario_name == "cash_dividend":
            assert final_day["Taxable_TICKA_shares"] == 100
            # Cash dividend is a positive cash flow
            dividend_day_flow = portfolio_df[
                portfolio_df["date"].dt.date
                == pd.to_datetime("2023-01-04").date()
            ]["net_cash_flow"].sum()
            assert dividend_day_flow == 50.00

        elif scenario_name == "full_reinvest":
            assert final_day["Ira_TICKB_shares"] == 102.5
            # Reinvestment is also treated as a positive cash flow (dividend
            # in, buy out)
            dividend_day_flow = portfolio_df[
                portfolio_df["date"].dt.date
                == pd.to_datetime("2023-01-04").date()
            ]["net_cash_flow"].sum()
            assert dividend_day_flow == 50.00

        elif scenario_name == "partial_reinvest":
            assert final_day["Taxable_TICKC_shares"] == 102.5
            # Total cash flow is the sum of the cash and reinvested portions
            dividend_day_flow = portfolio_df[
                portfolio_df["date"].dt.date
                == pd.to_datetime("2023-01-04").date()
            ]["net_cash_flow"].sum()
            assert dividend_day_flow == 100.00  # 25 cash + 75 reinvest

        elif scenario_name == "partial_sale":
            assert final_day["Taxable_TICKD_shares"] == 150
            dividend_day_flow = portfolio_df[
                portfolio_df["date"].dt.date
                == pd.to_datetime("2023-01-05").date()
            ]["net_cash_flow"].sum()
            assert dividend_day_flow == 75.00
