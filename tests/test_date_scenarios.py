"""
Comprehensive integration tests for various start and end dates.
scenarios.

- start date after end date (error case)
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from boglebench.core.portfolio import BogleBenchAnalyzer
from boglebench.utils.config import ConfigManager

# --- Transaction & Market Data Generators ---


def test_market_data():
    """
    Simple test to ensure market data generator works as expected.
    """
    dates = pd.date_range(start="2023-01-02", periods=5, freq="D")
    market_data = {
        "TICK": pd.DataFrame(
            {
                "date": dates,
                "close": [100, 101, 102, 103, 104],
                "dividend": [0, 0.5, 0, 0, 0],
            }
        ),
        "SPY": pd.DataFrame(
            {
                "date": dates,
                "close": [400, 401, 402, 403, 404],
                "dividend": [0, 0, 0, 0, 0],
            }
        ),
    }

    return market_data


def test_transactions():
    """
    Simple test to ensure transaction data generator works as expected.
    """
    transactions = pd.DataFrame(
        [
            {
                "date": "2023-01-02",
                "ticker": "TICK",
                "transaction_type": "BUY",
                "quantity": 10,
                "value_per_share": 100.00,
                "total_value": 1000.00,
                "account": "Taxable",
            },
            {
                "date": "2023-01-04",
                "ticker": "TICK",
                "transaction_type": "DIVIDEND",
                "quantity": 0,  # Cash dividend, no shares involved
                "value_per_share": 0,  # Not used for cash dividend
                "total_value": 50.00,  # $0.50/share dividend
                "account": "Taxable",
            },
            {
                "date": "2023-01-05",
                "ticker": "TICK",
                "transaction_type": "BUY",
                "quantity": 10,
                "value_per_share": 1030,
                "total_value": 1030.00,
                "account": "Taxable",
            },
            {
                "date": "2023-01-06",
                "ticker": "TICK",
                "transaction_type": "SELL",
                "quantity": 10,
                "value_per_share": 104,
                "total_value": 1040.00,
                "account": "Taxable",
            },
        ]
    )
    return transactions


# --- Scenario Data Generators ---


def scenario_user_provided_start_and_end_equal_transaction_dates(
    test_market_data, test_transactions
):
    """
    Scenario: User provides both start and end dates.
    - Verifies all transactions are included in the analysis.
    - Verifies only market data within the date range is used.
    - Verifies performance metrics are calculated correctly.
    - Verifies portfolio value at start and end dates.
    """
    # Provide explicit start and end dates matching the first and last transaction dates
    start_date = "2023-01-02"
    end_date = "2023-01-06"

    # Attach these dates to the returned tuple for use by the test harness
    return (
        "user_start_end_equal_transaction_dates",
        test_transactions,
        test_market_data,
        start_date,
        end_date,
    )


def scenario_no_user_provided_start_and_end_dates(
    test_market_data, test_transactions
):
    """
    Scenario: User does not provide start and end dates.
    - Verifies all transactions are included in the analysis.
    - Verifies market data from first transaction date to latest available date is used.
    - Verifies performance metrics are calculated correctly.
    - Verifies portfolio value at start and end dates.
    """
    # No explicit start and end dates provided
    start_date = None
    end_date = None

    # Attach these dates to the returned tuple for use by the test harness
    return (
        "no_user_start_end_dates",
        test_transactions,
        test_market_data,
        start_date,
        end_date,
    )


def scenario_user_provided_start_and_end_inside_transaction_dates(
    test_market_data, test_transactions
):
    """
    Scenario: User provides both start and end dates that are inside the
    transaction date range.
    """
    # Provide explicit start and end dates inside the transaction date range
    start_date = "2023-01-03"
    end_date = "2023-01-05"

    # Attach these dates to the returned tuple for use by the test harness
    return (
        "user_start_end_inside_transaction_dates",
        test_transactions,
        test_market_data,
        start_date,
        end_date,
    )


def scenario_user_end_date_only_transaction_dates(
    test_market_data, test_transactions
):
    """
    Scenario: User provides only an end date inside the
    transaction date range.
    """
    # Provide explicit start and end dates inside the transaction date range
    start_date = None
    end_date = "2023-01-06"

    # Attach these dates to the returned tuple for use by the test harness
    return (
        "user_end_date_only_transaction_dates",
        test_transactions,
        test_market_data,
        start_date,
        end_date,
    )


def scenario_user_start_end_dates_outside_transaction_dates(
    test_market_data, test_transactions
):
    """
    Scenario: User provides a start and end dates outside the
    transaction date range.
    """
    # Provide explicit start and end dates outside the transaction date range
    start_date = "2022-12-30"
    end_date = "2023-01-09"

    # Attach these dates to the returned tuple for use by the test harness
    return (
        "user_start_end_dates_outside_transaction_dates",
        test_transactions,
        test_market_data,
        start_date,
        end_date,
    )


def scenario_user_dates_outside_transaction_dates(
    test_market_data, test_transactions
):
    """
    Scenario: User provides a start date after the end date.
    - Verifies an appropriate error is raised.
    """
    # Provide explicit start and end dates outside the transaction date range
    start_date = "2023-01-10"
    end_date = "2023-01-15"

    # Attach these dates to the returned tuple for use by the test harness
    return (
        "user_dates_outside_transaction_dates",
        test_transactions,
        test_market_data,
        start_date,
        end_date,
    )


def user_start_date_after_end_date(test_market_data, test_transactions):
    """
    Scenario: User provides a start date after the end date.
    - Verifies an appropriate error is raised.
    """
    # Provide explicit start and end dates outside the transaction date range
    start_date = "2023-01-10"
    end_date = "2023-01-05"

    # Attach these dates to the returned tuple for use by the test harness
    return (
        "user_start_date_after_end_date",
        test_transactions,
        test_market_data,
        start_date,
        end_date,
    )


class TestDateScenarios:
    """A test class for various end-to-end date range scenarios."""

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

            config.config["analysis"]["start_date"] = None
            config.config["analysis"]["end_date"] = None

            yield config

    @pytest.fixture(
        params=[
            scenario_user_provided_start_and_end_equal_transaction_dates(
                test_market_data(), test_transactions()
            ),
            scenario_no_user_provided_start_and_end_dates(
                test_market_data(), test_transactions()
            ),
            scenario_user_provided_start_and_end_inside_transaction_dates(
                test_market_data(), test_transactions()
            ),
            scenario_user_end_date_only_transaction_dates(
                test_market_data(), test_transactions()
            ),
            scenario_user_start_end_dates_outside_transaction_dates(
                test_market_data(), test_transactions()
            ),
            scenario_user_dates_outside_transaction_dates(
                test_market_data(), test_transactions()
            ),
            user_start_date_after_end_date(
                test_market_data(), test_transactions()
            ),
        ],
        ids=lambda x: x[0],  # Use scenario name for test ID
    )
    def scenario_analyzer(self, request, temp_config, monkeypatch):
        """Fixture to set up BogleBenchAnalyzer for a given date range
        scenario."""

        _, transactions_df, market_data_dict, start_date, end_date = (
            request.param
        )

        # --- Mocks ---
        # Mock methods on the BogleBenchAnalyzer class to control date logic
        monkeypatch.setattr(
            BogleBenchAnalyzer, "_is_market_currently_open", lambda self: True
        )
        monkeypatch.setattr(
            BogleBenchAnalyzer,
            "_get_last_closed_market_day",
            lambda self: pd.to_datetime("2023-01-10", utc=True).date(),
        )

        # Create analyzer
        analyzer = BogleBenchAnalyzer()
        analyzer.config = temp_config

        # Set scenario-specific start and end dates
        analyzer.start_date = start_date
        analyzer.end_date = end_date

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

    def test_date_scenarios(self, scenario_analyzer):
        """
        Runs a date range scenario and performs assertions based on the scenario
        type.
        """

        analyzer, scenario_name, transactions_file = scenario_analyzer

        # --- Main Workflow ---
        analyzer.load_transactions(transactions_file)
        portfolio_df = analyzer.build_portfolio_history()
        results = analyzer.calculate_performance()

        # --- Assertions ---
        if scenario_name in [
            "user_dates_outside_transaction_dates",
            "user_start_date_after_end_date",
        ]:
            # Expect no portfolio data since the date range is invalid
            assert portfolio_df.empty
            assert results is not None
            assert results.portfolio_metrics is None
            assert results.benchmark_metrics is None
            return  # No further assertions needed for these scenarios
        else:
            assert not portfolio_df.empty
            assert results is not None
            assert "total_return" in results.portfolio_metrics["mod_dietz"]
            assert "total_return" in results.portfolio_metrics["twr"]
            assert "total_return" in results.benchmark_metrics

        first_day = portfolio_df.iloc[0]
        final_day = portfolio_df.iloc[-1]

        if scenario_name == "user_start_end_equal_transaction_dates":
            assert (
                first_day["date"].date() == pd.to_datetime("2023-01-02").date()
            )
            assert (
                final_day["date"].date() == pd.to_datetime("2023-01-06").date()
            )

            assert first_day["Taxable_TICK_shares"] == 10
            assert final_day["Taxable_TICK_shares"] == 10

            # Initial investment of $1000, plus $1030 buy, minus $1040 sell, plus
            # $50 dividend
            assert first_day["total_value"] == 1000.00
            assert final_day["total_value"] == 1040.00

        elif scenario_name == "no_user_start_end_dates":
            assert (
                first_day["date"].date() == pd.to_datetime("2023-01-02").date()
            )
            assert (
                final_day["date"].date() == pd.to_datetime("2023-01-10").date()
            )

            assert first_day["Taxable_TICK_shares"] == 10
            assert final_day["Taxable_TICK_shares"] == 10

            # Initial investment of $1000, plus $1030 buy, minus $1040 sell, plus
            # $50 dividend
            assert first_day["total_value"] == 1000.00
            assert final_day["total_value"] == 1040.00

        elif scenario_name == "user_start_end_inside_transaction_dates":
            assert (
                first_day["date"].date() == pd.to_datetime("2023-01-03").date()
            )
            assert (
                final_day["date"].date() == pd.to_datetime("2023-01-05").date()
            )

            assert first_day["Taxable_TICK_shares"] == 0
            assert final_day["Taxable_TICK_shares"] == 10

            assert first_day["total_value"] == 0.00
            assert final_day["total_value"] == 1030.00

        elif scenario_name == "user_end_date_only_transaction_dates":
            assert (
                first_day["date"].date() == pd.to_datetime("2023-01-02").date()
            )
            assert (
                final_day["date"].date() == pd.to_datetime("2023-01-06").date()
            )

            assert first_day["Taxable_TICK_shares"] == 10
            assert final_day["Taxable_TICK_shares"] == 10

            # Initial investment of $1000, plus $1030 buy, minus $1040 sell, plus
            # $50 dividend
            assert first_day["total_value"] == 1000.00
            assert final_day["total_value"] == 1040.00

        elif scenario_name == "user_start_end_dates_outside_transaction_dates":
            assert (
                first_day["date"].date() == pd.to_datetime("2022-12-30").date()
            )
            assert (
                final_day["date"].date() == pd.to_datetime("2023-01-09").date()
            )

            assert first_day["Taxable_TICK_shares"] == 0
            assert final_day["Taxable_TICK_shares"] == 10

            assert first_day["total_value"] == 0.00
            assert final_day["total_value"] == 1040.00
