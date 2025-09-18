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

# --- Test Data Generators ---


@pytest.fixture
def market_data_fixture():
    """Provides a consistent set of market data for tests."""
    dates = pd.date_range(start="2022-12-30", periods=15, freq="D", tz="UTC")
    return {
        "TICK": pd.DataFrame(
            {
                "date": dates,
                "close": range(100, 115),
                "adj_close": range(100, 115),
                "dividend": [0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "split_coefficient": [0] * 15,
            }
        ),
        "SPY": pd.DataFrame(
            {
                "date": dates,
                "close": range(400, 415),
                "adj_close": range(400, 415),
                "dividend": [0] * 15,
                "split_coefficient": [0] * 15,
            }
        ),
    }


@pytest.fixture
def transactions_fixture():
    """Provides a consistent set of transactions for tests."""
    return pd.DataFrame(
        [
            {
                "date": "2023-01-02",
                "ticker": "TICK",
                "transaction_type": "BUY",
                "quantity": 10,
                "value_per_share": 103.00,
                "total_value": 1030.00,
                "account": "Taxable",
            },
            {
                "date": "2023-01-03",
                "ticker": "TICK",
                "transaction_type": "BUY",
                "quantity": 1,
                "value_per_share": 104.00,
                "total_value": 104.00,
                "account": "Taxable",
            },
            {
                "date": "2023-01-04",
                "ticker": "TICK",
                "transaction_type": "SELL",
                "quantity": 1,
                "value_per_share": 105.00,
                "total_value": 105.00,
                "account": "Taxable",
            },
            {
                "date": "2023-01-04",
                "ticker": "TICK",
                "transaction_type": "DIVIDEND",
                "quantity": 0.0,
                "value_per_share": 0,
                "total_value": 5.00,  # 0.5 * 10 shares
                "account": "Taxable",
            },
            {
                "date": "2023-01-06",
                "ticker": "TICK",
                "transaction_type": "SELL",
                "quantity": 5,
                "value_per_share": 104.00,
                "total_value": 520.00,
                "account": "Taxable",
            },
        ]
    )


# --- Test Scenarios ---

# Each scenario is now just a tuple of (start_date, end_date) for the config
SCENARIOS = {
    "user_start_end_equal_transaction_dates": ("2023-01-02", "2023-01-06"),
    "no_user_start_end_dates": (None, None),
    "user_start_end_inside_transaction_dates": ("2023-01-03", "2023-01-05"),
    "user_end_date_only": (None, "2023-01-06"),
    "user_start_end_outside_transaction_dates": ("2022-12-30", "2023-01-09"),
    "user_dates_outside_transaction_range": ("2024-01-10", "2024-01-15"),
    "user_start_after_end": ("2023-01-10", "2023-01-05"),
}


class TestDateScenarios:
    """A test class for various end-to-end date range scenarios."""

    @pytest.fixture
    def temp_config(self):
        """Create a temporary configuration directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            (config_dir / "transactions").mkdir()
            (config_dir / "market_data").mkdir()
            (config_dir / "output").mkdir()

            config = ConfigManager()
            config.config["data"]["base_path"] = str(config_dir)

            config.config["settings"]["cache_market_data"] = True
            config.config["settings"]["force_refresh_market_data"] = False

            yield config

    @pytest.fixture(params=SCENARIOS.items(), ids=lambda x: x[0])
    def scenario_analyzer(
        self,
        request,
        temp_config,
        transactions_fixture,  # pylint: disable=redefined-outer-name
        market_data_fixture,  # pylint: disable=redefined-outer-name
        monkeypatch,
    ):
        """
        Fixture to set up BogleBenchAnalyzer for a given date range scenario.
        This now works by modifying the config before creating the analyzer.
        """
        scenario_name, (start_date, end_date) = request.param

        # Modify the config for the current scenario
        temp_config.config["analysis"]["start_date"] = start_date
        temp_config.config["analysis"]["end_date"] = end_date

        # Save transactions to a temporary file
        temp_data_path = temp_config.get_data_path()
        transactions_file_path = (
            temp_data_path / "transactions" / "transactions.csv"
        )
        transactions_fixture.to_csv(transactions_file_path, index=False)

        market_data_path = temp_config.get_market_data_path()
        for ticker, df in market_data_fixture.items():
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

        analyzer = BogleBenchAnalyzer()
        analyzer.config = temp_config

        yield analyzer, scenario_name, transactions_file_path

    @patch("boglebench.core.dates.AnalysisPeriod._get_last_closed_market_day")
    @patch("boglebench.core.dates.AnalysisPeriod._is_market_currently_open")
    def test_date_scenarios(
        self,
        mock_is_market_open,
        mock_last_closed_market_day,
        scenario_analyzer,
    ):
        """
        Runs a date range scenario and performs assertions based on the scenario type.
        """
        mock_is_market_open.return_value = True
        mock_last_closed_market_day.return_value = pd.to_datetime(
            "2023-01-10", utc=True
        )

        analyzer, scenario_name, transactions_file = scenario_analyzer

        # --- Handle the error case first ---
        if scenario_name == "user_start_after_end":
            with pytest.raises(ValueError, match="cannot be after end date"):
                analyzer.load_transactions(transactions_file)
                analyzer.build_portfolio_history()
            return  # End of test for this scenario

        # --- Main Workflow ---
        analyzer.load_transactions(transactions_file)
        portfolio_df = analyzer.build_portfolio_history()

        # --- Assertions ---
        if scenario_name == "user_dates_outside_transaction_range":
            # This range has no transactions, so the portfolio should be all
            # zeros
            assert portfolio_df["TICK_total_shares"].eq(0).all()
            assert portfolio_df["TICK_total_value"].eq(0).all()
            assert portfolio_df["total_value"].eq(0).all()
            assert portfolio_df["investment_cash_flow"].eq(0).all()
            assert portfolio_df["income_cash_flow"].eq(0).all()
            return

        # --- Assertions for all other valid scenarios ---
        assert not portfolio_df.empty
        first_day = portfolio_df.iloc[0]
        final_day = portfolio_df.iloc[-1]

        if scenario_name == "user_start_end_equal_transaction_dates":
            assert (
                first_day["date"].date() == pd.to_datetime("2023-01-02").date()
            )
            assert (
                final_day["date"].date() == pd.to_datetime("2023-01-06").date()
            )
            assert final_day["Taxable_TICK_shares"] == 5.0

        elif scenario_name == "no_user_start_end_dates":
            # Falls back to first transaction date and mocked end date
            assert (
                first_day["date"].date() == pd.to_datetime("2023-01-02").date()
            )
            assert (
                final_day["date"].date() == pd.to_datetime("2023-01-10").date()
            )
            assert final_day["Taxable_TICK_shares"] == 5.0

        elif scenario_name == "user_start_end_inside_transaction_dates":
            assert (
                first_day["date"].date() == pd.to_datetime("2023-01-03").date()
            )
            assert (
                final_day["date"].date() == pd.to_datetime("2023-01-05").date()
            )
            # The initial buy is before the start date
            assert first_day["Taxable_TICK_shares"] == 1.0
            assert final_day["Taxable_TICK_shares"] == 0.0

        elif scenario_name == "user_end_date_only":
            assert (
                first_day["date"].date() == pd.to_datetime("2023-01-02").date()
            )
            assert (
                final_day["date"].date() == pd.to_datetime("2023-01-06").date()
            )
            assert final_day["Taxable_TICK_shares"] == 5.0

        elif scenario_name == "user_start_end_outside_transaction_dates":
            assert (
                first_day["date"].date() == pd.to_datetime("2022-12-30").date()
            )
            assert (
                final_day["date"].date() == pd.to_datetime("2023-01-09").date()
            )
            assert first_day["Taxable_TICK_shares"] == 0
            assert final_day["Taxable_TICK_shares"] == 5.0
