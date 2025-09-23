"""
Test suite for advanced attribution analysis, including Brinson-Fachler
and complex transaction scenarios like short selling.
"""

import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from boglebench.core.portfolio import BogleBenchAnalyzer
from boglebench.utils.config import ConfigManager


class TestAdvancedAttribution:
    """Test suite for Brinson-Fachler attribution and complex scenarios."""

    @pytest.fixture
    def temp_config(self):
        """Create a temporary configuration and directory structure for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            (config_dir / "transactions").mkdir()
            (config_dir / "market_data").mkdir()
            (config_dir / "output").mkdir()

            config = ConfigManager()
            config.config["data"]["base_path"] = str(config_dir)
            config.config["analysis"]["start_date"] = "2023-01-03"
            config.config["analysis"]["end_date"] = "2023-01-24"

            config.config["analysis"]["attribution_analysis"] = {
                "enabled": True,
                "method": "Brinson",
                "transaction_groups": ["group_asset_class", "group_sector"],
            }
            config.config["benchmark"] = {
                "name": "Custom 60/40",
                "components": [
                    {
                        "symbol": "VTI",
                        "weight": 0.60,
                        "group_asset_class": "US Equity",
                        "group_sector": "Total Market",
                    },
                    {
                        "symbol": "VXUS",
                        "weight": 0.40,
                        "group_asset_class": "International Equity",
                        "group_sector": "Total Market",
                    },
                ],
            }
            yield config

    @pytest.fixture
    def market_data(self, temp_config):
        """Generate predictable market data parquet files for various scenarios."""
        dates = pd.to_datetime(
            [
                "2023-01-03",
                "2023-01-04",
                "2023-01-05",
                "2023-01-06",
                "2023-01-09",
                "2023-01-10",
                "2023-01-11",
                "2023-01-12",
                "2023-01-13",
                "2023-01-17",
                "2023-01-18",
                "2023-01-19",
                "2023-01-20",
                "2023-01-23",
                "2023-01-24",
            ],
            utc=True,
        )
        market_data_dict = {
            "MSFT": pd.DataFrame(
                {
                    "date": dates,
                    "close": np.linspace(100, 105, len(dates)),
                    "adj_close": np.linspace(100, 105, len(dates)),
                    "dividend": np.zeros(len(dates)),
                    "split_coefficient": np.zeros(len(dates)),
                }
            ),  # 5% return
            "AAPL": pd.DataFrame(
                {
                    "date": dates,
                    "close": np.linspace(150, 165, len(dates)),
                    "adj_close": np.linspace(150, 165, len(dates)),
                    "dividend": np.zeros(len(dates)),
                    "split_coefficient": np.zeros(len(dates)),
                }
            ),  # 10% return
            "TSLA": pd.DataFrame(
                {
                    "date": dates,
                    "close": np.linspace(200, 180, len(dates)),
                    "adj_close": np.linspace(200, 180, len(dates)),
                    "dividend": np.zeros(len(dates)),
                    "split_coefficient": np.zeros(len(dates)),
                }
            ),  # -10% return
            "VTI": pd.DataFrame(
                {
                    "date": dates,
                    "close": np.linspace(200, 208, len(dates)),
                    "adj_close": np.linspace(200, 208, len(dates)),
                    "dividend": np.zeros(len(dates)),
                    "split_coefficient": np.zeros(len(dates)),
                }
            ),  # 4% return
            "VXUS": pd.DataFrame(
                {
                    "date": dates,
                    "close": np.linspace(50, 50.5, len(dates)),
                    "adj_close": np.linspace(50, 50.5, len(dates)),
                    "dividend": np.zeros(len(dates)),
                    "split_coefficient": np.zeros(len(dates)),
                }
            ),  # 1% return
        }

        market_data_path = (
            Path(temp_config.get("data.base_path")) / "market_data"
        )
        for symbol, df in market_data_dict.items():
            df.to_parquet(market_data_path / f"{symbol}.parquet", index=False)

        market_data_path = (
            Path(temp_config.get("data.base_path")) / "market_data"
        )
        for symbol, df in market_data_dict.items():
            df.to_parquet(market_data_path / f"{symbol}.parquet", index=False)

        return market_data_dict

    @pytest.fixture
    def scenario_analyzer(self, temp_config, market_data, monkeypatch):
        """Fixture to set up BogleBenchAnalyzer pointed at the temp directory."""
        temp_data_path = Path(temp_config.get("data.base_path"))

        # Monkeypatch ConfigManager to use our temp paths consistently
        monkeypatch.setattr(
            ConfigManager,
            "get_data_path",
            lambda self, sub_dir=None: (
                temp_data_path / sub_dir if sub_dir else temp_data_path
            ),
        )
        monkeypatch.setattr(
            ConfigManager,
            "get_transactions_file_path",
            lambda self: temp_data_path / "transactions" / "transactions.csv",
        )
        monkeypatch.setattr(
            ConfigManager,
            "get_market_data_path",
            lambda self: temp_data_path / "market_data",
        )
        monkeypatch.setattr(
            ConfigManager,
            "get_output_path",
            lambda self: temp_data_path / "output",
        )

        analyzer = BogleBenchAnalyzer()
        analyzer.config = temp_config

        yield analyzer

    def _write_transactions(self, analyzer: BogleBenchAnalyzer, data: str):
        """Helper to write transaction string data to the temp CSV file."""
        transactions_file_path = analyzer.config.get_transactions_file_path()
        # The data string often has leading whitespace, so we dedent it.

        transactions_file_path.write_text(textwrap.dedent(data.strip()))

    def test_brinson_outperformance_scenario(self, scenario_analyzer):
        """Test Case 1.1: Verify Allocation & Selection for outperformance."""
        transactions_data = """
date,symbol,transaction_type,quantity,value_per_share,total_value,account,group_asset_class
2023-01-03,AAPL,BUY,100,150,15000,Test_Account,US Equity
        """
        analyzer = scenario_analyzer
        self._write_transactions(analyzer, transactions_data)
        analyzer.load_transactions()  # Load from the file we just wrote
        analyzer.build_portfolio_history()
        results = analyzer.calculate_performance()

        brinson_summary = results.brinson_summary
        assert brinson_summary is not None

        us_equity_results = brinson_summary.loc["US Equity"]
        assert us_equity_results["Allocation Effect"] > 0
        assert us_equity_results["Selection Effect"] > 0

    def test_brinson_selection_drilldown(self, scenario_analyzer):
        """Test Case 1.4: Verify the drill-down report math."""
        transactions_data = """
date,symbol,transaction_type,quantity,value_per_share,total_value,account,group_asset_class
2023-01-03,AAPL,BUY,50,150,7500,Test_Account,US Equity
2023-01-03,MSFT,BUY,50,100,5000,Test_Account,US Equity
"""
        analyzer = scenario_analyzer
        self._write_transactions(analyzer, transactions_data)
        analyzer.load_transactions()
        analyzer.build_portfolio_history()
        results = analyzer.calculate_performance()

        brinson_summary = results.brinson_summary
        selection_drilldown = results.selection_drilldown

        assert "US Equity" in selection_drilldown
        drilldown_df = selection_drilldown["US Equity"]
        total_selection_effect = brinson_summary.loc["US Equity"][
            "Selection Effect"
        ]

        assert np.isclose(
            drilldown_df["Contribution to Selection"].sum(),
            total_selection_effect,
        )
        assert drilldown_df.loc["AAPL"]["Contribution to Selection"] > 0
        assert drilldown_df.loc["MSFT"]["Contribution to Selection"] > 0

    def test_overlapping_holdings_round_trip(self, scenario_analyzer):
        """Test Case 4.1: Test performance with an overlapping round-trip trade."""
        transactions_data = """
date,symbol,transaction_type,quantity,value_per_share,total_value,account,group_asset_class
2023-01-03,MSFT,BUY,10,100,1000,Test_Account,US Equity
2023-01-06,AAPL,BUY,20,153.4,3068,Test_Account,US Equity
2023-01-10,AAPL,SELL,20,160.1,3202,Test_Account,US Equity
"""
        analyzer = scenario_analyzer
        self._write_transactions(analyzer, transactions_data)
        analyzer.load_transactions()
        analyzer.build_portfolio_history()
        history = analyzer.portfolio_history

        assert (
            history.loc[
                history["date"] == "2023-01-04", "Test_Account_AAPL_shares"
            ].iloc[0]
            == 0
        )
        assert (
            history.loc[
                history["date"] == "2023-01-06", "Test_Account_AAPL_shares"
            ].iloc[0]
            == 20
        )
        assert (
            history.loc[
                history["date"] == "2023-01-11", "Test_Account_AAPL_shares"
            ].iloc[0]
            == 0
        )

    def test_short_selling_scenario(self, scenario_analyzer):
        """Test Case 4.2: Test handling of a short-selling transaction."""
        transactions_data = """
date,symbol,transaction_type,quantity,value_per_share,total_value,account,group_asset_class
2023-01-05,TSLA,SELL,20,200,4000,Test_Account,US Equity
2023-01-17,TSLA,BUY,20,180,3600,Test_Account,US Equity
"""
        analyzer = scenario_analyzer
        self._write_transactions(analyzer, transactions_data)
        analyzer.load_transactions()
        analyzer.build_portfolio_history()
        history = analyzer.portfolio_history

        tsla_shares_on_jan_6 = history.loc[
            history["date"] == "2023-01-06", "Test_Account_TSLA_shares"
        ].iloc[0]
        assert (
            tsla_shares_on_jan_6 == -20
        ), "System should track negative share quantities for short positions."

        results = analyzer.calculate_performance()
        total_twr = results.portfolio_metrics["twr"]["total_return"]
        assert (
            total_twr > 0
        ), "Total return should be positive after a profitable short trade."
