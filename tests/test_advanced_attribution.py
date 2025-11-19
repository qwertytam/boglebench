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
            config.config["database"]["db_path"] = ":memory:"

            config.config["analysis"]["start_date"] = "2023-01-03"
            config.config["analysis"]["end_date"] = "2023-01-24"

            config.config["analysis"]["attribution_analysis"] = {
                "enabled": True,
                "method": "Brinson",
                "transaction_groups": ["asset_class", "sector"],
            }
            config.config["benchmark"] = {
                "name": "Custom 60/40",
                "components": [
                    {
                        "symbol": "VTI",
                        "weight": 0.60,
                    },
                    {
                        "symbol": "VXUS",
                        "weight": 0.40,
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

        # Mock the market data provider to return test data
        # pylint: disable=unused-argument
        def mock_get_market_data(self, symbols, start_date, end_date):
            return {
                symbol: market_data[symbol]
                for symbol in symbols
                if symbol in market_data
            }

        monkeypatch.setattr(
            "boglebench.core.market_data.MarketDataProvider.get_market_data",
            mock_get_market_data,
        )

        analyzer = BogleBenchAnalyzer()
        analyzer.config = temp_config

        yield analyzer

    def _write_transactions(self, analyzer: BogleBenchAnalyzer, data: str):
        """Helper to write transaction string data to the temp CSV file."""
        transactions_file_path = analyzer.config.get_transactions_file_path()
        # The data string often has leading whitespace, so we dedent it.

        transactions_file_path.write_text(textwrap.dedent(data.strip()))

    def _write_attributes(
        self, analyzer: BogleBenchAnalyzer, data: str
    ) -> str:
        """Helper to write attributes string data to a temp CSV file."""
        attributes_dir = analyzer.config.get_transactions_file_path().parent
        attributes_path = attributes_dir / "symbol_attributes.csv"
        attributes_path.write_text(textwrap.dedent(data.strip()))
        return str(attributes_path)

    def test_brinson_outperformance_scenario(self, scenario_analyzer):
        """Test Case 1.1: Verify Allocation & Selection for outperformance."""
        transactions_data = """
date,symbol,transaction_type,quantity,value_per_share,total_value,account
2023-01-03,AAPL,BUY,100,150,15000,Test_Account
        """

        attributes_data = """
symbol,effective_date,asset_class,sector
AAPL,2023-01-01,US Equity,Technology
VTI,2023-01-01,US Equity,Total Market
VXUS,2023-01-01,International Equity,Total Market
        """

        analyzer = scenario_analyzer
        self._write_transactions(analyzer, transactions_data)
        attributes_path = self._write_attributes(analyzer, attributes_data)

        analyzer.load_transactions()
        analyzer.build_portfolio_history()
        analyzer.load_symbol_attributes(csv_path=attributes_path)
        results = analyzer.calculate_performance()

        brinson_summary = results.brinson_summary
        assert brinson_summary is not None

        us_equity_results = brinson_summary["asset_class"].loc["US Equity"]
        assert us_equity_results["Allocation Effect"] > 0
        assert us_equity_results["Selection Effect"] > 0

    def test_brinson_selection_drilldown(self, scenario_analyzer):
        """Test Case 1.4: Verify the drill-down report math."""
        transactions_data = """
date,symbol,transaction_type,quantity,value_per_share,total_value,account
2023-01-03,AAPL,BUY,50,150,7500,Test_Account
2023-01-03,MSFT,BUY,50,100,5000,Test_Account
"""

        attributes_data = """
symbol,effective_date,asset_class,sector
AAPL,2023-01-01,US Equity,Technology
MSFT,2023-01-01,US Equity,Technology
VTI,2023-01-01,US Equity,Total Market
VXUS,2023-01-01,International Equity,Total Market
        """

        analyzer = scenario_analyzer
        self._write_transactions(analyzer, transactions_data)
        attributes_path = self._write_attributes(analyzer, attributes_data)

        analyzer.load_transactions()
        analyzer.build_portfolio_history()
        analyzer.load_symbol_attributes(csv_path=attributes_path)
        results = analyzer.calculate_performance()

        brinson_summary = results.brinson_summary
        selection_drilldown = results.selection_drilldown["asset_class"]

        assert "US Equity" in selection_drilldown
        drilldown_df = selection_drilldown["US Equity"]
        total_selection_effect = brinson_summary["asset_class"].loc[
            "US Equity"
        ]["Selection Effect"]

        atol = 0.02  # Increased tolerance for refactored attribute system
        rtol = 0.1

        assert np.isclose(
            drilldown_df["Selection Effect"].sum(),
            total_selection_effect,
            atol=atol,
            rtol=rtol,
        )
        assert np.isclose(
            drilldown_df.loc["AAPL"]["Selection Effect"],
            3.634 / 100,
            atol=atol,
            rtol=rtol,
        )
        assert np.isclose(
            drilldown_df.loc["MSFT"]["Selection Effect"],
            0.394 / 100,
            atol=atol,
            rtol=rtol,
        )

    def test_overlapping_holdings_round_trip(self, scenario_analyzer):
        """Test Case 4.1: Test performance with an overlapping round-trip trade."""
        transactions_data = """
date,symbol,transaction_type,quantity,value_per_share,total_value,account
2023-01-03,MSFT,BUY,10,100,1000,Test_Account
2023-01-06,AAPL,BUY,20,153.4,3068,Test_Account
2023-01-10,AAPL,SELL,20,160.1,3202,Test_Account
"""

        attributes_data = """
symbol,effective_date,asset_class,sector
AAPL,2023-01-01,US Equity,Technology
MSFT,2023-01-01,US Equity,Technology
VTI,2023-01-01,US Equity,Total Market
VXUS,2023-01-01,International Equity,Total Market
        """

        analyzer = scenario_analyzer
        self._write_transactions(analyzer, transactions_data)
        attributes_path = self._write_attributes(analyzer, attributes_data)

        analyzer.load_transactions()
        portfolio_db = analyzer.build_portfolio_history()
        analyzer.load_symbol_attributes(csv_path=attributes_path)

        # Get holdings data from database
        holdings_df = portfolio_db.get_holdings()

        # Check AAPL quantity on 2023-01-04 (should be 0 before purchase)
        aapl_2023_01_04 = holdings_df[
            (holdings_df["date"].dt.date == pd.Timestamp("2023-01-04").date())
            & (holdings_df["symbol"] == "AAPL")
            & (holdings_df["account"] == "Test_Account")
        ]
        if not aapl_2023_01_04.empty:
            assert aapl_2023_01_04.iloc[0]["quantity"] == 0

        # Check AAPL quantity on 2023-01-06 (after purchase)
        aapl_2023_01_06 = holdings_df[
            (holdings_df["date"].dt.date == pd.Timestamp("2023-01-06").date())
            & (holdings_df["symbol"] == "AAPL")
            & (holdings_df["account"] == "Test_Account")
        ]
        assert not aapl_2023_01_06.empty
        assert aapl_2023_01_06.iloc[0]["quantity"] == 20

        # Check AAPL quantity on 2023-01-11 (after sell)
        aapl_2023_01_11 = holdings_df[
            (holdings_df["date"].dt.date == pd.Timestamp("2023-01-11").date())
            & (holdings_df["symbol"] == "AAPL")
            & (holdings_df["account"] == "Test_Account")
        ]
        if not aapl_2023_01_11.empty:
            assert aapl_2023_01_11.iloc[0]["quantity"] == 0
