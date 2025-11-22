"""
Test suite for verifying caching optimizations work correctly.
"""

import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from boglebench.core.attribution import AttributionCalculator
from boglebench.core.brinson_attribution import BrinsonAttributionCalculator
from boglebench.core.portfolio import BogleBenchAnalyzer
from boglebench.utils.config import ConfigManager


class TestCachingOptimization:
    """Test caching optimizations in attribution calculators."""

    @pytest.fixture
    def temp_config(self):
        """Create a temporary configuration and directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            (config_dir / "transactions").mkdir()
            (config_dir / "market_data").mkdir()
            (config_dir / "output").mkdir()

            config = ConfigManager()
            config.config["data"]["base_path"] = str(config_dir)
            config.config["data"][
                "attributes_file"
            ] = "transactions/attributes.csv"
            config.config["database"]["db_path"] = ":memory:"

            config.config["analysis"]["start_date"] = "2023-01-03"
            config.config["analysis"]["end_date"] = "2023-01-06"

            config.config["benchmark"] = {
                "name": "Test Benchmark",
                "components": [
                    {"symbol": "VTI", "weight": 1.0},
                ],
            }
            yield config

    @pytest.fixture
    def market_data(self, temp_config):
        """Generate predictable market data."""
        dates = pd.to_datetime(
            ["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"],
            utc=True,
        )
        market_data_dict = {
            "MSFT": pd.DataFrame(
                {
                    "date": dates,
                    "close": [100, 101, 102, 103],
                    "adj_close": [100, 101, 102, 103],
                    "dividend": [0, 0, 0, 0],
                    "split_coefficient": [0, 0, 0, 0],
                }
            ),
            "VTI": pd.DataFrame(
                {
                    "date": dates,
                    "close": [150, 151, 152, 153],
                    "adj_close": [150, 151, 152, 153],
                    "dividend": [0, 0, 0, 0],
                    "split_coefficient": [0, 0, 0, 0],
                }
            ),
        }

        base_path = temp_config.config["data"]["base_path"]
        for symbol, df in market_data_dict.items():
            df.to_parquet(f"{base_path}/market_data/{symbol}.parquet")

        yield market_data_dict

    @pytest.fixture
    def transactions_csv(self, temp_config):
        """Create test transactions CSV."""
        transactions = textwrap.dedent(
            """
            date,symbol,transaction_type,quantity,value_per_share,total_value,account
            2023-01-03,MSFT,BUY,10,100,1000,Account1
            """
        ).strip()

        path = (
            Path(temp_config.config["data"]["base_path"])
            / "transactions"
            / "transactions.csv"
        )
        path.write_text(transactions)
        return path

    @pytest.fixture
    def attributes_csv(self, temp_config):
        """Create test attributes CSV."""
        attributes = textwrap.dedent(
            """
            symbol,asset_class,sector
            MSFT,Equity,Technology
            VTI,Equity,Diversified
            """
        ).strip()

        path = (
            Path(temp_config.config["data"]["base_path"])
            / "transactions"
            / "attributes.csv"
        )
        path.write_text(attributes)
        return path

    def test_attribution_calculator_caching(
        self, temp_config, market_data, transactions_csv, attributes_csv
    ):
        """Test that AttributionCalculator pre-caches data on initialization."""
        # Create analyzer and build portfolio
        analyzer = BogleBenchAnalyzer(config_path=None)
        analyzer.config = temp_config
        analyzer.load_transactions(transactions_csv)
        analyzer.build_portfolio_history()
        analyzer.build_symbol_attributes()

        # Create attribution calculator
        attrib_calc = AttributionCalculator(
            portfolio_db=analyzer.portfolio_db,
        )

        # Verify caches are populated after initialization
        assert (
            attrib_calc._symbol_data_cache is not None
        ), "Symbol data cache should be populated"
        assert (
            attrib_calc._account_data_cache is not None
        ), "Account data cache should be populated"
        assert (
            attrib_calc._attributes_cache is not None
        ), "Attributes cache should be populated"

        # Verify data is correct
        assert not attrib_calc._symbol_data_cache.empty
        assert not attrib_calc._account_data_cache.empty
        assert not attrib_calc._attributes_cache.empty

    def test_brinson_attribution_caching(
        self, temp_config, market_data, transactions_csv, attributes_csv
    ):
        """Test that BrinsonAttributionCalculator pre-caches data on initialization."""
        # Create analyzer and build portfolio
        analyzer = BogleBenchAnalyzer(config_path=None)
        analyzer.config = temp_config
        analyzer.load_transactions(transactions_csv)
        analyzer.build_portfolio_history()
        analyzer.build_symbol_attributes()

        # Create Brinson calculator
        brinson_calc = BrinsonAttributionCalculator(
            benchmark_history=analyzer.benchmark_history,
            portfolio_db=analyzer.portfolio_db,
        )

        # Verify caches are populated after initialization
        assert (
            brinson_calc._symbol_data_cache is not None
        ), "Symbol data cache should be populated"
        assert (
            len(brinson_calc._attributes_cache) > 0
        ), "Attributes cache should be populated"

        # Verify both history and non-history caches are populated
        assert (
            "history_False" in brinson_calc._attributes_cache
        ), "Non-history attributes should be cached"
        assert (
            "history_True" in brinson_calc._attributes_cache
        ), "History attributes should be cached"

        # Verify data is correct
        assert not brinson_calc._symbol_data_cache.empty
        assert not brinson_calc._attributes_cache["history_False"].empty

    def test_caching_improves_performance(
        self, temp_config, market_data, transactions_csv, attributes_csv
    ):
        """Test that caching actually improves performance on repeated calls."""
        import time

        # Create analyzer and build portfolio
        analyzer = BogleBenchAnalyzer(config_path=None)
        analyzer.config = temp_config
        analyzer.load_transactions(transactions_csv)
        analyzer.build_portfolio_history()
        analyzer.build_symbol_attributes()

        # Create attribution calculator
        attrib_calc = AttributionCalculator(
            portfolio_db=analyzer.portfolio_db,
        )

        # First call - cache should be warm already due to pre-caching
        start = time.time()
        result1 = attrib_calc.calculate(group_by="symbol")
        first_call_time = time.time() - start

        # Second call - should use cached data
        start = time.time()
        result2 = attrib_calc.calculate(group_by="symbol")
        second_call_time = time.time() - start

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)

        # Second call should be as fast or faster (both use cache)
        # This mainly tests that we're not regressing performance
        assert (
            second_call_time <= first_call_time * 2
        ), "Cached call should not be significantly slower"

    def test_parallel_factor_attribution_works(
        self, temp_config, market_data, transactions_csv, attributes_csv
    ):
        """Test that parallel factor attribution calculations work correctly."""
        # Enable attribution analysis
        temp_config.config["analysis"]["attribution_analysis"] = {
            "enabled": True,
            "method": "Brinson",
            "attributes_to_analyze": ["asset_class", "sector"],
        }

        # Create analyzer and build portfolio
        analyzer = BogleBenchAnalyzer(config_path=None)
        analyzer.config = temp_config
        analyzer.load_transactions(transactions_csv)
        analyzer.build_portfolio_history()
        analyzer.build_symbol_attributes()

        # Calculate performance (this triggers parallel factor attribution)
        results = analyzer.calculate_performance()

        # Verify factor attributions were calculated
        assert (
            results.factor_attributions is not None
        ), "Factor attributions should be calculated"
        assert (
            len(results.factor_attributions) > 0
        ), "Should have at least one factor attribution"

        # Verify we got the expected factors
        expected_factors = {"asset_class", "sector"}
        actual_factors = set(results.factor_attributions.keys())
        assert expected_factors.issubset(
            actual_factors
        ), f"Expected factors {expected_factors}, got {actual_factors}"

        # Verify each factor attribution has data
        for factor, attribution in results.factor_attributions.items():
            assert (
                not attribution.empty
            ), f"Factor {factor} should have attribution data"
