"""
Integration tests for the composite benchmark feature.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from boglebench.core.portfolio import BogleBenchAnalyzer
from boglebench.utils.config import ConfigManager


@pytest.fixture
def mock_market_data():
    """Provides mock market data for two symbols, VTI and VXUS."""
    dates = pd.to_datetime(
        pd.date_range("2023-01-01", "2023-03-30", freq="D"), utc=True
    )
    vti_close = (
        pd.concat(
            [
                pd.Series(range(100, 130)),  # Jan: VTI rises
                pd.Series(range(130, 101, -1)),  # Feb: VTI falls
                pd.Series(range(102, 132)),  # Mar: VTI rises again
            ]
        ).reset_index(drop=True),
    )
    vxus_close = (
        pd.concat(
            [
                pd.Series(range(80, 50, -1)),  # Jan: VXUS falls
                pd.Series(range(51, 80)),  # Feb: VXUS rises
                pd.Series(range(80, 50, -1)),  # Mar: VXUS falls again
            ]
        ).reset_index(drop=True),
    )
    zeros = pd.Series([0.0] * len(dates))

    return {
        "VTI": pd.DataFrame(
            {
                "date": dates,
                "close": vti_close[0],
                "adj_close": vti_close[0],
                "dividend": zeros,
                "split_coefficient": zeros,
            }
        ),
        "VXUS": pd.DataFrame(
            {
                "date": dates,
                "close": vxus_close[0],
                "adj_close": vxus_close[0],
                "dividend": zeros,
                "split_coefficient": zeros,
            }
        ),
    }


@pytest.fixture
def transactions_fixture():
    """Provides a minimal set of transactions."""
    return pd.DataFrame(
        [
            {
                "date": "2023-01-02",
                "symbol": "VTI",
                "transaction_type": "BUY",
                "quantity": 10,
                "value_per_share": 100.00,
                "total_value": 1000.00,
                "account": "Taxable",
            }
        ]
    )


@pytest.fixture
def temp_config_file():
    """Creates a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False, suffix=".yml"
    ) as f:
        config_path = f.name
        f.write(
            """
analysis:
  start_date: "2023-01-01"
  end_date: "2023-03-30"
data:
  base_path: "" # Will be set by the test
database:
  db_path: ":memory:"
benchmark:
  rebalancing: monthly
  components:
    - { symbol: VTI, weight: 0.6 }
    - { symbol: VXUS, weight: 0.4 }
settings:
  cache_market_data: true
  force_refresh_market_data: false
"""
        )
    yield Path(config_path)
    Path(config_path).unlink()


@patch("boglebench.utils.config.ConfigManager.get_market_data_path")
# pylint: disable=redefined-outer-name
def test_composite_benchmark_calculation(
    mock_market_data_path,
    temp_config_file,
    transactions_fixture,
    mock_market_data,
):
    """
    Tests the end-to-end composite benchmark calculation.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # --- Setup ---
        config = ConfigManager(temp_config_file)
        config.config["data"]["base_path"] = temp_dir

        mock_market_data_path.return_value = Path(temp_dir) / "market_data"

        # Save mock data to files
        temp_data_path = config.get_data_path()
        transactions_path = temp_data_path / "input"
        transactions_path.mkdir(parents=True, exist_ok=True)
        transactions_file_path = transactions_path / "transactions.csv"
        transactions_fixture.to_csv(transactions_file_path, index=False)

        market_data_path = config.get_market_data_path()
        market_data_path.mkdir(exist_ok=True)
        for symbol, df in mock_market_data.items():
            df.to_parquet(market_data_path / f"{symbol}.parquet", index=False)

        # --- Execution ---
        analyzer = BogleBenchAnalyzer(config_path=temp_config_file)
        analyzer.load_transactions(transactions_file_path)
        analyzer.build_portfolio_history()
        results = analyzer.calculate_performance()

        benchmark_history = analyzer.benchmark_history

        # --- Assertions ---
        assert (
            not benchmark_history.empty
        ), "Benchmark history should be generated"

        # Check that the benchmark value reflects the component movements
        # Start of Feb, VTI is high, VXUS is low.
        # Start of Mar, VTI is low, VXUS is high.
        jan_31_val = benchmark_history.at["2023-01-31", "adj_close"]
        feb_28_val = benchmark_history.at["2023-02-28", "adj_close"]

        # In Jan, VTI (60%) went up, VXUS (40%) went down. Net should be up slightly.
        # VTI: 100->129 (+29%), VXUS: 80->51 (-36.25%). (0.6*1.29) + (0.4*0.6375) > 1
        assert jan_31_val == 10350

        # In Feb, VTI (60%) went down, VXUS (40%) went up.
        # This tests that rebalancing happened on Feb 1st.
        # VTI: 129->102 (-21%), VXUS: 51->79 (+55%). (0.6*0.79) + (0.4*1.55) > 1
        assert feb_28_val > jan_31_val

        # Final check on the results object
        benchmark_metrics = results.benchmark_metrics
        assert benchmark_metrics is not None
        assert "Benchmark" == benchmark_metrics["name"]
        assert (
            benchmark_metrics["total_return"] > 0.1
        )  # Overall return should be >10%
        assert (
            benchmark_metrics["volatility"] > 0.01
        )  # Some volatility should be present
