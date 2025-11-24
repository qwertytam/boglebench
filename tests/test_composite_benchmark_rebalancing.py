"""
Tests for composite benchmark rebalancing logic, especially handling non-market days.

This module tests that rebalancing occurs on the next market day when the intended
rebalancing date falls on a weekend or holiday.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from boglebench.core.composite_benchmark import CompositeBenchmarkBuilder
from boglebench.core.portfolio import BogleBenchAnalyzer
from boglebench.utils.config import ConfigManager


@pytest.fixture
def mock_market_data_with_gaps():
    """
    Provides mock market data with specific gaps to test rebalancing.

    Creates a scenario where:
    - Jan 1, 2023 is a Sunday (non-market day, should rebalance on Jan 3)
    - Feb 1, 2023 is a Wednesday (market day, should rebalance on Feb 1)
    - Mar 1, 2023 is a Wednesday (market day, should rebalance on Mar 1)
    - Apr 1, 2023 is a Saturday (non-market day, should rebalance on Apr 3)
    """
    # Create dates excluding weekends
    dates = pd.date_range(
        "2023-01-03", "2023-04-30", freq="B"
    )  # Business days only
    dates = pd.to_datetime(dates, utc=True)

    n = len(dates)
    zeros = [0.0] * n

    # VTI starts at 100, gradually increases
    vti_prices = [100.0 + i * 0.5 for i in range(n)]

    # VXUS starts at 50, gradually increases
    vxus_prices = [50.0 + i * 0.25 for i in range(n)]

    # BND starts at 30, gradually decreases
    bnd_prices = [30.0 - i * 0.1 for i in range(n)]

    return {
        "VTI": pd.DataFrame(
            {
                "date": dates,
                "close": vti_prices,
                "adj_close": vti_prices,
                "dividend": zeros,
                "split_coefficient": zeros,
            }
        ),
        "VXUS": pd.DataFrame(
            {
                "date": dates,
                "close": vxus_prices,
                "adj_close": vxus_prices,
                "dividend": zeros,
                "split_coefficient": zeros,
            }
        ),
        "BND": pd.DataFrame(
            {
                "date": dates,
                "close": bnd_prices,
                "adj_close": bnd_prices,
                "dividend": zeros,
                "split_coefficient": zeros,
            }
        ),
    }


@pytest.fixture
def temp_config_monthly():
    """Creates a config file with monthly rebalancing."""
    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False, suffix=".yml"
    ) as f:
        config_path = f.name
        f.write(
            """
analysis:
  start_date: "2023-01-03"
  end_date: "2023-04-30"
data:
  base_path: ""
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


@pytest.fixture
def temp_config_weekly():
    """Creates a config file with weekly rebalancing."""
    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False, suffix=".yml"
    ) as f:
        config_path = f.name
        f.write(
            """
analysis:
  start_date: "2023-01-03"
  end_date: "2023-01-31"
data:
  base_path: ""
database:
  db_path: ":memory:"
benchmark:
  rebalancing: weekly
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


def test_rebalancing_schedule_creation():
    """
    Test that _create_rebalancing_schedule correctly identifies rebalancing days.
    """

    # Mock market data with specific dates
    market_days = pd.DatetimeIndex(
        [
            "2022-12-30",  # Friday before Jan 1
            "2023-01-03",  # Tuesday (Jan 1 was Sunday, Jan 2 market closed)
            "2023-01-04",
            "2023-01-05",
            "2023-02-01",  # Wednesday
            "2023-02-02",
            "2023-03-01",  # Wednesday
            "2023-03-02",
            "2023-04-03",  # Monday (Apr 1 was Saturday)
            "2023-04-04",
        ],
        tz="UTC",
    )

    mock_market_data = {
        "VTI": pd.DataFrame(
            {
                "date": market_days,
                "adj_close": [100.0] * len(market_days),
            }
        ),
        "VXUS": pd.DataFrame(
            {
                "date": market_days,
                "adj_close": [50.0] * len(market_days),
            }
        ),
    }

    # Create builder with mock config
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yml") as f:
        f.write(
            """
benchmark:
  rebalancing: monthly
  components:
    - { symbol: VTI, weight: 0.6 }
    - { symbol: VXUS, weight: 0.4 }
"""
        )
        f.flush()
        config = ConfigManager(str(Path(f.name)))

        builder = CompositeBenchmarkBuilder(
            config=config,
            market_data=mock_market_data,
            start_date=market_days[0],
            end_date=market_days[-1],
        )

        # Test the rebalancing schedule
        schedule = builder.get_rebalancing_schedule(market_days)

        # Expected: Jan 3 (next day after Jan 1), Feb 1, Mar 1, Apr 3 (next day after Apr 1)
        expected_rebalance_dates = pd.to_datetime(
            [
                "2023-01-03",  # First market day after Jan 1 (Sunday)
                "2023-02-01",  # Feb 1 is a market day
                "2023-03-01",  # Mar 1 is a market day
                "2023-04-03",  # First market day after Apr 1 (Saturday)
            ],
            utc=True,
        )

        assert (
            schedule.sum() == 4
        ), f"Expected 4 rebalancing days, got {schedule.sum()}"

        for expected_date in expected_rebalance_dates:
            assert schedule.loc[
                expected_date
            ], f"Expected rebalancing on {expected_date}"


# pylint: disable=redefined-outer-name
def test_monthly_rebalancing_on_non_market_day(
    temp_config_monthly, mock_market_data_with_gaps
):
    """
    Test that monthly rebalancing occurs on the next market day when the 1st falls on a weekend.
    """
    config = ConfigManager(temp_config_monthly)

    builder = CompositeBenchmarkBuilder(
        config=config,
        market_data=mock_market_data_with_gaps,
        start_date=pd.Timestamp("2023-01-03", tz="UTC"),
        end_date=pd.Timestamp("2023-04-30", tz="UTC"),
    )

    benchmark_df = builder.build()

    assert not benchmark_df.empty, "Benchmark should be generated"

    # Check that weights changed on expected dates
    # Jan 3 (first market day after Jan 1 Sunday)
    jan_3_vti_weight = benchmark_df.loc["2023-01-03", "VTI_weight"]
    jan_4_vti_weight = benchmark_df.loc["2023-01-04", "VTI_weight"]

    if not isinstance(jan_3_vti_weight, float):
        jan_3_vti_weight = 0.0
    if not isinstance(jan_4_vti_weight, float):
        jan_4_vti_weight = 0.0

    # After rebalancing on Jan 3, VTI weight should be ~0.6
    assert (
        abs(jan_3_vti_weight - 0.6) < 0.01
    ), f"Expected VTI weight ~0.6 on Jan 3 (rebalance day), got {jan_3_vti_weight}"

    # Feb 1 is a Wednesday (market day), should rebalance
    feb_1_vti_weight = benchmark_df.loc["2023-02-01", "VTI_weight"]
    if not isinstance(feb_1_vti_weight, float):
        feb_1_vti_weight = 0.0
    assert (
        abs(feb_1_vti_weight - 0.6) < 0.01
    ), f"Expected VTI weight ~0.6 on Feb 1 (rebalance day), got {feb_1_vti_weight}"

    # Apr 3 (first market day after Apr 1 Saturday)
    apr_3_vti_weight = benchmark_df.loc["2023-04-03", "VTI_weight"]
    if not isinstance(apr_3_vti_weight, float):
        apr_3_vti_weight = 0.0
    assert (
        abs(apr_3_vti_weight - 0.6) < 0.01
    ), f"Expected VTI weight ~0.6 on Apr 3 (rebalance day), got {apr_3_vti_weight}"


# pylint: disable=unused-argument
def test_weekly_rebalancing_on_non_market_monday(
    temp_config_weekly, mock_market_data_with_gaps
):
    """
    Test that weekly rebalancing shifts to Tuesday when Monday is a holiday.
    """
    # Create market data that excludes Monday Jan 16 (MLK Day)
    dates = pd.bdate_range("2023-01-03", "2023-01-31", freq="B")
    # Remove Jan 16 (Monday holiday)
    dates = dates[dates != pd.Timestamp("2023-01-16")]
    dates = pd.to_datetime(dates, utc=True)

    n = len(dates)
    market_data = {
        "VTI": pd.DataFrame(
            {
                "date": dates,
                "close": [100.0] * n,
                "adj_close": [100.0] * n,
                "dividend": [0.0] * n,
                "split_coefficient": [0.0] * n,
            }
        ),
        "VXUS": pd.DataFrame(
            {
                "date": dates,
                "close": [50.0] * n,
                "adj_close": [50.0] * n,
                "dividend": [0.0] * n,
                "split_coefficient": [0.0] * n,
            }
        ),
    }

    config = ConfigManager(temp_config_weekly)

    builder = CompositeBenchmarkBuilder(
        config=config,
        market_data=market_data,
        start_date=dates[0],
        end_date=dates[-1],
    )

    schedule = builder.get_rebalancing_schedule(dates)

    # Should rebalance on:
    # - Jan 3 (Tuesday, since Jan 2 Monday was a market day but before our start)
    # - Jan 9 (Monday)
    # - Jan 17 (Tuesday, since Jan 16 Monday was a holiday)
    # - Jan 23 (Monday)
    # - Jan 30 (Monday)

    jan_16 = pd.Timestamp("2023-01-16", tz="UTC")
    try:
        _ = schedule.loc[jan_16]
    except KeyError:
        assert True, "Jan 16 is not a market day, as expected"
    else:
        assert False, "Jan 16 should not be a rebalancing day"

    jan_17 = pd.Timestamp("2023-01-17", tz="UTC")
    assert schedule.loc[
        jan_17
    ], "Expected rebalancing on Jan 17 (Tuesday after MLK Monday holiday)"


def test_no_duplicate_rebalancing():
    """
    Test that if multiple intended rebalancing dates map to the same market day,
    we only rebalance once.
    """
    # Create scenario where both quarterly and monthly would map to same day
    market_days = pd.DatetimeIndex(
        [
            "2022-12-30",  # Friday before Jan 1
            "2023-01-03",  # Tuesday (both Jan 1 monthly and Q1 quarterly map here)
            "2023-01-04",
            "2023-02-01",
        ],
        tz="UTC",
    )

    mock_market_data = {
        "VTI": pd.DataFrame(
            {
                "date": market_days,
                "adj_close": [100.0, 101.0, 102.0, 103.0],
            }
        ),
        "VXUS": pd.DataFrame(
            {
                "date": market_days,
                "adj_close": [50.0, 50.5, 51.0, 51.5],
            }
        ),
    }

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yml") as f:
        f.write(
            """
benchmark:
  rebalancing: quarterly
  components:
    - { symbol: VTI, weight: 0.6 }
    - { symbol: VXUS, weight: 0.4 }
"""
        )
        f.flush()
        config = ConfigManager(str(Path(f.name)))

        builder = CompositeBenchmarkBuilder(
            config=config,
            market_data=mock_market_data,
            start_date=market_days[0],
            end_date=market_days[-1],
        )

        schedule = builder.get_rebalancing_schedule(market_days)

        # Should only rebalance once on Jan 3
        assert schedule.loc["2023-01-03"]
        assert (
            schedule.sum() == 1
        ), f"Expected exactly 1 rebalancing day, got {schedule.sum()}"


@patch("boglebench.utils.config.ConfigManager.get_market_data_path")
def test_end_to_end_with_non_market_rebalancing(
    temp_config_monthly,
    mock_market_data_with_gaps,
):
    """
    End-to-end test ensuring the full analyzer works with non-market day rebalancing.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        config = ConfigManager(temp_config_monthly)
        config.config["data"]["base_path"] = temp_dir

        # Create minimal transactions
        transactions = pd.DataFrame(
            [
                {
                    "date": "2023-01-03",
                    "symbol": "VTI",
                    "transaction_type": "BUY",
                    "quantity": 10,
                    "value_per_share": 100.0,
                    "total_value": 1000.0,
                    "account": "Taxable",
                }
            ]
        )

        # Save transactions
        temp_data_path = config.get_data_path()
        transactions_path = temp_data_path / "input"
        transactions_path.mkdir(parents=True, exist_ok=True)
        transactions_file = transactions_path / "transactions.csv"
        transactions.to_csv(transactions_file, index=False)

        # Mock the MarketDataProvider to return our test data
        with patch(
            "boglebench.core.portfolio.MarketDataProvider"
        ) as mock_provider:
            mock_provider_instance = mock_provider.return_value
            mock_provider_instance.get_market_data.return_value = (
                mock_market_data_with_gaps
            )

            # Run analysis
            analyzer = BogleBenchAnalyzer(config_path=temp_config_monthly)
            analyzer.load_transactions(transactions_file)
            analyzer.build_portfolio_history()
            _ = analyzer.calculate_performance()

            benchmark_history = analyzer.benchmark_history

            assert (
                not benchmark_history.empty
            ), "Benchmark history should be generated"

            # Verify rebalancing occurred on next market days
            assert (
                pd.Timestamp("2023-01-03", tz="UTC") in benchmark_history.index
            )
            assert (
                pd.Timestamp("2023-02-01", tz="UTC") in benchmark_history.index
            )
            assert (
                pd.Timestamp("2023-04-03", tz="UTC") in benchmark_history.index
            )

            # Check that weights are correct on rebalancing days
            jan_3_vti = benchmark_history.loc[
                pd.Timestamp("2023-01-03", tz="UTC"), "VTI_weight"
            ]
            if not isinstance(jan_3_vti, float):
                jan_3_vti = 0.0
            assert (
                abs(jan_3_vti - 0.2) < 0.05
            ), f"Expected VTI weight near 0.2 on Jan 3, got {jan_3_vti}"
