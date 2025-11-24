# pylint: disable=protected-access

"""
Performance test for vectorized price lookup optimization.

This test measures the performance improvement from using pre-built numpy arrays
and binary search (searchsorted) instead of repeated DataFrame filtering.
"""

import time

import pandas as pd
import pytest

from boglebench.core.constants import TransactionTypes
from boglebench.core.history_builder import PortfolioHistoryBuilder
from boglebench.utils.config import ConfigManager


def generate_large_market_data(num_days=1000, num_symbols=10):
    """Generate market data for performance testing."""
    dates = pd.date_range("2020-01-01", periods=num_days, freq="D", tz="UTC")

    market_data = {}
    for i in range(num_symbols):
        symbol = f"STOCK{i}"
        # Create realistic price data with some variation
        base_price = 100 + i * 10
        # pylint: disable=no-member
        prices = base_price + (i + 1) * (dates.day % 10)

        market_data[symbol] = pd.DataFrame(
            {
                "date": dates,
                "close": prices,
                "adj_close": prices
                * 0.98,  # Slightly different adjusted prices
            }
        )

    return market_data


def generate_test_transactions(num_symbols=10):
    """Generate sample transactions for testing."""
    symbols = [f"STOCK{i}" for i in range(num_symbols)]

    data = {
        "date": pd.to_datetime(
            [
                "2020-01-15",
                "2020-02-01",
                "2020-03-15",
                "2020-06-01",
                "2020-09-01",
                "2021-01-15",
                "2021-06-01",
                "2021-12-01",
            ]
            * num_symbols,
            utc=True,
        ),
        "account": ["Brokerage"] * (8 * num_symbols),
        "symbol": symbols * 8,
        "transaction_type": [TransactionTypes.BUY.value] * (8 * num_symbols),
        "quantity": [10.0] * (8 * num_symbols),
        "price": [100.0] * (8 * num_symbols),
        "total_value": [-1000.0] * (8 * num_symbols),
        "fees": [0.0] * (8 * num_symbols),
    }
    return pd.DataFrame(data)


@pytest.fixture
def config():
    """Create test config."""
    config_dict = {
        "settings": {
            "annual_trading_days": 252,
            "annual_risk_free_rate": 0.03,
        }
    }
    config = ConfigManager()  # pylint: disable=redefined-outer-name
    config.config = config_dict
    return config


# pylint: disable=redefined-outer-name
def test_price_lookup_performance(config):
    """
    Test performance of vectorized price lookup vs original method.

    This test demonstrates the performance improvement from using pre-built
    numpy arrays with binary search (O(log n)) instead of repeated DataFrame
    filtering (O(n)).

    Expected improvement: 50-100x faster per lookup, resulting in 70-75%
    reduction in portfolio building time.
    """
    # Generate test data
    num_days = 717  # Same as profiling scenario
    num_symbols = 10
    market_data = generate_large_market_data(
        num_days=num_days, num_symbols=num_symbols
    )
    transactions = generate_test_transactions(num_symbols=num_symbols)

    # Create builder (this builds the price lookup structure)
    start_date = pd.Timestamp("2020-01-01", tz="UTC")
    end_date = pd.Timestamp("2021-12-31", tz="UTC")

    builder = PortfolioHistoryBuilder(
        config=config,
        transactions=transactions,
        market_data=market_data,
        start_date=start_date,
        end_date=end_date,
    )

    # Test dates to lookup
    test_dates = pd.date_range("2020-01-01", "2021-12-31", freq="7D", tz="UTC")

    # Measure time for old method (_get_price_for_date)
    old_start = time.time()
    old_results = []
    for date in test_dates:
        for symbol in [f"STOCK{i}" for i in range(num_symbols)]:
            price = builder._get_price_for_date(symbol, date, adjusted=False)
            old_results.append(price)
    old_time = time.time() - old_start

    # Measure time for new method (_get_price_fast)
    new_start = time.time()
    new_results = []
    for date in test_dates:
        for symbol in [f"STOCK{i}" for i in range(num_symbols)]:
            price = builder._get_price_fast(symbol, date, adjusted=False)
            new_results.append(price)
    new_time = time.time() - new_start

    # Calculate improvement
    speedup = old_time / new_time if new_time > 0 else 0
    improvement_pct = (
        ((old_time - new_time) / old_time) * 100 if old_time > 0 else 0
    )

    # Number of lookups performed
    num_lookups = len(test_dates) * num_symbols

    print(f"\n{'='*70}")
    print("PRICE LOOKUP PERFORMANCE COMPARISON")
    print(f"{'='*70}")
    print("Test configuration:")
    print(f"  - Days of market data: {num_days}")
    print(f"  - Symbols: {num_symbols}")
    print(f"  - Test lookups: {num_lookups:,}")
    print("\nOriginal method (_get_price_for_date):")
    print(f"  - Total time: {old_time:.4f}s")
    print(f"  - Per lookup: {(old_time/num_lookups)*1000:.3f}ms")
    print("\nOptimized method (_get_price_fast):")
    print(f"  - Total time: {new_time:.4f}s")
    print(f"  - Per lookup: {(new_time/num_lookups)*1000:.3f}ms")
    print("\nPerformance improvement:")
    print(f"  - Speedup: {speedup:.1f}x faster")
    print(f"  - Improvement: {improvement_pct:.1f}%")
    print("\nExpected impact on portfolio building (717 days):")
    print("  - Original: ~10.4s (with 17,925 lookups)")
    print(f"  - Optimized: ~{10.4/speedup:.1f}s (estimated)")
    print(f"  - Time saved: ~{10.4 - (10.4/speedup):.1f}s")
    print(f"{'='*70}\n")

    # Verify results match
    assert len(old_results) == len(new_results), "Result count mismatch"

    # Allow for small floating point differences
    for i, (old_val, new_val) in enumerate(zip(old_results, new_results)):
        assert (
            abs(old_val - new_val) < 0.01
        ), f"Value mismatch at index {i}: {old_val} vs {new_val}"

    # Verify we got a significant speedup
    assert speedup > 10, f"Expected at least 10x speedup, got {speedup:.1f}x"
    assert new_time < old_time, "New method should be faster"


def test_price_lookup_accuracy(config):
    """
    Verify that the optimized price lookup produces identical results to the original.

    Tests various scenarios:
    - Exact date matches
    - Forward fill (date between market dates)
    - Backward fill (date before first market date)
    - Missing symbol handling
    """
    market_data = {
        "AAPL": pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2020-01-01", "2020-01-03", "2020-01-07"], utc=True
                ),
                "close": [100.0, 102.0, 105.0],
                "adj_close": [98.0, 100.0, 103.0],
            }
        )
    }

    transactions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"], utc=True),
            "account": ["Brokerage"],
            "symbol": ["AAPL"],
            "transaction_type": [TransactionTypes.BUY.value],
            "quantity": [10.0],
            "price": [100.0],
            "total_value": [-1000.0],
            "fees": [0.0],
        }
    )

    builder = PortfolioHistoryBuilder(
        config=config,
        transactions=transactions,
        market_data=market_data,
        start_date=pd.Timestamp("2020-01-01", tz="UTC"),
        end_date=pd.Timestamp("2020-01-10", tz="UTC"),
    )

    # Test exact match
    date1 = pd.Timestamp("2020-01-03", tz="UTC")
    old_price1 = builder._get_price_for_date("AAPL", date1, adjusted=False)
    new_price1 = builder._get_price_fast("AAPL", date1, adjusted=False)
    assert (
        abs(old_price1 - new_price1) < 0.01
    ), f"Exact match failed: {old_price1} vs {new_price1}"
    assert abs(old_price1 - 102.0) < 0.01, "Expected price 102.0"

    # Test forward fill
    date2 = pd.Timestamp("2020-01-05", tz="UTC")
    old_price2 = builder._get_price_for_date("AAPL", date2, adjusted=False)
    new_price2 = builder._get_price_fast("AAPL", date2, adjusted=False)
    assert (
        abs(old_price2 - new_price2) < 0.01
    ), f"Forward fill failed: {old_price2} vs {new_price2}"
    assert (
        abs(old_price2 - 102.0) < 0.01
    ), "Expected forward filled price 102.0"

    # Test backward fill (date before first data point)
    date3 = pd.Timestamp("2019-12-31", tz="UTC")
    old_price3 = builder._get_price_for_date("AAPL", date3, adjusted=False)
    new_price3 = builder._get_price_fast("AAPL", date3, adjusted=False)
    assert (
        abs(old_price3 - new_price3) < 0.01
    ), f"Backward fill failed: {old_price3} vs {new_price3}"
    assert (
        abs(old_price3 - 100.0) < 0.01
    ), "Expected backward filled price 100.0"

    # Test adjusted prices
    adj_price1 = builder._get_price_for_date("AAPL", date1, adjusted=True)
    adj_price2 = builder._get_price_fast("AAPL", date1, adjusted=True)
    assert (
        abs(adj_price1 - adj_price2) < 0.01
    ), f"Adjusted price failed: {adj_price1} vs {adj_price2}"
    assert abs(adj_price1 - 100.0) < 0.01, "Expected adjusted price 100.0"

    # Test missing symbol
    missing_old = builder._get_price_for_date("MSFT", date1, adjusted=False)
    missing_new = builder._get_price_fast("MSFT", date1, adjusted=False)
    assert missing_old == 0.0, "Expected 0.0 for missing symbol (old)"
    assert missing_new == 0.0, "Expected 0.0 for missing symbol (new)"

    print(
        "\n✅ All accuracy tests passed - optimized method produces identical results"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
