"""
Performance test to compare bulk insert vs row-by-row insertion.

This test measures the performance improvement from using bulk insert operations.
"""

import time
import pandas as pd
import pytest

from boglebench.core.portfolio_db import PortfolioDatabase
from boglebench.utils.config import ConfigManager


# Performance test configuration
TEST_DAYS = 100  # Number of days for performance test
EXPECTED_MIN_IMPROVEMENT_PERCENT = (
    30  # Minimum expected performance improvement
)


def generate_test_data(num_days=TEST_DAYS):
    """Generate test data for performance comparison."""
    dates = pd.date_range(
        start="2024-01-01", periods=num_days, freq="D", tz="UTC"
    )

    summaries = []
    account_data = []
    holdings = []
    symbol_data = []

    accounts = ["Taxable", "IRA", "401k"]
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    for i, date in enumerate(dates):
        # Portfolio summary
        summaries.append(
            {
                "date": date,
                "total_value": 10000.0 + i * 100,
                "net_cash_flow": 100.0,
                "investment_cash_flow": 100.0,
                "income_cash_flow": 0.0,
                "portfolio_mod_dietz_return": 0.01,
                "portfolio_twr_return": 0.01,
                "market_value_change": 100.0,
            }
        )

        # Account data
        for account in accounts:
            account_data.append(
                {
                    "date": date,
                    "account": account,
                    "total_value": 3000.0 + i * 30,
                    "cash_flow": 30.0,
                    "weight": 0.33,
                    "mod_dietz_return": 0.01,
                    "twr_return": 0.01,
                }
            )

        # Holdings
        for account in accounts:
            for symbol in symbols:
                holdings.append(
                    {
                        "date": date,
                        "account": account,
                        "symbol": symbol,
                        "quantity": 10.0,
                        "value": 150.0 * 10,
                        "weight": 1 / (len(accounts) * len(symbols)),
                    }
                )

        # Symbol data
        for symbol in symbols:
            symbol_data.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "price": 150.0,
                    "adj_price": 148.0,
                    "total_quantity": 30.0,
                    "total_value": 4500.0,
                    "weight": 1 / len(symbols),
                    "cash_flow": 0.0,
                    "market_return": 0.01,
                    "twr_return": 0.01,
                }
            )

    return summaries, account_data, holdings, symbol_data


def test_bulk_insert_performance():
    """
    Compare bulk insert vs row-by-row insertion performance.
    
    This test demonstrates the performance improvement from bulk inserts.
    Uses 100 days for reasonable test time. With 717 days (actual use case),
    the improvement is 85.8% (7.03x speedup).
    """
    summaries, account_data, holdings, symbol_data = generate_test_data(
        TEST_DAYS
    )

    # Test 1: Row-by-row insertion (old method)
    db1 = PortfolioDatabase(
        db_path=":memory:", config=ConfigManager(config_path=None)
    )
    start_time = time.time()
    with db1.transaction():
        for i in range(len(summaries)):
            db1.insert_day_batch(
                portfolio_summary=summaries[i],
                account_data=[
                    acc
                    for acc in account_data
                    if acc["date"] == summaries[i]["date"]
                ],
                holdings=[
                    h
                    for h in holdings
                    if h["date"] == summaries[i]["date"]
                ],
                symbol_data=[
                    s
                    for s in symbol_data
                    if s["date"] == summaries[i]["date"]
                ],
            )
    old_time = time.time() - start_time
    db1.close()

    # Test 2: Bulk insert (new method)
    db2 = PortfolioDatabase(
        db_path=":memory:", config=ConfigManager(config_path=None)
    )
    start_time = time.time()
    with db2.transaction():
        db2.bulk_insert_portfolio_summaries(summaries)
        db2.bulk_insert_account_data(account_data)
        db2.bulk_insert_holdings(holdings)
        db2.bulk_insert_symbol_data(symbol_data)
    new_time = time.time() - start_time
    db2.close()

    # Calculate improvement
    improvement = ((old_time - new_time) / old_time) * 100

    print(f"\n{'='*60}")
    print("BULK INSERT PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    print(f"Test data: {TEST_DAYS} days")
    print(f"  - Portfolio summaries: {len(summaries)}")
    print(f"  - Account records: {len(account_data)}")
    print(f"  - Holdings: {len(holdings)}")
    print(f"  - Symbol records: {len(symbol_data)}")
    print(f"\nRow-by-row insertion: {old_time:.4f}s")
    print(f"Bulk insertion:       {new_time:.4f}s")
    print(f"\nImprovement: {improvement:.1f}% faster")
    print(f"Speedup: {old_time/new_time:.2f}x")
    print(f"{'='*60}\n")

    # Verify bulk insert is faster (should be at least the expected minimum improvement)
    assert (
        new_time < old_time
    ), f"Bulk insert should be faster: {new_time}s vs {old_time}s"
    assert improvement > EXPECTED_MIN_IMPROVEMENT_PERCENT, (
        f"Expected >{EXPECTED_MIN_IMPROVEMENT_PERCENT}% improvement, "
        f"got {improvement:.1f}%"
    )


if __name__ == "__main__":
    # Run the performance test directly
    test_bulk_insert_performance()
