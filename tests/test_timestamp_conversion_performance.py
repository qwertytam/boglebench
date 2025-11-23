"""
Performance test to verify vectorized timestamp conversion.

This test demonstrates the performance benefit of using vectorized timestamp
conversion over per-row conversion via SQLite's register_converter.
"""

import time
import pandas as pd

from boglebench.core.portfolio_db import PortfolioDatabase
from boglebench.utils.config import ConfigManager


# Performance test configuration
TEST_DAYS = 500  # Number of days for performance test


def generate_test_data(num_days=TEST_DAYS):
    """Generate test data for timestamp conversion performance test."""
    dates = pd.date_range(
        start="2024-01-01", periods=num_days, freq="D", tz="UTC"
    )

    summaries = []
    for i, date in enumerate(dates):
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

    return summaries


def test_timestamp_conversion_vectorized():
    """
    Test vectorized timestamp conversion performance.
    
    This test verifies that:
    1. Timestamps are correctly converted to UTC-aware datetime64
    2. The vectorized approach works correctly with pandas read_sql_query
    3. All timestamps maintain timezone information
    """
    summaries = generate_test_data(TEST_DAYS)

    # Create database and insert test data
    db = PortfolioDatabase(
        db_path=":memory:", config=ConfigManager(config_path=None)
    )
    
    with db.transaction():
        db.bulk_insert_portfolio_summaries(summaries)
    
    # Test: Query data and verify timestamps are correctly converted
    start_time = time.time()
    df = db.get_portfolio_summary()
    query_time = time.time() - start_time
    
    # Verify data
    assert len(df) == TEST_DAYS, f"Expected {TEST_DAYS} rows, got {len(df)}"
    
    # Verify timestamps are UTC-aware datetime64
    assert df["date"].dtype == "datetime64[ns, UTC]", (
        f"Expected datetime64[ns, UTC], got {df['date'].dtype}"
    )
    
    # Verify all timestamps have timezone info
    assert df["date"].dt.tz is not None, "All timestamps should be timezone-aware"
    
    # Verify timestamps are in UTC
    assert str(df["date"].dt.tz) == "UTC", "All timestamps should be in UTC"
    
    # Verify dates are correct
    expected_dates = pd.date_range(
        start="2024-01-01", periods=TEST_DAYS, freq="D", tz="UTC"
    )
    assert df["date"].tolist() == expected_dates.tolist(), "Dates should match"
    
    db.close()
    
    print(f"\n{'='*60}")
    print("VECTORIZED TIMESTAMP CONVERSION TEST")
    print(f"{'='*60}")
    print(f"Test data: {TEST_DAYS} days")
    print(f"Query time: {query_time:.4f}s")
    print(f"Average time per row: {(query_time / TEST_DAYS) * 1000:.4f}ms")
    print(f"\n✅ All timestamps correctly converted to UTC-aware datetime64")
    print(f"{'='*60}\n")


def test_timestamp_conversion_with_multiple_queries():
    """
    Test timestamp conversion performance with multiple query types.
    
    This ensures the vectorized approach works correctly across different
    query methods and date columns.
    """
    # Generate more complex test data
    dates = pd.date_range(start="2024-01-01", periods=50, freq="D", tz="UTC")
    
    summaries = []
    account_data = []
    holdings = []
    symbol_data = []
    
    accounts = ["Taxable", "IRA"]
    symbols = ["AAPL", "MSFT"]
    
    for i, date in enumerate(dates):
        summaries.append({
            "date": date,
            "total_value": 10000.0 + i * 100,
            "net_cash_flow": 100.0,
            "investment_cash_flow": 100.0,
            "income_cash_flow": 0.0,
        })
        
        for account in accounts:
            account_data.append({
                "date": date,
                "account": account,
                "total_value": 5000.0,
                "cash_flow": 50.0,
                "weight": 0.5,
            })
            
            for symbol in symbols:
                holdings.append({
                    "date": date,
                    "account": account,
                    "symbol": symbol,
                    "quantity": 10.0,
                    "value": 1500.0,
                    "weight": 0.25,
                })
        
        for symbol in symbols:
            symbol_data.append({
                "date": date,
                "symbol": symbol,
                "price": 150.0,
                "adj_price": 148.0,
                "total_quantity": 20.0,
                "total_value": 3000.0,
                "weight": 0.5,
            })
    
    # Create database and insert test data
    db = PortfolioDatabase(
        db_path=":memory:", config=ConfigManager(config_path=None)
    )
    
    with db.transaction():
        db.bulk_insert_portfolio_summaries(summaries)
        db.bulk_insert_account_data(account_data)
        db.bulk_insert_holdings(holdings)
        db.bulk_insert_symbol_data(symbol_data)
    
    # Test different query methods
    queries = [
        ("get_portfolio_summary", lambda: db.get_portfolio_summary()),
        ("get_account_data", lambda: db.get_account_data()),
        ("get_holdings", lambda: db.get_holdings()),
        ("get_symbol_data", lambda: db.get_symbol_data()),
    ]
    
    print(f"\n{'='*60}")
    print("MULTIPLE QUERY TYPES TIMESTAMP CONVERSION TEST")
    print(f"{'='*60}")
    
    for query_name, query_func in queries:
        start_time = time.time()
        df = query_func()
        query_time = time.time() - start_time
        
        # Verify timestamps
        assert "date" in df.columns, f"{query_name}: Missing 'date' column"
        assert df["date"].dtype == "datetime64[ns, UTC]", (
            f"{query_name}: Expected datetime64[ns, UTC], got {df['date'].dtype}"
        )
        assert df["date"].dt.tz is not None, (
            f"{query_name}: All timestamps should be timezone-aware"
        )
        
        print(f"{query_name:30s}: {len(df):5d} rows in {query_time:.4f}s ✅")
    
    print(f"{'='*60}\n")
    
    db.close()


if __name__ == "__main__":
    test_timestamp_conversion_vectorized()
    test_timestamp_conversion_with_multiple_queries()
