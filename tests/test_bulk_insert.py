"""
Test bulk insert operations for portfolio database.

Validates that bulk insert methods work correctly and produce identical
results to the original row-by-row insertion approach.
"""

import pandas as pd
import pytest

from boglebench.core.portfolio_db import PortfolioDatabase
from boglebench.utils.config import ConfigManager


@pytest.fixture
def test_config():
    """Create a test config."""
    return ConfigManager(config_path=None)


@pytest.fixture
def in_memory_db(test_config):  # pylint: disable=redefined-outer-name
    """Create an in-memory database for testing."""
    return PortfolioDatabase(db_path=":memory:", config=test_config)


# pylint: disable=redefined-outer-name
def test_bulk_insert_portfolio_summaries(in_memory_db):
    """Test bulk insert of portfolio summaries."""
    summaries = [
        {
            "date": pd.Timestamp("2024-01-01", tz="UTC"),
            "total_value": 10000.0,
            "net_cash_flow": 1000.0,
            "investment_cash_flow": 1000.0,
            "income_cash_flow": 0.0,
            "portfolio_mod_dietz_return": 0.05,
            "portfolio_twr_return": 0.05,
            "market_value_change": 500.0,
        },
        {
            "date": pd.Timestamp("2024-01-02", tz="UTC"),
            "total_value": 10500.0,
            "net_cash_flow": 0.0,
            "investment_cash_flow": 0.0,
            "income_cash_flow": 0.0,
            "portfolio_mod_dietz_return": 0.02,
            "portfolio_twr_return": 0.02,
            "market_value_change": 200.0,
        },
    ]

    with in_memory_db.transaction():
        in_memory_db.bulk_insert_portfolio_summaries(summaries)

    # Verify data was inserted
    result = in_memory_db.get_portfolio_summary()
    assert len(result) == 2
    assert result["total_value"].tolist() == [10000.0, 10500.0]


def test_bulk_insert_account_data(in_memory_db):
    """Test bulk insert of account data."""
    # Need portfolio summary first due to foreign key constraint
    summaries = [
        {
            "date": pd.Timestamp("2024-01-01", tz="UTC"),
            "total_value": 10000.0,
            "net_cash_flow": 1000.0,
            "investment_cash_flow": 1000.0,
            "income_cash_flow": 0.0,
            "portfolio_mod_dietz_return": 0.05,
            "portfolio_twr_return": 0.05,
            "market_value_change": 500.0,
        }
    ]

    account_data = [
        {
            "date": pd.Timestamp("2024-01-01", tz="UTC"),
            "account": "Taxable",
            "total_value": 5000.0,
            "cash_flow": 500.0,
            "weight": 0.5,
            "mod_dietz_return": 0.04,
            "twr_return": 0.04,
        },
        {
            "date": pd.Timestamp("2024-01-01", tz="UTC"),
            "account": "IRA",
            "total_value": 5000.0,
            "cash_flow": 500.0,
            "weight": 0.5,
            "mod_dietz_return": 0.06,
            "twr_return": 0.06,
        },
    ]

    with in_memory_db.transaction():
        in_memory_db.bulk_insert_portfolio_summaries(summaries)
        in_memory_db.bulk_insert_account_data(account_data)

    # Verify data was inserted
    cursor = in_memory_db.get_cursor()
    cursor.execute("SELECT COUNT(*) FROM account_data")
    count = cursor.fetchone()[0]
    assert count == 2


def test_bulk_insert_holdings(in_memory_db):
    """Test bulk insert of holdings."""
    # Need portfolio summary and account data first due to foreign key constraints
    summaries = [
        {
            "date": pd.Timestamp("2024-01-01", tz="UTC"),
            "total_value": 10000.0,
            "net_cash_flow": 1000.0,
            "investment_cash_flow": 1000.0,
            "income_cash_flow": 0.0,
            "portfolio_mod_dietz_return": 0.05,
            "portfolio_twr_return": 0.05,
            "market_value_change": 500.0,
        }
    ]

    account_data = [
        {
            "date": pd.Timestamp("2024-01-01", tz="UTC"),
            "account": "Taxable",
            "total_value": 5000.0,
            "cash_flow": 500.0,
            "weight": 0.5,
            "mod_dietz_return": 0.04,
            "twr_return": 0.04,
        },
        {
            "date": pd.Timestamp("2024-01-01", tz="UTC"),
            "account": "IRA",
            "total_value": 5000.0,
            "cash_flow": 500.0,
            "weight": 0.5,
            "mod_dietz_return": 0.06,
            "twr_return": 0.06,
        },
    ]

    holdings = [
        {
            "date": pd.Timestamp("2024-01-01", tz="UTC"),
            "account": "Taxable",
            "symbol": "AAPL",
            "quantity": 10.0,
            "value": 1500.0,
            "weight": 0.3,
        },
        {
            "date": pd.Timestamp("2024-01-01", tz="UTC"),
            "account": "IRA",
            "symbol": "MSFT",
            "quantity": 5.0,
            "value": 2000.0,
            "weight": 0.4,
        },
    ]

    with in_memory_db.transaction():
        in_memory_db.bulk_insert_portfolio_summaries(summaries)
        in_memory_db.bulk_insert_account_data(account_data)
        in_memory_db.bulk_insert_holdings(holdings)

    # Verify data was inserted
    cursor = in_memory_db.get_cursor()
    cursor.execute("SELECT COUNT(*) FROM holdings")
    count = cursor.fetchone()[0]
    assert count == 2


def test_bulk_insert_symbol_data(in_memory_db):
    """Test bulk insert of symbol data."""
    # Need portfolio summary first due to foreign key constraint
    summaries = [
        {
            "date": pd.Timestamp("2024-01-01", tz="UTC"),
            "total_value": 10000.0,
            "net_cash_flow": 1000.0,
            "investment_cash_flow": 1000.0,
            "income_cash_flow": 0.0,
            "portfolio_mod_dietz_return": 0.05,
            "portfolio_twr_return": 0.05,
            "market_value_change": 500.0,
        }
    ]

    symbol_data = [
        {
            "date": pd.Timestamp("2024-01-01", tz="UTC"),
            "symbol": "AAPL",
            "price": 150.0,
            "adj_price": 148.5,
            "total_quantity": 10.0,
            "total_value": 1500.0,
            "weight": 0.3,
            "cash_flow": 0.0,
            "market_return": 0.02,
            "twr_return": 0.02,
        },
        {
            "date": pd.Timestamp("2024-01-01", tz="UTC"),
            "symbol": "MSFT",
            "price": 400.0,
            "adj_price": 398.0,
            "total_quantity": 5.0,
            "total_value": 2000.0,
            "weight": 0.4,
            "cash_flow": 100.0,
            "market_return": 0.03,
            "twr_return": 0.03,
        },
    ]

    with in_memory_db.transaction():
        in_memory_db.bulk_insert_portfolio_summaries(summaries)
        in_memory_db.bulk_insert_symbol_data(symbol_data)

    # Verify data was inserted
    cursor = in_memory_db.get_cursor()
    cursor.execute("SELECT COUNT(*) FROM symbol_data")
    count = cursor.fetchone()[0]
    assert count == 2


def test_bulk_insert_empty_lists(in_memory_db):
    """Test that bulk inserts handle empty lists gracefully."""
    # Should not raise any errors
    with in_memory_db.transaction():
        in_memory_db.bulk_insert_portfolio_summaries([])
        in_memory_db.bulk_insert_account_data([])
        in_memory_db.bulk_insert_holdings([])
        in_memory_db.bulk_insert_symbol_data([])

    # Verify no data was inserted
    cursor = in_memory_db.get_cursor()
    cursor.execute("SELECT COUNT(*) FROM portfolio_summary")
    assert cursor.fetchone()[0] == 0


# pylint: disable=unused-argument
def test_bulk_insert_vs_single_insert_equivalence(in_memory_db):
    """
    Test that bulk insert produces identical results to single inserts.

    This ensures backward compatibility and correctness.
    """
    # Create test data
    date = pd.Timestamp("2024-01-01", tz="UTC")

    # Single insert approach
    db1 = PortfolioDatabase(
        db_path=":memory:", config=ConfigManager(config_path=None)
    )
    with db1.transaction():
        db1.insert_portfolio_summary(
            date=date,
            total_value=10000.0,
            net_cash_flow=1000.0,
            investment_cash_flow=1000.0,
            income_cash_flow=0.0,
            portfolio_mod_dietz_return=0.05,
            portfolio_twr_return=0.05,
            market_value_change=500.0,
        )
        db1.insert_account_data(
            date=date,
            account="Taxable",
            total_value=5000.0,
            cash_flow=500.0,
            weight=0.5,
            mod_dietz_return=0.04,
            twr_return=0.04,
        )

    # Bulk insert approach
    db2 = PortfolioDatabase(
        db_path=":memory:", config=ConfigManager(config_path=None)
    )
    with db2.transaction():
        db2.bulk_insert_portfolio_summaries(
            [
                {
                    "date": date,
                    "total_value": 10000.0,
                    "net_cash_flow": 1000.0,
                    "investment_cash_flow": 1000.0,
                    "income_cash_flow": 0.0,
                    "portfolio_mod_dietz_return": 0.05,
                    "portfolio_twr_return": 0.05,
                    "market_value_change": 500.0,
                }
            ]
        )
        db2.bulk_insert_account_data(
            [
                {
                    "date": date,
                    "account": "Taxable",
                    "total_value": 5000.0,
                    "cash_flow": 500.0,
                    "weight": 0.5,
                    "mod_dietz_return": 0.04,
                    "twr_return": 0.04,
                }
            ]
        )

    # Compare results
    result1 = db1.get_portfolio_summary()
    result2 = db2.get_portfolio_summary()

    assert len(result1) == len(result2)
    assert result1["total_value"].iloc[0] == result2["total_value"].iloc[0]
    assert result1["net_cash_flow"].iloc[0] == result2["net_cash_flow"].iloc[0]

    db1.close()
    db2.close()
