"""Test SQLite compatibility of database schema."""

import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from boglebench.core.db_schema import ALL_TABLES


def test_schema_creates_successfully():
    """Test that all tables and views can be created in SQLite."""
    # Create in-memory database
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # Create all tables and views
    for sql in ALL_TABLES:
        cursor.executescript(sql)

    # Verify all views were created
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='view' ORDER BY name;"
    )
    views = [row[0] for row in cursor.fetchall()]

    expected_views = [
        "current_symbol_attributes",
        "holdings_with_attributes",
        "latest_holdings",
        "latest_portfolio",
        "symbol_data_with_attributes",
    ]

    assert set(views) == set(
        expected_views
    ), f"Expected views {expected_views}, got {views}"

    conn.close()


def test_holdings_with_attributes_temporal_lookup():
    """Test that holdings_with_attributes view correctly performs temporal attribute lookups."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # Create all tables and views
    for sql in ALL_TABLES:
        cursor.executescript(sql)

    # Insert test data with temporal attributes
    cursor.executescript(
        """
        -- Holdings at different dates
        INSERT INTO holdings (date, account, symbol, quantity, value, weight)
        VALUES 
            ('2023-01-15', 'Account1', 'AAPL', 100, 15000, 0.5),
            ('2023-03-15', 'Account1', 'AAPL', 100, 16000, 0.5),
            ('2023-06-15', 'Account1', 'AAPL', 100, 18000, 0.5);

        -- Symbol data
        INSERT INTO symbol_data (date, symbol, price, adj_price, total_quantity, total_value, weight)
        VALUES 
            ('2023-01-15', 'AAPL', 150, 150, 100, 15000, 1.0),
            ('2023-03-15', 'AAPL', 160, 160, 100, 16000, 1.0),
            ('2023-06-15', 'AAPL', 180, 180, 100, 18000, 1.0);

        -- Symbol attributes with temporal changes
        -- Period 1: 2023-01-01 to 2023-02-28 (sector: Technology)
        INSERT INTO symbol_attributes (symbol, effective_date, end_date, asset_class, geography, sector)
        VALUES ('AAPL', '2023-01-01', '2023-02-28', 'Equity', 'US', 'Technology');

        -- Period 2: 2023-03-01 to 2023-05-31 (sector: Consumer Electronics)
        INSERT INTO symbol_attributes (symbol, effective_date, end_date, asset_class, geography, sector)
        VALUES ('AAPL', '2023-03-01', '2023-05-31', 'Equity', 'US', 'Consumer Electronics');

        -- Period 3: 2023-06-01 onwards (sector: Information Technology)
        INSERT INTO symbol_attributes (symbol, effective_date, end_date, asset_class, geography, sector)
        VALUES ('AAPL', '2023-06-01', NULL, 'Equity', 'US', 'Information Technology');
    """
    )

    # Query the view
    cursor.execute(
        """
        SELECT date, symbol, sector 
        FROM holdings_with_attributes 
        ORDER BY date;
    """
    )
    results = cursor.fetchall()

    # Verify temporal lookup worked correctly
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"

    # Check each date has the correct sector
    assert results[0] == (
        "2023-01-15",
        "AAPL",
        "Technology",
    ), "Wrong sector for 2023-01-15"
    assert results[1] == (
        "2023-03-15",
        "AAPL",
        "Consumer Electronics",
    ), "Wrong sector for 2023-03-15"
    assert results[2] == (
        "2023-06-15",
        "AAPL",
        "Information Technology",
    ), "Wrong sector for 2023-06-15"

    conn.close()


def test_symbol_data_with_attributes_temporal_lookup():
    """Test that symbol_data_with_attributes view correctly performs temporal attribute lookups."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # Create all tables and views
    for sql in ALL_TABLES:
        cursor.executescript(sql)

    # Insert test data
    cursor.executescript(
        """
        -- Symbol data
        INSERT INTO symbol_data (date, symbol, price, adj_price, total_quantity, total_value, weight)
        VALUES 
            ('2023-01-15', 'SPY', 400, 400, 100, 40000, 1.0),
            ('2023-06-15', 'SPY', 440, 440, 100, 44000, 1.0);

        -- Symbol attributes with temporal changes
        INSERT INTO symbol_attributes (symbol, effective_date, end_date, asset_class, fund_type)
        VALUES ('SPY', '2023-01-01', '2023-05-31', 'Equity', 'ETF');

        INSERT INTO symbol_attributes (symbol, effective_date, end_date, asset_class, fund_type)
        VALUES ('SPY', '2023-06-01', NULL, 'Mixed', 'Index Fund');
    """
    )

    # Query the view
    cursor.execute(
        """
        SELECT date, symbol, asset_class, fund_type 
        FROM symbol_data_with_attributes 
        ORDER BY date;
    """
    )
    results = cursor.fetchall()

    # Verify temporal lookup worked correctly
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"

    assert results[0] == (
        "2023-01-15",
        "SPY",
        "Equity",
        "ETF",
    ), "Wrong attributes for 2023-01-15"
    assert results[1] == (
        "2023-06-15",
        "SPY",
        "Mixed",
        "Index Fund",
    ), "Wrong attributes for 2023-06-15"

    conn.close()


def test_holdings_with_attributes_null_handling():
    """Test that holdings_with_attributes view handles NULL attributes correctly."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # Create all tables and views
    for sql in ALL_TABLES:
        cursor.executescript(sql)

    # Insert holdings without corresponding attributes
    cursor.execute(
        """
        INSERT INTO holdings (date, account, symbol, quantity, value, weight)
        VALUES ('2023-01-15', 'Account1', 'UNKNOWN', 100, 10000, 0.5);
    """
    )

    # Query the view
    cursor.execute(
        """
        SELECT date, symbol, asset_class, geography, sector 
        FROM holdings_with_attributes 
        WHERE symbol = 'UNKNOWN';
    """
    )
    results = cursor.fetchall()

    # Verify the row exists with NULL attributes
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    assert results[0][0] == "2023-01-15", "Wrong date"
    assert results[0][1] == "UNKNOWN", "Wrong symbol"
    assert results[0][2] is None, "asset_class should be NULL"
    assert results[0][3] is None, "geography should be NULL"
    assert results[0][4] is None, "sector should be NULL"

    conn.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
