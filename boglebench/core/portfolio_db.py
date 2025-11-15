"""
Normalized portfolio database manager.

This module provides the main database interface for storing and querying
portfolio history using a normalized SQLite schema. Combines query, insert,
allocation, and symbol attribute operations through mixin classes. Serves as
the single source of truth for portfolio data.
"""

from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from ..utils.config import ConfigManager
from ..utils.logging_config import get_logger
from .db_operations import DatabaseOperations
from .db_schema import ALL_TABLES, SCHEMA_VERSION
from .portfolio_allocations import PortfolioAllocationMixin
from .portfolio_db_inserts import PortfolioInsertMixin
from .portfolio_db_queries import PortfolioQueryMixin
from .symbol_attributes import SymbolAttributesMixin

logger = get_logger(__name__)


class PortfolioDatabase(
    DatabaseOperations,
    PortfolioInsertMixin,
    PortfolioQueryMixin,
    SymbolAttributesMixin,
    PortfolioAllocationMixin,
):
    """
    Manages SQLite database for normalized portfolio history with temporal attributes.

    This is the single source of truth for portfolio data.
    All queries return pandas DataFrames in long (normalized) format.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        config: Optional[ConfigManager] = None,
    ):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database. If None, uses config or in-memory.
            config: ConfigManager instance for retrieving default paths.
        """
        super().__init__(db_path, config)
        self.connect()
        self._initialize_schema()
        logger.info("📊 Portfolio database ready: %s", self.db_path)

    def _initialize_schema(self):
        """Create tables if they don't exist."""
        cursor = self.get_cursor()

        for table_sql in ALL_TABLES:
            cursor.executescript(table_sql)

        # Store schema version
        cursor.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )

        # Store creation timestamp
        cursor.execute(
            "INSERT OR IGNORE INTO metadata (key, value) VALUES (?, datetime('now'))",
            ("created_at",),
        )

        conn = self.get_connection()
        conn.commit()
        logger.debug("Database schema initialized")

    def clear_all(self):
        """Delete all portfolio data (keep schema)."""
        cursor = self.get_cursor()
        cursor.execute("DELETE FROM portfolio_summary")
        conn = self.get_connection()
        conn.commit()
        logger.warning("⚠️ All portfolio data cleared")

    def clear_date_range(
        self, start_date: pd.Timestamp, end_date: pd.Timestamp
    ):
        """Delete all data for a date range (for rebuilding)."""
        cursor = self.get_cursor()
        cursor.execute(
            "DELETE FROM portfolio_summary WHERE date BETWEEN ? AND ?",
            (start_date, end_date),
        )
        conn = self.get_connection()
        conn.commit()
        logger.info(
            "Cleared data from %s to %s", start_date.date(), end_date.date()
        )

    # ========== STATISTICS ==========

    def get_stats(self) -> Dict:
        """Get database statistics."""
        cursor = self.get_cursor()

        stats = {}

        # Row counts
        for table in [
            "portfolio_summary",
            "account_data",
            "holdings",
            "symbol_data",
            "symbol_attributes",
        ]:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats[f"{table}_rows"] = cursor.fetchone()[0]

        # Date range
        start_date, end_date = self.get_date_range()
        stats["start_date"] = start_date
        stats["end_date"] = end_date
        stats["num_days"] = (
            (end_date - start_date).days + 1 if start_date and end_date else 0
        )

        # Accounts and symbols
        stats["num_accounts"] = len(self.get_accounts())
        stats["num_symbols"] = len(self.get_symbols())

        # Symbol attributes stats
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM symbol_attributes")
        stats["num_symbols_with_attributes"] = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM symbol_attributes WHERE end_date IS NULL"
        )
        stats["num_current_attributes"] = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM symbol_attributes WHERE end_date IS NOT NULL"
        )
        stats["num_historical_attributes"] = cursor.fetchone()[0]

        # Database file size
        if self.db_path != ":memory:":
            db_file = Path(self.db_path)
            if db_file.exists():
                stats["db_size_mb"] = db_file.stat().st_size / (1024 * 1024)

        return stats

    def print_stats(self):
        """Print database statistics."""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("📊 PORTFOLIO DATABASE STATISTICS")
        print("=" * 60)

        if stats.get("start_date") and stats.get("end_date"):
            print("\n📅 Date Range:")
            print(f"   Start: {stats['start_date'].date()}")
            print(f"   End:   {stats['end_date'].date()}")
            print(f"   Days:  {stats['num_days']}")

        print("\n🏦 Portfolio Structure:")
        print(f"   Accounts: {stats['num_accounts']}")
        print(f"   Symbols:  {stats['num_symbols']}")

        print("\n📋 Symbol Attributes:")
        print(
            f"   Symbols with attributes: {stats['num_symbols_with_attributes']}"
        )
        print(f"   Current versions:        {stats['num_current_attributes']}")
        print(
            f"   Historical versions:     {stats['num_historical_attributes']}"
        )

        print("\n📊 Table Row Counts:")
        print(f"   Portfolio Summary:  {stats['portfolio_summary_rows']:,}")
        print(f"   Account Data:       {stats['account_data_rows']:,}")
        print(f"   Holdings:           {stats['holdings_rows']:,}")
        print(f"   Symbol Data:        {stats['symbol_data_rows']:,}")
        print(f"   Symbol Attributes:  {stats['symbol_attributes_rows']:,}")

        if "db_size_mb" in stats:
            print(f"\n💾 Database Size: {stats['db_size_mb']:.2f} MB")

        print("=" * 60 + "\n")

    # ========== CONTEXT MANAGER ==========

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        if exc_type:
            self.rollback()
        self.close()

    def __repr__(self):
        """String representation."""
        db_path = getattr(self, "db_path", "<not initialized>")
        return f"PortfolioDatabase(db_path='{db_path}')"
