"""
Base database operations for PortfolioDatabase.
Handles connection management, transactions, and basic utilities.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

from ..utils.config import ConfigManager
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class DatabaseOperations:
    """Base class for database connection and transaction management."""

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
        if db_path is None:
            if config is not None:
                data_dir = config.get_data_path()
                db_path = Path(data_dir) / "portfolio_history.db"
            else:
                db_path = ":memory:"
                logger.warning(
                    "No db_path or config provided, using in-memory database"
                )

        self.db_path = str(db_path)
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self):
        """Establish database connection with optimizations."""
        self.conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            check_same_thread=False,
        )

        # Enable foreign keys
        self.conn.execute("PRAGMA foreign_keys = ON")

        # Performance optimizations
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.execute("PRAGMA cache_size = -64000")  # 64MB cache
        self.conn.execute("PRAGMA temp_store = MEMORY")

        logger.debug("Connected to database: %s", self.db_path)

    def get_cursor(self) -> sqlite3.Cursor:
        """
        Get a database cursor, ensuring connection is established.

        Returns:
            sqlite3.Cursor: Database cursor

        Raises:
            RuntimeError: If database connection is not established
        """
        if self.conn is None:
            raise RuntimeError("Database connection is not established")
        return self.conn.cursor()

    def get_connection(self) -> sqlite3.Connection:
        """
        Get database connection, ensuring it's established.

        Returns:
            sqlite3.Connection: Database connection

        Raises:
            RuntimeError: If database connection is not established
        """
        if self.conn is None:
            raise RuntimeError("Database connection is not established")
        return self.conn

    def commit(self):
        """Commit current transaction."""
        conn = self.get_connection()
        conn.commit()

    @contextmanager
    def transaction(self):
        """Context manager for transactions."""
        conn = self.get_connection()
        try:
            conn.execute("BEGIN TRANSACTION")
            yield
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error("Transaction rolled back: %s", e)
            raise

    def rollback(self):
        """Rollback current transaction."""
        conn = self.get_connection()
        conn.rollback()

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.debug("Database connection closed")

    def optimize(self):
        """Run VACUUM and ANALYZE to optimize database."""
        logger.info("Optimizing database...")
        conn = self.get_connection()
        conn.execute("VACUUM")
        conn.execute("ANALYZE")
        conn.commit()
        logger.info("✅ Database optimized")

    @staticmethod
    def normalize_params(params):
        """Ensure params is an acceptable type for pandas.read_sql_query (tuple or mapping)."""
        if isinstance(params, list):
            return tuple(params)
        return params
