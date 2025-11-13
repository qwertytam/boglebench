"""
Database manager for normalized portfolio history with temporal symbol attributes.
Single source of truth - no backward compatibility with wide DataFrames.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from ..utils.config import ConfigManager
from ..utils.logging_config import get_logger
from .db_schema import ALL_TABLES, SCHEMA_VERSION

logger = get_logger(__name__)


class PortfolioDatabase:
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
        if db_path is None:
            if config is not None:
                data_dir = config.get_data_directory()
                db_path = Path(data_dir) / "portfolio_history.db"
            else:
                db_path = ":memory:"
                logger.warning(
                    "No db_path or config provided, using in-memory database"
                )

        self.db_path = str(db_path)
        self.conn: Optional[sqlite3.Connection] = None
        self._connect()
        self._initialize_schema()
        logger.info("📊 Portfolio database ready: %s", self.db_path)

    def _connect(self):
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

    def _get_cursor(self) -> sqlite3.Cursor:
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

    def _get_connection(self) -> sqlite3.Connection:
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
        conn = self._get_connection()
        conn.commit()

    def _normalize_params(self, params):
        """Ensure params is an acceptable type for pandas.read_sql_query (tuple or mapping)."""
        if isinstance(params, list):
            return tuple(params)
        return params

    def _initialize_schema(self):
        """Create tables if they don't exist."""
        cursor = self._get_cursor()

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

        conn = self._get_connection()
        conn.commit()
        logger.debug("Database schema initialized")

    @contextmanager
    def transaction(self):
        """Context manager for transactions."""
        conn = self._get_connection()
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
        conn = self._get_connection()
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
        conn = self._get_connection()
        conn.execute("VACUUM")
        conn.execute("ANALYZE")
        conn.commit()
        logger.info("✅ Database optimized")

    def clear_all(self):
        """Delete all portfolio data (keep schema)."""
        cursor = self._get_cursor()
        cursor.execute("DELETE FROM portfolio_summary")
        conn = self._get_connection()
        conn.commit()
        logger.warning("⚠️ All portfolio data cleared")

    def clear_date_range(
        self, start_date: pd.Timestamp, end_date: pd.Timestamp
    ):
        """Delete all data for a date range (for rebuilding)."""
        cursor = self._get_cursor()
        cursor.execute(
            "DELETE FROM portfolio_summary WHERE date BETWEEN ? AND ?",
            (start_date, end_date),
        )
        conn = self._get_connection()
        conn.commit()
        logger.info(
            "Cleared data from %s to %s", start_date.date(), end_date.date()
        )

    # ========== INSERT METHODS ==========

    def insert_portfolio_summary(
        self,
        date: pd.Timestamp,
        total_value: float,
        net_cash_flow: float = 0,
        investment_cash_flow: float = 0,
        income_cash_flow: float = 0,
        portfolio_mod_dietz_return: Optional[float] = None,
        portfolio_twr_return: Optional[float] = None,
        market_value_change: Optional[float] = None,
    ):
        """Insert portfolio summary for a date."""
        cursor = self._get_cursor()
        cursor.execute(
            """
            INSERT INTO portfolio_summary (
                date, total_value, net_cash_flow, investment_cash_flow,
                income_cash_flow, portfolio_mod_dietz_return, 
                portfolio_twr_return, market_value_change
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                date,
                total_value,
                net_cash_flow,
                investment_cash_flow,
                income_cash_flow,
                portfolio_mod_dietz_return,
                portfolio_twr_return,
                market_value_change,
            ),
        )

    def insert_account_data(
        self,
        date: pd.Timestamp,
        account: str,
        total_value: float,
        cash_flow: float = 0,
        weight: float = 0,
        mod_dietz_return: Optional[float] = None,
        twr_return: Optional[float] = None,
    ):
        """Insert account data for a date."""
        cursor = self._get_cursor()
        cursor.execute(
            """
            INSERT INTO account_data (
                date, account, total_value, cash_flow, weight,
                mod_dietz_return, twr_return
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                date,
                account,
                total_value,
                cash_flow,
                weight,
                mod_dietz_return,
                twr_return,
            ),
        )

    def insert_holding(
        self,
        date: pd.Timestamp,
        account: str,
        symbol: str,
        quantity: float,
        value: float,
        weight: float = 0,
    ):
        """Insert holding data for a date."""
        cursor = self._get_cursor()
        cursor.execute(
            """
            INSERT INTO holdings (
                date, account, symbol, quantity, value, weight
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (date, account, symbol, quantity, value, weight),
        )

    def insert_symbol_data(
        self,
        date: pd.Timestamp,
        symbol: str,
        price: Optional[float],
        adj_price: Optional[float],
        total_quantity: float,
        total_value: float,
        weight: float = 0,
        cash_flow: float = 0,
        market_return: Optional[float] = None,
        twr_return: Optional[float] = None,
    ):
        """Insert symbol data for a date."""
        cursor = self._get_cursor()
        cursor.execute(
            """
            INSERT INTO symbol_data (
                date, symbol, price, adj_price, total_quantity, total_value,
                weight, cash_flow, market_return, twr_return
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                date,
                symbol,
                price,
                adj_price,
                total_quantity,
                total_value,
                weight,
                cash_flow,
                market_return,
                twr_return,
            ),
        )

    def insert_day_batch(
        self,
        portfolio_summary: Dict,
        account_data: List[Dict],
        holdings: List[Dict],
        symbol_data: List[Dict],
    ):
        """
        Insert all data for a single day in a batch.

        Args:
            portfolio_summary: Dict with portfolio-level data
            account_data: List of dicts with account-level data
            holdings: List of dicts with holding-level data
            symbol_data: List of dicts with symbol-level data
        """
        self.insert_portfolio_summary(**portfolio_summary)

        for acc_data in account_data:
            self.insert_account_data(**acc_data)

        for holding in holdings:
            self.insert_holding(**holding)

        for sym_data in symbol_data:
            self.insert_symbol_data(**sym_data)

    def bulk_insert_days(
        self,
        days_data: List[Dict],
    ):
        """
        Bulk insert multiple days of data efficiently.

        Args:
            days_data: List of dicts, each containing:
                - portfolio_summary: Dict
                - account_data: List[Dict]
                - holdings: List[Dict]
                - symbol_data: List[Dict]
        """
        with self.transaction():
            for day in days_data:
                self.insert_day_batch(
                    portfolio_summary=day["portfolio_summary"],
                    account_data=day["account_data"],
                    holdings=day["holdings"],
                    symbol_data=day["symbol_data"],
                )

    # ========== TEMPORAL SYMBOL ATTRIBUTES METHODS ==========

    def insert_symbol_attributes(
        self,
        symbol: str,
        effective_date: pd.Timestamp,
        asset_class: Optional[str] = None,
        geography: Optional[str] = None,
        region: Optional[str] = None,
        sector: Optional[str] = None,
        style: Optional[str] = None,
        market_cap: Optional[str] = None,
        fund_type: Optional[str] = None,
        expense_ratio: Optional[float] = None,
        dividend_yield: Optional[float] = None,
        is_esg: bool = False,
        description: Optional[str] = None,
        source: str = "user",
        end_date: Optional[pd.Timestamp] = None,
    ):
        """
        Insert symbol attributes with temporal tracking.

        Args:
            symbol: Stock/ETF ticker
            effective_date: Date from which these attributes are effective
            asset_class: 'Equity', 'Bond', 'Real Estate', etc.
            geography: 'US', 'International', 'Emerging Markets', etc.
            region: 'North America', 'Europe', 'Asia Pacific', etc.
            sector: 'Technology', 'Healthcare', 'Financials', etc.
            style: 'Growth', 'Value', 'Blend'
            market_cap: 'Large', 'Mid', 'Small', 'Blend'
            fund_type: 'ETF', 'Mutual Fund', 'Individual Stock'
            expense_ratio: Annual expense ratio (e.g., 0.0003 for 0.03%)
            dividend_yield: Current dividend yield (e.g., 0.015 for 1.5%)
            is_esg: ESG compliant
            description: Free-form description
            source: How this data was obtained ('user', 'api', 'inferred')
            end_date: End date (usually NULL for current/latest version)
        """
        cursor = self._get_cursor()
        cursor.execute(
            """
            INSERT INTO symbol_attributes (
                symbol, effective_date, end_date, asset_class, geography, 
                region, sector, style, market_cap, fund_type, expense_ratio, 
                dividend_yield, is_esg, description, source, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (
                symbol,
                effective_date,
                end_date,
                asset_class,
                geography,
                region,
                sector,
                style,
                market_cap,
                fund_type,
                expense_ratio,
                dividend_yield,
                is_esg,
                description,
                source,
            ),
        )

    def upsert_symbol_attributes(
        self,
        symbol: str,
        effective_date: pd.Timestamp,
        asset_class: Optional[str] = None,
        geography: Optional[str] = None,
        region: Optional[str] = None,
        sector: Optional[str] = None,
        style: Optional[str] = None,
        market_cap: Optional[str] = None,
        fund_type: Optional[str] = None,
        expense_ratio: Optional[float] = None,
        dividend_yield: Optional[float] = None,
        is_esg: bool = False,
        description: Optional[str] = None,
        source: str = "user",
    ):
        """
        Insert or update symbol attributes for a specific effective date.

        This will update the existing record if (symbol, effective_date) already exists,
        or insert a new version if the effective_date is different.

        Args:
            symbol: Stock/ETF ticker
            effective_date: Date from which these attributes are effective
            [other args same as insert_symbol_attributes]
        """
        cursor = self._get_cursor()
        cursor.execute(
            """
            INSERT INTO symbol_attributes (
                symbol, effective_date, asset_class, geography, region, 
                sector, style, market_cap, fund_type, expense_ratio, 
                dividend_yield, is_esg, description, source, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(symbol, effective_date) DO UPDATE SET
                asset_class = excluded.asset_class,
                geography = excluded.geography,
                region = excluded.region,
                sector = excluded.sector,
                style = excluded.style,
                market_cap = excluded.market_cap,
                fund_type = excluded.fund_type,
                expense_ratio = excluded.expense_ratio,
                dividend_yield = excluded.dividend_yield,
                is_esg = excluded.is_esg,
                description = excluded.description,
                source = excluded.source,
                updated_at = datetime('now')
            """,
            (
                symbol,
                effective_date,
                asset_class,
                geography,
                region,
                sector,
                style,
                market_cap,
                fund_type,
                expense_ratio,
                dividend_yield,
                is_esg,
                description,
                source,
            ),
        )

    def get_symbol_attributes(
        self,
        symbol: Optional[str] = None,
        as_of_date: Optional[pd.Timestamp] = None,
        include_history: bool = False,
    ) -> pd.DataFrame:
        """Query symbol attributes."""
        if include_history:
            query = "SELECT * FROM symbol_attributes"
            params = []

            if symbol:
                query += " WHERE symbol = ?"
                params.append(symbol)

            query += " ORDER BY symbol, effective_date"

        elif as_of_date:
            query = """
            SELECT * FROM symbol_attributes sa
            WHERE sa.effective_date <= ?
            AND (sa.end_date IS NULL OR sa.end_date >= ?)
            """
            # convert Timestamp to string to satisfy DB parameter type expectations
            params = [as_of_date.isoformat(), as_of_date.isoformat()]

            if symbol:
                query += " AND sa.symbol = ?"
                params.append(symbol)

            query += " ORDER BY symbol, effective_date DESC"

        else:
            query = "SELECT * FROM current_symbol_attributes"
            params = []

        conn = self._get_connection()
        df = pd.read_sql_query(
            query, conn, params=self._normalize_params(params)
        )
        if not df.empty:
            df["effective_date"] = pd.to_datetime(
                df["effective_date"], utc=True
            )
            if "end_date" in df.columns:
                df["end_date"] = pd.to_datetime(df["end_date"], utc=True)
        return df

    def get_symbol_attributes_at_date(
        self,
        date: pd.Timestamp,
        symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get symbol attributes that were effective on a specific date."""
        query = """
        SELECT 
            sa.symbol,
            sa.effective_date,
            sa.end_date,
            sa.asset_class,
            sa.geography,
            sa.region,
            sa.sector,
            sa.style,
            sa.market_cap,
            sa.fund_type,
            sa.expense_ratio,
            sa.dividend_yield,
            sa.is_esg,
            sa.description,
            sa.source
        FROM symbol_attributes sa
        WHERE sa.effective_date <= ?
        AND (sa.end_date IS NULL OR sa.end_date >= ?)
        """
        params = [date.isoformat(), date.isoformat()]

        if symbol:
            query += " AND sa.symbol = ?"
            params.append(symbol)

        query += """
        AND sa.effective_date = (
            SELECT MAX(sa2.effective_date)
            FROM symbol_attributes sa2
            WHERE sa2.symbol = sa.symbol
            AND sa2.effective_date <= ?
            AND (sa2.end_date IS NULL OR sa2.end_date >= ?)
        )
        ORDER BY sa.symbol
        """
        params.extend([date.isoformat(), date.isoformat()])

        conn = self._get_connection()
        df = pd.read_sql_query(
            query, conn, params=self._normalize_params(params)
        )
        if not df.empty:
            df["effective_date"] = pd.to_datetime(
                df["effective_date"], utc=True
            )
            if "end_date" in df.columns:
                df["end_date"] = pd.to_datetime(df["end_date"], utc=True)
        return df

    def get_attribute_changes(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Get history of attribute changes."""
        query = """
        SELECT 
            symbol,
            effective_date,
            end_date,
            asset_class,
            geography,
            region,
            sector,
            style,
            market_cap,
            fund_type,
            source,
            updated_at
        FROM symbol_attributes
        WHERE 1=1
        """
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if start_date:
            query += " AND effective_date >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND effective_date <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY symbol, effective_date"

        conn = self._get_connection()
        df = pd.read_sql_query(
            query, conn, params=self._normalize_params(params)
        )
        if not df.empty:
            df["effective_date"] = pd.to_datetime(
                df["effective_date"], utc=True
            )
            if "end_date" in df.columns:
                df["end_date"] = pd.to_datetime(df["end_date"], utc=True)
            df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True)
        return df

    def bulk_upsert_symbol_attributes(
        self,
        attributes_df: pd.DataFrame,
        effective_date: Optional[pd.Timestamp] = None,
    ):
        """
        Bulk insert/update symbol attributes from a DataFrame.

        Args:
            attributes_df: DataFrame with columns matching symbol_attributes table
            effective_date: If provided, use this as effective_date for all rows.
                          Otherwise, must be in attributes_df
        """
        required_cols = ["symbol"]
        optional_cols = [
            "effective_date",
            "asset_class",
            "geography",
            "region",
            "sector",
            "style",
            "market_cap",
            "fund_type",
            "expense_ratio",
            "dividend_yield",
            "is_esg",
            "description",
            "source",
        ]

        for req_col in required_cols:
            if req_col not in attributes_df.columns:
                raise ValueError(
                    f"attributes_df must contain '{req_col}' column"
                )

        # Handle effective_date
        if effective_date is not None:
            attributes_df = attributes_df.copy()
            attributes_df["effective_date"] = effective_date
        elif "effective_date" not in attributes_df.columns:
            raise ValueError(
                "Either provide effective_date parameter or include "
                "'effective_date' column in attributes_df"
            )

        # Fill missing optional columns with None
        for col in optional_cols:
            if col not in attributes_df.columns:
                attributes_df[col] = None

        # Default source to 'user' if not provided
        if (
            "source" not in attributes_df.columns
            or attributes_df["source"].isna().all()
        ):
            attributes_df["source"] = "user"

        with self.transaction():
            for _, row in attributes_df.iterrows():
                self.upsert_symbol_attributes(
                    symbol=row["symbol"],
                    effective_date=pd.to_datetime(
                        row["effective_date"], utc=True
                    ),
                    asset_class=row.get("asset_class"),
                    geography=row.get("geography"),
                    region=row.get("region"),
                    sector=row.get("sector"),
                    style=row.get("style"),
                    market_cap=row.get("market_cap"),
                    fund_type=row.get("fund_type"),
                    expense_ratio=row.get("expense_ratio"),
                    dividend_yield=row.get("dividend_yield"),
                    is_esg=row.get("is_esg", False),
                    description=row.get("description"),
                    source=row.get("source", "user"),
                )

        logger.info("✅ Upserted %d symbol attributes", len(attributes_df))

    # ========== QUERY METHODS (unchanged from previous) ==========

    def get_portfolio_summary(
        self,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Query portfolio summary data."""
        query = "SELECT * FROM portfolio_summary"
        params = []

        if start_date or end_date:
            query += " WHERE "
            conditions = []
            if start_date:
                conditions.append("date >= ?")
                params.append(start_date)
            if end_date:
                conditions.append("date <= ?")
                params.append(end_date)
            query += " AND ".join(conditions)

        query += " ORDER BY date"

        conn = self._get_connection()
        df = pd.read_sql_query(
            query, conn, params=self._normalize_params(params)
        )
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], utc=True)
        return df

    def get_account_data(
        self,
        account: Optional[str] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Query account data."""
        query = "SELECT * FROM account_data"
        params = []
        conditions = []

        if account:
            conditions.append("account = ?")
            params.append(account)
        if start_date:
            conditions.append("date >= ?")
            params.append(start_date.isoformat())
        if end_date:
            conditions.append("date <= ?")
            params.append(end_date.isoformat())

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY date, account"

        conn = self._get_connection()
        df = pd.read_sql_query(
            query, conn, params=self._normalize_params(params)
        )
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], utc=True)
        return df

    def get_holdings(
        self,
        account: Optional[str] = None,
        symbol: Optional[str] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        date: Optional[pd.Timestamp] = None,
        include_zero: bool = False,
    ) -> pd.DataFrame:
        """Query holdings data."""
        query = "SELECT * FROM holdings"
        params = []
        conditions = []

        if date:
            conditions.append("date = ?")
            params.append(date.isoformat())
        else:
            if start_date:
                conditions.append("date >= ?")
                params.append(start_date.isoformat())
            if end_date:
                conditions.append("date <= ?")
                params.append(end_date.isoformat())

        if account:
            conditions.append("account = ?")
            params.append(account)
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)

        if not include_zero:
            conditions.append("(quantity != 0 OR value != 0)")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY date, account, symbol"

        conn = self._get_connection()
        df = pd.read_sql_query(
            query, conn, params=self._normalize_params(params)
        )
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], utc=True)
        return df

    def get_symbol_data(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Query symbol data."""
        query = "SELECT * FROM symbol_data"
        params = []
        conditions = []

        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if start_date:
            conditions.append("date >= ?")
            params.append(start_date.isoformat())
        if end_date:
            conditions.append("date <= ?")
            params.append(end_date.isoformat())

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY date, symbol"

        conn = self._get_connection()
        df = pd.read_sql_query(
            query, conn, params=self._normalize_params(params)
        )
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], utc=True)
        return df

    def get_latest_holdings(
        self,
        account: Optional[str] = None,
        include_zero: bool = False,
    ) -> pd.DataFrame:
        """Get the most recent holdings."""
        query = "SELECT * FROM latest_holdings"
        params = []
        conditions = []

        if account:
            conditions.append("account = ?")
            params.append(account)

        if not include_zero:
            conditions.append("(quantity != 0 OR value != 0)")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY account, symbol"

        conn = self._get_connection()
        df = pd.read_sql_query(
            query, conn, params=self._normalize_params(params)
        )
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], utc=True)
        return df

    def get_latest_portfolio(self) -> pd.Series:
        """Get the most recent portfolio summary as a Series."""
        conn = self._get_connection()
        df = pd.read_sql_query("SELECT * FROM latest_portfolio", conn)
        if df.empty:
            return pd.Series(dtype=float)

        df["date"] = pd.to_datetime(df["date"], utc=True)
        return df.iloc[0]

    def get_date_range(
        self,
    ) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """Get the min and max dates in the portfolio_summary table."""
        cursor = self._get_cursor()
        cursor.execute("SELECT MIN(date), MAX(date) FROM portfolio_summary")
        result = cursor.fetchone()

        if result and result[0] and result[1]:
            start = pd.to_datetime(result[0], utc=True)
            end = pd.to_datetime(result[1], utc=True)
            return start, end
        return None, None

    def get_accounts(self) -> List[str]:
        """Get list of all accounts in the database."""
        cursor = self._get_cursor()
        cursor.execute(
            "SELECT DISTINCT account FROM account_data ORDER BY account"
        )
        return [row[0] for row in cursor.fetchall()]

    def get_symbols(self) -> List[str]:
        """Get list of all symbols in the database."""
        cursor = self._get_cursor()
        cursor.execute(
            "SELECT DISTINCT symbol FROM symbol_data ORDER BY symbol"
        )
        return [row[0] for row in cursor.fetchall()]

    def get_holdings_with_prices(
        self,
        account: Optional[str] = None,
        symbol: Optional[str] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Get holdings joined with price data and temporal attributes."""
        query = "SELECT * FROM holdings_with_attributes"
        params = []
        conditions = []

        if account:
            conditions.append("account = ?")
            params.append(account)
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if start_date:
            conditions.append("date >= ?")
            params.append(start_date.isoformat())
        if end_date:
            conditions.append("date <= ?")
            params.append(end_date.isoformat())

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY date, account, symbol"

        conn = self._get_connection()
        df = pd.read_sql_query(
            query, conn, params=self._normalize_params(params)
        )
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], utc=True)
        return df

    # ========== ALLOCATION & PERFORMANCE WITH TEMPORAL ATTRIBUTES ==========

    def get_allocation_by_attribute(
        self,
        attribute: str,
        date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Get portfolio allocation by a specific attribute, using temporal attributes."""
        valid_attributes = [
            "asset_class",
            "geography",
            "region",
            "sector",
            "style",
            "market_cap",
            "fund_type",
        ]
        if attribute not in valid_attributes:
            raise ValueError(
                f"Invalid attribute: {attribute}. "
                f"Must be one of: {', '.join(valid_attributes)}"
            )

        if date is None:
            date_filter = "sd.date = (SELECT MAX(date) FROM symbol_data)"
            params = []
        else:
            date_filter = "sd.date = ?"
            params = [date]

        query = f"""
        SELECT 
            sa.{attribute} as category,
            SUM(sd.weight) as total_weight,
            SUM(sd.total_value) as total_value,
            COUNT(DISTINCT sd.symbol) as num_symbols,
            AVG(sd.twr_return) as avg_twr_return
        FROM symbol_data sd
        LEFT JOIN LATERAL (
            SELECT * FROM symbol_attributes sa_inner
            WHERE sa_inner.symbol = sd.symbol
            AND sa_inner.effective_date <= sd.date
            AND (sa_inner.end_date IS NULL OR sa_inner.end_date >= sd.date)
            ORDER BY sa_inner.effective_date DESC
            LIMIT 1
        ) sa ON TRUE
        WHERE {date_filter}
        GROUP BY sa.{attribute}
        ORDER BY total_weight DESC
        """

        conn = self._get_connection()
        df = pd.read_sql_query(
            query, conn, params=self._normalize_params(params)
        )
        return df

    def get_allocation_time_series(
        self,
        attribute: str,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Get portfolio allocation over time by attribute, with temporal tracking."""
        valid_attributes = [
            "asset_class",
            "geography",
            "region",
            "sector",
            "style",
            "market_cap",
            "fund_type",
        ]
        if attribute not in valid_attributes:
            raise ValueError(
                f"Invalid attribute: {attribute}. "
                f"Must be one of: {', '.join(valid_attributes)}"
            )

        query = f"""
        SELECT 
            sd.date,
            sa.{attribute} as category,
            SUM(sd.weight) as total_weight,
            SUM(sd.total_value) as total_value
        FROM symbol_data sd
        LEFT JOIN LATERAL (
            SELECT * FROM symbol_attributes sa_inner
            WHERE sa_inner.symbol = sd.symbol
            AND sa_inner.effective_date <= sd.date
            AND (sa_inner.end_date IS NULL OR sa_inner.end_date >= sd.date)
            ORDER BY sa_inner.effective_date DESC
            LIMIT 1
        ) sa ON TRUE
        WHERE 1=1
        """
        params = []

        if start_date:
            query += " AND sd.date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND sd.date <= ?"
            params.append(end_date)

        query += f" GROUP BY sd.date, sa.{attribute} ORDER BY sd.date, total_weight DESC"

        conn = self._get_connection()
        df = pd.read_sql_query(
            query, conn, params=self._normalize_params(params)
        )
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], utc=True)
        return df

    def get_performance_by_attribute(
        self,
        attribute: str,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Get performance metrics grouped by attribute, with temporal tracking."""
        valid_attributes = [
            "asset_class",
            "geography",
            "region",
            "sector",
            "style",
            "market_cap",
            "fund_type",
        ]
        if attribute not in valid_attributes:
            raise ValueError(
                f"Invalid attribute: {attribute}. "
                f"Must be one of: {', '.join(valid_attributes)}"
            )

        query = f"""
        SELECT 
            sa.{attribute} as category,
            AVG(sd.weight) as avg_weight,
            AVG(sd.twr_return) as avg_daily_return,
            MIN(sd.twr_return) as min_return,
            MAX(sd.twr_return) as max_return,
            COUNT(*) as num_observations
        FROM symbol_data sd
        LEFT JOIN LATERAL (
            SELECT * FROM symbol_attributes sa_inner
            WHERE sa_inner.symbol = sd.symbol
            AND sa_inner.effective_date <= sd.date
            AND (sa_inner.end_date IS NULL OR sa_inner.end_date >= sd.date)
            ORDER BY sa_inner.effective_date DESC
            LIMIT 1
        ) sa ON TRUE
        WHERE 1=1
        """
        params = []

        if start_date:
            query += " AND sd.date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND sd.date <= ?"
            params.append(end_date)

        query += f" GROUP BY sa.{attribute} ORDER BY avg_weight DESC"

        conn = self._get_connection()
        df = pd.read_sql_query(
            query, conn, params=self._normalize_params(params)
        )

        # Calculate volatility using pandas
        if not df.empty and start_date:
            detail_query = f"""
            SELECT 
                sa.{attribute} as category,
                sd.twr_return
            FROM symbol_data sd
            LEFT JOIN LATERAL (
                SELECT * FROM symbol_attributes sa_inner
                WHERE sa_inner.symbol = sd.symbol
                AND sa_inner.effective_date <= sd.date
                AND (sa_inner.end_date IS NULL OR sa_inner.end_date >= sd.date)
                ORDER BY sa_inner.effective_date DESC
                LIMIT 1
            ) sa ON TRUE
            WHERE sd.twr_return IS NOT NULL
            """
            detail_params = []
            if start_date:
                detail_query += " AND sd.date >= ?"
                detail_params.append(start_date)
            if end_date:
                detail_query += " AND sd.date <= ?"
                detail_params.append(end_date)

            detail_df = pd.read_sql_query(
                detail_query,
                conn,
                params=self._normalize_params(detail_params),
            )
            volatility = detail_df.groupby("category")["twr_return"].std()
            df = df.merge(
                volatility.rename("volatility").reset_index(),
                on="category",
                how="left",
            )

        return df

    # ========== STATISTICS ==========

    def get_stats(self) -> Dict:
        """Get database statistics."""
        cursor = self._get_cursor()

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
        return f"PortfolioDatabase(db_path='{self.db_path}')"
