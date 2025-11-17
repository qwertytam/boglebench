"""
Database query operations for portfolio data.

This module provides methods for querying portfolio summary, account data,
holdings, and symbol attributes from the normalized database. Supports
filtering by date ranges and other criteria for flexible data retrieval.
"""

from typing import List, Optional, Tuple, cast

import pandas as pd

from .portfolio_db_mixins_protocol import DatabaseProtocol


class PortfolioQueryMixin:
    """Mixin class providing query methods for portfolio data."""

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
                params.append(start_date.isoformat())
            if end_date:
                conditions.append("date <= ?")
                params.append(end_date.isoformat())
            query += " AND ".join(conditions)

        query += " ORDER BY date"

        conn = cast(DatabaseProtocol, self).get_connection()
        df = pd.read_sql_query(
            query,
            conn,
            params=cast(DatabaseProtocol, self).normalize_params(params),
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

        conn = cast(DatabaseProtocol, self).get_connection()
        df = pd.read_sql_query(
            query,
            conn,
            params=cast(DatabaseProtocol, self).normalize_params(params),
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

        conn = cast(DatabaseProtocol, self).get_connection()
        df = pd.read_sql_query(
            query,
            conn,
            params=cast(DatabaseProtocol, self).normalize_params(params),
        )
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], utc=True)
        return df

    def get_symbol_data(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Query symbol data."""
        query = "SELECT * FROM symbol_data"
        params = []
        conditions = []

        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)

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

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY date, symbol"

        conn = cast(DatabaseProtocol, self).get_connection()
        df = pd.read_sql_query(
            query,
            conn,
            params=cast(DatabaseProtocol, self).normalize_params(params),
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

        conn = cast(DatabaseProtocol, self).get_connection()
        df = pd.read_sql_query(
            query,
            conn,
            params=cast(DatabaseProtocol, self).normalize_params(params),
        )
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], utc=True)
        return df

    def get_latest_portfolio(self) -> pd.Series:
        """Get the most recent portfolio summary as a Series."""
        conn = cast(DatabaseProtocol, self).get_connection()
        df = pd.read_sql_query("SELECT * FROM latest_portfolio", conn)
        if df.empty:
            return pd.Series(dtype=float)

        df["date"] = pd.to_datetime(df["date"], utc=True)
        return df.iloc[0]

    def get_date_range(
        self,
    ) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """Get the min and max dates in the portfolio_summary table."""
        cursor = cast(DatabaseProtocol, self).get_cursor()
        cursor.execute("SELECT MIN(date), MAX(date) FROM portfolio_summary")
        result = cursor.fetchone()

        if result and result[0] and result[1]:
            start = pd.to_datetime(result[0], utc=True)
            end = pd.to_datetime(result[1], utc=True)
            return start, end
        return None, None

    def get_accounts(self) -> List[str]:
        """Get list of all accounts in the database."""
        cursor = cast(DatabaseProtocol, self).get_cursor()
        cursor.execute(
            "SELECT DISTINCT account FROM account_data ORDER BY account"
        )
        return [row[0] for row in cursor.fetchall()]

    def get_symbols(self) -> List[str]:
        """Get list of all symbols in the database."""
        cursor = cast(DatabaseProtocol, self).get_cursor()
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

        conn = cast(DatabaseProtocol, self).get_connection()
        df = pd.read_sql_query(
            query,
            conn,
            params=cast(DatabaseProtocol, self).normalize_params(params),
        )
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], utc=True)
        return df

    def get_cash_flows(
        self,
        accounts: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None,
        date: Optional[pd.Timestamp] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Query cash flows for specified accounts and symbols.

        Cash flows are derived from the symbol_data table (aggregated across accounts)
        or can be filtered by account through holdings with calculated cash flows.

        Args:
            accounts: List of account names to filter by. If None, all accounts.
            symbols: List of symbol tickers to filter by. If None, all symbols.
            date: Specific date to query. Mutually exclusive with start_date/end_date.
            start_date: Start date for date range query.
            end_date: End date for date range query.

        Returns:
            DataFrame with columns: date, symbol, cash_flow
            If accounts specified, also includes: account (via join with holdings)

        Examples:
            # Get cash flows for AAPL in Test_Account for a specific date
            df = db.get_cash_flows(
                accounts=['Test_Account'],
                symbols=['AAPL'],
                date=pd.Timestamp('2023-06-05', tz='UTC')
            )

            # Get cash flows for multiple symbols across date range
            df = db.get_cash_flows(
                symbols=['AAPL', 'GOOGL'],
                start_date=pd.Timestamp('2023-06-01', tz='UTC'),
                end_date=pd.Timestamp('2023-06-30', tz='UTC')
            )

            # Get all cash flows for all accounts and symbols
            df = db.get_cash_flows()
        """
        params = []
        conditions = []

        # If accounts are specified, we need to join with holdings to filter by account
        if accounts:
            query = """
            SELECT 
                h.date,
                h.account,
                h.symbol,
                sd.cash_flow
            FROM holdings h
            INNER JOIN symbol_data sd ON h.date = sd.date AND h.symbol = sd.symbol
            WHERE 1=1
            """
        else:
            query = """
            SELECT 
                date,
                symbol,
                cash_flow
            FROM symbol_data
            WHERE 1=1
            """

        # Date filtering
        if accounts:
            date_prefix = "h.date"
        else:
            date_prefix = "date"

        if date:
            conditions.append(f"{date_prefix} = ?")
            params.append(date.isoformat())
        else:
            if start_date:
                conditions.append(f"{date_prefix} >= ?")
                params.append(start_date.isoformat())
            if end_date:
                conditions.append(f"{date_prefix} <= ?")
                params.append(end_date.isoformat())

        # Account filtering (only applies if joining with holdings)
        if accounts:
            if len(accounts) == 1:
                conditions.append("h.account = ?")
                params.append(accounts[0])
            else:
                placeholders = ",".join("?" * len(accounts))
                conditions.append(f"h.account IN ({placeholders})")
                params.extend(accounts)

        # Symbol filtering
        if symbols:
            symbol_prefix = "h.symbol" if accounts else "symbol"
            if len(symbols) == 1:
                conditions.append(f"{symbol_prefix} = ?")
                params.append(symbols[0])
            else:
                placeholders = ",".join("?" * len(symbols))
                conditions.append(f"{symbol_prefix} IN ({placeholders})")
                params.extend(symbols)

        # Add conditions to query
        if conditions:
            query += " AND " + " AND ".join(conditions)

        # Order results
        if accounts:
            query += " ORDER BY h.date, h.account, h.symbol"
        else:
            query += " ORDER BY date, symbol"

        conn = cast(DatabaseProtocol, self).get_connection()
        df = pd.read_sql_query(
            query,
            conn,
            params=cast(DatabaseProtocol, self).normalize_params(params),
        )

        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], utc=True)

        return df
