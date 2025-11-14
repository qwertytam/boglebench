"""
Query methods for PortfolioDatabase.
Handles read operations for portfolio, account, holdings, and symbol data.
"""

from typing import List, Optional, Tuple

import pandas as pd

from .portfolio_db_mixins_protocol import DatabaseProtocol


class PortfolioQueryMixin(DatabaseProtocol):
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
                params.append(start_date)
            if end_date:
                conditions.append("date <= ?")
                params.append(end_date)
            query += " AND ".join(conditions)

        query += " ORDER BY date"

        conn = self.get_connection()
        df = pd.read_sql_query(
            query, conn, params=self.normalize_params(params)
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

        conn = self.get_connection()
        df = pd.read_sql_query(
            query, conn, params=self.normalize_params(params)
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

        conn = self.get_connection()
        df = pd.read_sql_query(
            query, conn, params=self.normalize_params(params)
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

        conn = self.get_connection()
        df = pd.read_sql_query(
            query, conn, params=self.normalize_params(params)
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

        conn = self.get_connection()
        df = pd.read_sql_query(
            query, conn, params=self.normalize_params(params)
        )
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], utc=True)
        return df

    def get_latest_portfolio(self) -> pd.Series:
        """Get the most recent portfolio summary as a Series."""
        conn = self.get_connection()
        df = pd.read_sql_query("SELECT * FROM latest_portfolio", conn)
        if df.empty:
            return pd.Series(dtype=float)

        df["date"] = pd.to_datetime(df["date"], utc=True)
        return df.iloc[0]

    def get_date_range(
        self,
    ) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """Get the min and max dates in the portfolio_summary table."""
        cursor = self.get_cursor()
        cursor.execute("SELECT MIN(date), MAX(date) FROM portfolio_summary")
        result = cursor.fetchone()

        if result and result[0] and result[1]:
            start = pd.to_datetime(result[0], utc=True)
            end = pd.to_datetime(result[1], utc=True)
            return start, end
        return None, None

    def get_accounts(self) -> List[str]:
        """Get list of all accounts in the database."""
        cursor = self.get_cursor()
        cursor.execute(
            "SELECT DISTINCT account FROM account_data ORDER BY account"
        )
        return [row[0] for row in cursor.fetchall()]

    def get_symbols(self) -> List[str]:
        """Get list of all symbols in the database."""
        cursor = self.get_cursor()
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

        conn = self.get_connection()
        df = pd.read_sql_query(
            query, conn, params=self.normalize_params(params)
        )
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], utc=True)
        return df
