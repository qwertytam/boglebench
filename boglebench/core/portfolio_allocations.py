"""
Portfolio allocation analysis methods.

This module provides methods for analyzing portfolio allocations by different
attributes (account, asset class, geography, sector, etc.). Calculates
allocation percentages, performance by grouping, and attribution analysis
at various levels of granularity.
"""

from typing import Optional, cast

import pandas as pd

from .portfolio_db_mixins_protocol import DatabaseProtocol


class PortfolioAllocationMixin:
    """Mixin class providing allocation and performance analysis methods."""

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

        conn = cast(DatabaseProtocol, self).get_connection()
        df = pd.read_sql_query(
            query,
            conn,
            params=cast(DatabaseProtocol, self).normalize_params(params),
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

        conn = cast(DatabaseProtocol, self).get_connection()
        df = pd.read_sql_query(
            query,
            conn,
            params=cast(DatabaseProtocol, self).normalize_params(params),
            parse_dates=["date"],
        )
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

        conn = cast(DatabaseProtocol, self).get_connection()
        df = pd.read_sql_query(
            query,
            conn,
            params=cast(DatabaseProtocol, self).normalize_params(params),
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
                params=cast(DatabaseProtocol, self).normalize_params(
                    detail_params
                ),
            )
            volatility = detail_df.groupby("category")["twr_return"].std()
            df = df.merge(
                volatility.rename("volatility").reset_index(),
                on="category",
                how="left",
            )

        return df
