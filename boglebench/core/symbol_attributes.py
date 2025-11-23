"""
Temporal symbol attributes management.

This module provides methods for managing time-based symbol attributes in the
portfolio database. Attributes like asset class, geography, sector, etc. can
change over time, and this module tracks those changes with effective dates.
"""

from typing import Optional, cast

import pandas as pd

from ..utils.logging_config import get_logger
from .portfolio_db_mixins_protocol import DatabaseProtocol

logger = get_logger(__name__)


class SymbolAttributesMixin:
    """Mixin class providing methods for managing temporal symbol attributes."""

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
        cursor = cast(DatabaseProtocol, self).get_cursor()
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
        cursor = cast(DatabaseProtocol, self).get_cursor()
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

        conn = cast(DatabaseProtocol, self).get_connection()
        df = pd.read_sql_query(
            query,
            conn,
            params=cast(DatabaseProtocol, self).normalize_params(params),
            parse_dates=["effective_date", "end_date"],
        )
        date_cols = ["effective_date"]
        if "end_date" in df.columns:
            date_cols.append("end_date")
        return cast(DatabaseProtocol, self).ensure_datetime_utc(df, date_cols)

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

        conn = cast(DatabaseProtocol, self).get_connection()
        df = pd.read_sql_query(
            query,
            conn,
            params=cast(DatabaseProtocol, self).normalize_params(params),
            parse_dates=["effective_date", "end_date"],
        )
        date_cols = ["effective_date"]
        if "end_date" in df.columns:
            date_cols.append("end_date")
        return cast(DatabaseProtocol, self).ensure_datetime_utc(df, date_cols)

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

        conn = cast(DatabaseProtocol, self).get_connection()
        df = pd.read_sql_query(
            query,
            conn,
            params=cast(DatabaseProtocol, self).normalize_params(params),
            parse_dates=["effective_date", "end_date", "updated_at"],
        )
        date_cols = ["effective_date", "updated_at"]
        if "end_date" in df.columns:
            date_cols.append("end_date")
        return cast(DatabaseProtocol, self).ensure_datetime_utc(df, date_cols)

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

        with cast(DatabaseProtocol, self).transaction():
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

    def get_attribute_coverage(self) -> pd.DataFrame:
        """
        Get statistics on attribute coverage across all symbols.

        Returns:
            DataFrame with columns:
            - attribute: Attribute name
            - count: Number of symbols with this attribute
            - percentage: Percentage of symbols with this attribute
            - unique_values: Number of unique values for this attribute
        """
        attribute_columns = [
            "asset_class",
            "geography",
            "region",
            "sector",
            "style",
            "market_cap",
            "fund_type",
        ]

        df = self.get_symbol_attributes()

        if df.empty:
            return pd.DataFrame(
                columns=["attribute", "count", "percentage", "unique_values"]
            )

        total_symbols = len(df)
        stats = []

        for col in attribute_columns:
            if col in df.columns:
                non_null_count = df[col].notna().sum()
                unique_count = df[col].nunique()
                percentage = (
                    (non_null_count / total_symbols * 100)
                    if total_symbols > 0
                    else 0
                )

                stats.append(
                    {
                        "attribute": col,
                        "count": non_null_count,
                        "percentage": percentage,
                        "unique_values": unique_count,
                    }
                )

        return pd.DataFrame(stats)

    def get_attribute_columns_in_use(
        self, full_coverage: bool = False
    ) -> list[str]:
        """
        Get list of attribute columns that have data in the database.

        Args:
            full_coverage: If True, only return attributes that have values
                           for all symbols. If False, return attributes with
                           values for at least one symbol.
        Returns:
            List of attribute column names in use.
        """
        coverage_df = self.get_attribute_coverage()
        if coverage_df.empty:
            return []

        df = self.get_symbol_attributes()
        total_symbols = len(df["symbol"].unique())

        if full_coverage:
            in_use = coverage_df[coverage_df["count"] == total_symbols][
                "attribute"
            ].tolist()
        else:
            in_use = coverage_df[coverage_df["count"] > 0][
                "attribute"
            ].tolist()

        return in_use
