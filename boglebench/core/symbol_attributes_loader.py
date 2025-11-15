"""
Symbol attributes extraction and database loading.

This module extracts symbol attributes (asset class, geography, sector, etc.)
from transaction data and loads them into the database with temporal tracking.
Handles attribute mapping and ensures attributes are properly associated with
their effective dates.
"""

from typing import List

import pandas as pd

from ..core.portfolio_db import PortfolioDatabase
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


def extract_symbol_attributes_from_transactions(
    transactions_df: pd.DataFrame,
    attribute_columns: List[str],
) -> pd.DataFrame:
    """
    Extract unique symbol attributes from transactions DataFrame.

    Args:
        transactions_df: Transactions DataFrame with attribute columns
        attribute_columns: List of column names containing attributes

    Returns:
        DataFrame with one row per symbol and its attributes
    """
    if not attribute_columns:
        logger.warning("No attribute columns found in transactions")
        return pd.DataFrame()

    # Get the most recent attributes for each symbol
    # (in case attributes changed between transactions)
    attributes_df = (
        transactions_df.sort_values("date")
        .groupby("symbol")[["symbol"] + attribute_columns]
        .last()
        .reset_index(drop=True)
    )

    # Map transaction column names to database schema
    column_mapping = {
        "group_asset_class": "asset_class",
        "group_geography": "geography",
        "group_region": "region",
        "group_sector": "sector",
        "group_style": "style",
        "group_market_cap": "market_cap",
        "group_fund_type": "fund_type",
        "asset_class": "asset_class",
        "geography": "geography",
        "region": "region",
        "sector": "sector",
        "style": "style",
        "market_cap": "market_cap",
        "fund_type": "fund_type",
    }

    # Rename columns that match mapping
    rename_dict = {
        col: column_mapping[col]
        for col in attributes_df.columns
        if col in column_mapping
    }
    attributes_df = attributes_df.rename(columns=rename_dict)

    # Add source tag
    attributes_df["source"] = "transactions"

    logger.info("Extracted attributes for %d symbols", len(attributes_df))

    return attributes_df


def load_symbol_attributes_to_database(
    db: PortfolioDatabase,
    transactions_df: pd.DataFrame,
    attribute_columns: List[str],
    effective_date: pd.Timestamp,
) -> None:
    """
    Extract attributes from transactions and load into database.

    Args:
        db: PortfolioDatabase instance
        transactions_df: Transactions DataFrame
        attribute_columns: List of attribute column names
        effective_date: Effective date for all attributes (usually portfolio start date)
    """
    if not attribute_columns:
        logger.info("No attribute columns to load")
        return

    logger.info("📊 Extracting symbol attributes from transactions...")

    attributes_df = extract_symbol_attributes_from_transactions(
        transactions_df, attribute_columns
    )

    if attributes_df.empty:
        logger.warning("No attributes extracted")
        return

    # Add effective_date
    attributes_df["effective_date"] = effective_date

    # Load to database
    db.bulk_upsert_symbol_attributes(
        attributes_df, effective_date=effective_date
    )

    logger.info(
        "✅ Loaded attributes for %d symbols with effective date %s",
        len(attributes_df),
        effective_date.date(),
    )


def load_symbol_attributes_from_csv(
    db: PortfolioDatabase,
    csv_path: str,
    effective_date: pd.Timestamp,
) -> None:
    """
    Load symbol attributes from a CSV file.

    CSV should have columns: symbol, asset_class, geography, region, sector,
    style, market_cap, fund_type, description

    Args:
        db: PortfolioDatabase instance
        csv_path: Path to CSV file
        effective_date: Effective date for all attributes
    """
    logger.info("📄 Loading symbol attributes from %s...", csv_path)

    try:
        attributes_df = pd.read_csv(csv_path)

        # Validate required column
        if "symbol" not in attributes_df.columns:
            raise ValueError("CSV must contain 'symbol' column")

        # Add metadata
        attributes_df["source"] = "csv_file"

        # Load to database
        db.bulk_upsert_symbol_attributes(
            attributes_df, effective_date=effective_date
        )

        logger.info(
            "✅ Loaded attributes for %d symbols from CSV", len(attributes_df)
        )

    except Exception as e:
        logger.error("Failed to load attributes from CSV: %s", e)
        raise
