"""
Symbol attributes loading and database management.

This module loads symbol attributes (asset class, geography, sector, etc.)
from CSV files or external APIs and stores them in the database with temporal tracking.
Attributes are completely separate from transaction data and must be loaded independently.
"""

import pandas as pd

from ..core.portfolio_db import PortfolioDatabase
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


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
