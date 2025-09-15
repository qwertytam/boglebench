"""Module for loading and validating transaction data."""

import re
from datetime import tzinfo
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from ..utils.logging_config import get_logger
from ..utils.tools import to_tzts
from .constants import (
    DateAndTimeConstants,
    Defaults,
    DividendTypes,
    TransactionTypes,
)

logger = get_logger()


def _is_iso8601_date(date_str: str) -> bool:
    """Check if date string is in ISO8601 format (YYYY-MM-DD)."""

    # Pattern for YYYY-MM-DD format
    iso_pattern = r"^\d{4}-\d{2}-\d{2}"

    return bool(re.match(iso_pattern, date_str))


def get_valid_transactions(series: pd.Series) -> pd.Series:
    """
    Identifies transactions in a Series with valid types.

    Args:
        series: A pandas Series containing transaction type strings.

    Returns:
        A Series of booleans indicating which transactions are valid.
    """
    # Get the set of all valid transaction type strings from our Enum
    valid_types = TransactionTypes.all_types()

    # Return the unique values from the original series that were invalid
    return series.isin(valid_types)


def is_series_valid(series: pd.Series) -> bool:
    """
    Checks if all transaction types in a Series are valid.

    Args:
        series: A pandas Series containing transaction type strings.

    Returns:
        True if all types are valid, False otherwise.
    """
    valid_transactions_mask = get_valid_transactions(series)
    invalid_transactions = series[~valid_transactions_mask]
    return len(invalid_transactions) == 0


def _clean_transaction_data(
    df: pd.DataFrame,
    default_tz: Union[str, tzinfo] = DateAndTimeConstants.TZ_UTC.value,
) -> pd.DataFrame:
    """Clean and validate transaction data."""
    # Make a copy to avoid modifying original
    df = df.copy()

    # Validate required columns
    reqd_columns = [
        "date",
        "ticker",
        "transaction_type",
        "quantity",
        "value_per_share",
        "total_value",
    ]

    missing_columns = [col for col in reqd_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Convert date column - enforce ISO8601 format
    try:
        # First check if dates are in ISO8601 format (YYYY-MM-DD)
        for i, date_str in enumerate(df["date"]):
            if not _is_iso8601_date(str(date_str)):
                raise ValueError(
                    f"Date at row {i} ('{date_str}')"
                    f" is not in ISO8601 format (YYYY-MM-DD)."
                    f" Please use format like '2023-01-15'."
                )

        # df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        df["date"] = to_tzts(df["date"], tz=default_tz, format="%Y-%m-%d")
        logger.debug("Converted 'date' column to type %s", df["date"].dtype)
    except ValueError as e:
        if "is not in ISO8601 format" in str(e):
            raise e  # Re-raise our custom error
        else:
            raise ValueError(f"Error parsing dates: {e}") from e
    except Exception as e:
        raise ValueError(f"Error parsing dates: {e}") from e

    # Clean ticker symbols (uppercase, strip whitespace)
    df["ticker"] = df["ticker"].str.upper().str.strip()

    opt_columns = [
        "account",
        "group1",
        "group2",
        "group3",
        "div_type",
        "div_pay_date",
        "div_record_date",
        "div_ex_date",
        "split_ratio",
        "notes",
    ]

    missing_cols = [col for col in reqd_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Add optional columns if not present with default values
    for col in opt_columns:
        if col not in df.columns:
            if col == "account":
                df[col] = "Default"
            elif col.startswith("group"):
                df[col] = "Unassigned"
            elif col == "notes":
                df[col] = ""
            elif col == "div_type":
                df[col] = DividendTypes.CASH
            elif col == "div_pay_date":
                df[col] = df["date"]
            elif col in ["div_record_date", "div_ex_date"]:
                df[col] = pd.NaT
            elif col == "split_ratio":
                df[col] = Defaults.DEFAULT_ZERO

            logger.debug("ℹ️  No '%s' column found. Added default values.", col)

    # Convert optional date columns
    for date_col in ["div_pay_date", "div_ex_date", "div_record_date"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Clean account names (strip whitespace, title case)
    df["account"] = df["account"].str.strip().str.title()

    # Clean Group columns (strip whitespace, title case)
    for group_col in ["group1", "group2", "group3"]:
        if group_col in df.columns:
            df[group_col] = (
                df[group_col]
                .fillna("Unassigned")
                .astype(str)
                .str.strip()
                .str.title()
            )

    # Clean Notes column (strip whitespace only, preserve case)
    if "notes" in df.columns:
        df["notes"] = df["notes"].fillna("").astype(str).str.strip()

    df["transaction_type"] = df["transaction_type"].str.upper().str.strip()
    invalid_types = df[~get_valid_transactions(df["transaction_type"])]
    if not invalid_types.empty:
        invalid_type_list = invalid_types["transaction_type"].unique()
        raise ValueError(
            f"Invalid transaction types found: {invalid_type_list}"
        )

    # Validate numeric fields
    numeric_columns = ["quantity", "value_per_share", "total_value"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            raise ValueError(f"Invalid numeric values in column: {col}")
        if col in ["quantity", "value_per_share"]:
            is_dividend_or_fee = df["transaction_type"].isin(
                [TransactionTypes.DIVIDEND, TransactionTypes.FEE]
            )
            if (df[col] < 0).any() or (
                (df[col] == 0) & ~is_dividend_or_fee
            ).any():
                raise ValueError(
                    f"Non-positive values found in column: {col} "
                    f"for non-dividend/fee transactions"
                )

    # Check total value for each transaction
    # Ensure positive for checking - will be adjusted for SELL later
    df["total_value"] = np.abs(df["total_value"])

    error_limit = 0.01  # In dollars; allow small rounding errors
    exclude_ttype = [
        TransactionTypes.DIVIDEND,
        TransactionTypes.DIVIDEND_REINVEST,
        TransactionTypes.FEE,
    ]
    df.loc[
        ~df["transaction_type"].isin(exclude_ttype),
        "total_value_check",
    ] = (
        df.loc[
            ~df["transaction_type"].isin(exclude_ttype),
            "quantity",
        ]
        * df.loc[
            ~df["transaction_type"].isin(exclude_ttype),
            "value_per_share",
        ]
    )
    value_mismatch = (
        np.abs(df["total_value"] - df["total_value_check"]) > error_limit
    )
    if value_mismatch.any():
        mismatch_rows = df[value_mismatch]
        logger.warning(
            "⚠️  Total value mismatch in %s transactions. "
            "Using quantity * value_per_share.",
            len(mismatch_rows),
        )
        logger.debug(
            "Original data with mismatches:\n%s",
            mismatch_rows[
                [
                    "date",
                    "ticker",
                    "transaction_type",
                    "quantity",
                    "value_per_share",
                    "total_value",
                    "total_value_check",
                ]
            ],
        )
        df.loc[value_mismatch, "total_value"] = df.loc[
            value_mismatch, "total_value_check"
        ]

    # For SELL transactions, make shares negative for easier calculations
    df.loc[df["transaction_type"] == TransactionTypes.SELL, "quantity"] *= -1
    df.loc[
        df["transaction_type"] == TransactionTypes.SELL, "total_value"
    ] *= -1

    # For DIVIDEND transactions, make quantity zero and total value negative
    df.loc[df["transaction_type"] == TransactionTypes.DIVIDEND, "quantity"] = 0
    df.loc[
        df["transaction_type"] == TransactionTypes.DIVIDEND, "total_value"
    ] *= -1

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)
    logger.debug(
        "Cleaned transaction data:\n%s \nwith columns of types:\n%s",
        df.head(),
        df.dtypes,
    )
    return df


def load_validate_transactions(file_path: Path) -> pd.DataFrame:
    """
    Load and validate transaction data from CSV file.

    Args:
        file_path: Path to transactions CSV

    Returns:
        DataFrame with processed transaction data

    Raises:
        FileNotFoundError: If transaction file doesn't exist
        ValueError: If required columns are missing or data is invalid
    """

    if not file_path.exists():
        raise FileNotFoundError(f"Transaction file not found: {file_path}")

    logger.debug("Loading transactions from: %s", file_path)

    # Load CSV with flexible parsing
    try:
        df = pd.read_csv(file_path)
    except ValueError as e:
        raise ValueError(f"Error reading CSV file: {e}") from e

    # Clean and validate data
    df = _clean_transaction_data(df)

    logger.debug(
        "✅ Loaded %d transactions for %d unique assets with date range: %s to %s",
        len(df),
        df["ticker"].nunique(),
        df["date"].min(),
        df["date"].max(),
    )

    return df
