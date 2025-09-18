"""
Unit tests for transaction validation functions."""

import pandas as pd
import pytest

from boglebench.core.transaction_loader import (
    get_valid_transactions,
    is_series_valid,
)

# --- Test Cases ---


@pytest.fixture
def series_all_valid() -> pd.Series:
    """A pandas Series where all transaction types are valid."""
    return pd.Series(["BUY", "SELL", "DIVIDEND", "FEE", "SPLIT"])


@pytest.fixture
def series_with_invalid() -> pd.Series:
    """A pandas Series with a mix of valid and invalid types."""
    return pd.Series(["BUY", "INVALID_TYPE", "SELL", "ANOTHER_BAD_ONE"])


@pytest.fixture
def series_all_invalid() -> pd.Series:
    """A pandas Series where all transaction types are invalid."""
    return pd.Series(["UNKNOWN", "TRANSFER", "ADJUSTMENT"])


@pytest.fixture
def series_empty() -> pd.Series:
    """An empty pandas Series."""
    return pd.Series([], dtype=str)


# --- Tests for get_valid_transactions() ---


def test_get_valid_transactions_all_valid(series_all_valid):
    """
    Tests that it returns all True for a fully valid series.
    """
    expected = pd.Series([True, True, True, True, True])
    result = get_valid_transactions(series_all_valid)
    pd.testing.assert_series_equal(result, expected)


def test_get_valid_transactions_with_invalid(series_with_invalid):
    """
    Tests that it correctly creates a boolean mask for a mixed series.
    """
    expected = pd.Series([True, False, True, False])
    result = get_valid_transactions(series_with_invalid)
    pd.testing.assert_series_equal(result, expected)


def test_get_valid_transactions_all_invalid(series_all_invalid):
    """
    Tests that it returns all False for a fully invalid series.
    """
    expected = pd.Series([False, False, False])
    result = get_valid_transactions(series_all_invalid)
    pd.testing.assert_series_equal(result, expected)


def test_get_valid_transactions_empty(series_empty):
    """
    Tests that it returns an empty boolean Series for empty input.
    """
    expected = pd.Series([], dtype=bool)
    result = get_valid_transactions(series_empty)
    pd.testing.assert_series_equal(result, expected)


# --- Tests for is_series_valid() ---


def test_is_series_valid_all_valid(series_all_valid):
    """
    Tests that it returns True when all transaction types are valid.
    """
    assert is_series_valid(series_all_valid) is True


def test_is_series_valid_with_invalid(series_with_invalid):
    """
    Tests that it returns False when the series contains invalid types.
    """
    assert is_series_valid(series_with_invalid) is False


def test_is_series_valid_all_invalid(series_all_invalid):
    """
    Tests that it returns False when all types are invalid.
    """
    assert is_series_valid(series_all_invalid) is False


def test_is_series_valid_empty(series_empty):
    """
    Tests that it returns True for an empty series (as there are no invalid items).
    """
    assert is_series_valid(series_empty) is True
