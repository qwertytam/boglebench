"""
Collection of utility functions and classes for Boglebench

"""

from __future__ import annotations

import json
from datetime import date, tzinfo
from datetime import datetime as dt
from typing import Any, Literal, Optional, Union
from zoneinfo import ZoneInfo  # pylint: disable=wrong-import-order

import numpy as np
import pandas as pd

from ..core.constants import DateAndTimeConstants, TransactionTypes
from ..core.types import DateLike, NonExistentTime, SeqLike
from ..utils.logging_config import get_logger

logger = get_logger()


def to_tzts(
    x: Union[DateLike, SeqLike, None],
    tz: Union[str, tzinfo, None] = DateAndTimeConstants.TZ_UTC.value,
    *,
    default: Union[DateLike, None] = None,
    errors: Literal["raise", "coerce", "ignore"] = "raise",
    nonexistent: NonExistentTime = "shift_forward",
    ambiguous: Any = "NaT",
    **to_datetime_kwargs: Any,
) -> Union[pd.Timestamp, pd.Series, None]:
    """
    Convert a scalar or sequence of date-like values to a pandas Timestamp.

    This function combines timezone localization/conversion with robust parsing
    and fallback defaults.

    Behavior:
    - Handles scalar values (Timestamp, datetime, str, etc.) and sequences
      (Series, list, Index).
    - If `tz` is provided:
        - Naïve timestamps are localized to that timezone.
        - Tz-aware timestamps are converted to that timezone.
    - If a value is missing or fails to parse:
        - If `default` is set, it's used as a fallback.
        - Otherwise, behavior depends on the `errors` parameter.

    Parameters
    ----------
    x : DateLike | SeqLike | None
        The primary input(s) to convert.
    tz : str | tzinfo | None, optional
        IANA timezone name or tzinfo object. Defaults to UTC. If None, the
        function will not apply any timezone localization.
    default : DateLike | None, optional
        A fallback value to parse if the primary input is missing or invalid.
    errors : {'raise', 'coerce', 'ignore'}, default 'raise'
        Determines how to handle parsing errors, passed to `pd.to_datetime`.
        - 'raise': raise an exception.
        - 'coerce': return `NaT` for parsing errors.
        - 'ignore': return the original input for parsing errors.
    nonexistent : str or timedelta, default 'shift_forward'
        How to handle wall times that fall in a DST gap.
    ambiguous : str, bool, or array-like, default 'NaT'
        How to handle wall times that fall in a DST overlap.
    **to_datetime_kwargs : Any
        Additional keyword arguments passed to `pd.to_datetime`.

    Returns
    -------
    pd.Timestamp | pd.Series | None
        The converted and timezone-aware pandas object(s).

    Raises
    ------
    TypeError
        If the input type is not supported.
    ValueError
        If `errors='raise'` and a value cannot be parsed.
    """
    tz_obj = ZoneInfo(str(tz)) if isinstance(tz, str) else tz

    def _convert_one(val: Union[DateLike, None]) -> Optional[pd.Timestamp]:
        """Processes a single scalar value."""
        # Use default if initial value is None or an empty string
        if val is None or (isinstance(val, str) and not val.strip()):
            val = default

        # If still no value, return None (which pandas will handle as NaT)
        if val is None:
            return None

        # Attempt to convert to a timestamp
        ts = pd.to_datetime(val, errors=errors, **to_datetime_kwargs)
        if pd.isna(ts):
            return None  # Return None for NaT to satisfy type checkers

        # Apply timezone if specified
        if tz_obj:
            if ts.tz is None:
                return ts.tz_localize(
                    tz_obj, nonexistent=nonexistent, ambiguous=ambiguous
                )
            return ts.tz_convert(tz_obj)
        return ts

    # Process based on input type
    if isinstance(x, pd.Series):
        return x.apply(_convert_one)
    if isinstance(x, (list, tuple, pd.Index)):
        # Let Series constructor handle conversion of None to NaT
        return pd.Series([_convert_one(v) for v in x])
    if (
        isinstance(x, (pd.Timestamp, dt, date, str, int, float, np.datetime64))
        or x is None
    ):
        return _convert_one(x)

    logger.error("to_timestamp: unsupported input type %s", type(x))
    raise TypeError(f"Unsupported input type for to_timestamp: {type(x)}")


def to_tzts_scaler(
    x: Union[DateLike, None],
    tz: Union[str, tzinfo, None] = DateAndTimeConstants.TZ_UTC.value,
    *,
    default: Union[DateLike, None] = None,
    errors: Literal["raise", "coerce", "ignore"] = "raise",
    nonexistent: NonExistentTime = "shift_forward",
    ambiguous: Any = "NaT",
    **to_datetime_kwargs: Any,
) -> Union[pd.Timestamp, None]:
    """
    Convert a scalar date-like value to a pandas Timestamp.

    This function combines timezone localization/conversion with robust parsing
    and fallback defaults.

    Behavior:
    - Handles scalar values (Timestamp, datetime, str, etc.) and sequences
      (Series, list, Index).
    - If `tz` is provided:
        - Naïve timestamps are localized to that timezone.
        - Tz-aware timestamps are converted to that timezone.
    - If a value is missing or fails to parse:
        - If `default` is set, it's used as a fallback.
        - Otherwise, behavior depends on the `errors` parameter.

    Parameters
    ----------
    x : DateLike | SeqLike | None
        The primary input(s) to convert.
    tz : str | tzinfo | None, optional
        IANA timezone name or tzinfo object. Defaults to UTC. If None, the
        function will not apply any timezone localization.
    default : DateLike | None, optional
        A fallback value to parse if the primary input is missing or invalid.
    errors : {'raise', 'coerce', 'ignore'}, default 'raise'
        Determines how to handle parsing errors, passed to `pd.to_datetime`.
        - 'raise': raise an exception.
        - 'coerce': return `NaT` for parsing errors.
        - 'ignore': return the original input for parsing errors.
    nonexistent : str or timedelta, default 'shift_forward'
        How to handle wall times that fall in a DST gap.
    ambiguous : str, bool, or array-like, default 'NaT'
        How to handle wall times that fall in a DST overlap.
    **to_datetime_kwargs : Any
        Additional keyword arguments passed to `pd.to_datetime`.

    Returns
    -------
    pd.Timestamp | pd.Series | None
        The converted and timezone-aware pandas object(s).

    Raises
    ------
    TypeError
        If the input type is not supported.
    ValueError
        If `errors='raise'` and a value cannot be parsed.
    """
    result = to_tzts(
        x,
        tz,
        default=default,
        errors=errors,
        nonexistent=nonexistent,
        ambiguous=ambiguous,
        **to_datetime_kwargs,
    )
    if isinstance(result, pd.Series):
        raise TypeError("to_tzts_scaler only accepts scalar inputs")
    return result


def is_tz_aware(x: Union[pd.Timestamp, pd.Series]) -> bool:
    """
    Check if a pandas Timestamp or Series is timezone-aware.

    Parameters
    ----------
    x : pd.Timestamp | pd.Series
        The scalar or series to check.

    Returns
    -------
    bool
        True if the object has timezone information, False otherwise.

    Raises
    ------
    TypeError
        If the input is not a pandas Timestamp or Series.
    """
    if isinstance(x, pd.Timestamp):
        # For a scalar, check the tzinfo attribute
        return x.tzinfo is not None
    if isinstance(x, pd.Series):
        # For a series, check the tz of its dtype
        if pd.api.types.is_datetime64_any_dtype(x.dtype):
            return x.dt.tz is not None
        # The series is not a datetime series
        return False

    raise TypeError(
        f"Input must be a pandas Timestamp or Series, not {type(x)}"
    )


def ensure_timestamp(
    value: Union[DateLike, None],
    default: Union[DateLike, None] = None,
    *,
    tz: Optional[str] = None,
) -> pd.Timestamp:
    """
    Convert `value` to a pandas.Timestamp.

    Behavior
    --------
    - Accepts a `pd.Timestamp`, a date-like `str`, or `None`.
    - If `value` is None/empty or cannot be parsed, returns `default`
      converted the same way (which itself may be Timestamp/str/None).
    - If both `value` and `default` are None, raises ValueError.
    - If `tz` is provided:
        * naive timestamps are localized to that timezone
        * tz-aware timestamps are converted to that timezone

    Parameters
    ----------
    value : pd.Timestamp | str | None
        The primary input to convert.
    default : pd.Timestamp | str | None, optional
        Fallback returned when `value` is missing or unparsable.
    tz : str, optional
        IANA timezone name (e.g., "America/New_York").

    Returns
    -------
    pd.Timestamp

    Raises
    -------
    ValueError is both value and default are None.
    """
    ts: Union[DateLike, None]

    def _coerce(x: Union[DateLike, None]) -> Optional[pd.Timestamp]:
        if x is None:
            return None
        if isinstance(x, pd.Timestamp):
            ts = x
        else:
            # Strip empty strings to treat them as missing
            if isinstance(x, str):
                x = x.strip()
                if not x:
                    return None
            ts = pd.to_datetime(x, errors="raise")

        if tz:
            if ts.tz is None:
                ts = ts.tz_localize(tz)
            else:
                ts = ts.tz_convert(tz)
        return ts

    # Try the primary value
    try:
        ts = _coerce(value)
        if ts is not None:
            return ts
    except Exception:  # pylint: disable=broad-exception-caught
        # fall through to default if parsing fails
        pass

    # Fallback to default
    if default is None:
        raise ValueError("Both primary value and default are None.")

    result = _coerce(default)
    if result is None:
        raise ValueError(f"Unable to convert using default value: '{default}'")

    return result


def to_tz_mixed(
    x: Union[DateLike, SeqLike],
    tz: Union[str, tzinfo] = DateAndTimeConstants.TZ_UTC.value,
    *,
    nonexistent: NonExistentTime = "shift_forward",  # DST spring-forward gaps
    ambiguous: Any = "NaT",  # DST fall-back duplicates
    errors: str = "coerce",  # default for pd.to_datetime
    **to_datetime_kwargs: Any,  # passed through to pd.to_datetime
) -> Union[pd.Timestamp, pd.Series, None]:
    """
    Convert scalars or sequences of date-like values to tz-aware pandas
    Timestamps using the modern ZoneInfo backend.

    Rules
    -----
    - Naïve -> tz_localize(tz, nonexistent=..., ambiguous=...)
    - tz-aware -> tz_convert(tz)
    - Extra kwargs go to pd.to_datetime (e.g., format=, dayfirst=, utc=,
      exact=).
      NOTE: passing utc=True makes naïve inputs UTC first, then they are
      converted.
    """
    tz_obj: tzinfo = tz if isinstance(tz, tzinfo) else ZoneInfo(str(tz))

    dt_kwargs = dict(to_datetime_kwargs)
    dt_kwargs.setdefault("errors", errors)

    def _one(v: DateLike) -> Optional[pd.Timestamp]:
        if v is None:
            return None

        ts = (
            v
            if isinstance(v, pd.Timestamp)
            else pd.to_datetime(v, **dt_kwargs)
        )
        if pd.isna(ts):
            return None
        if ts.tz is None:
            return ts.tz_localize(
                tz_obj, nonexistent=nonexistent, ambiguous=ambiguous
            )
        return ts.tz_convert(tz_obj)

    if isinstance(x, pd.Series):
        # logger.debug("to_tz_mixed: processing pd.Series input")
        return x.apply(_one)
    if isinstance(x, (list, tuple, pd.Index)):
        # logger.debug("to_tz_mixed: processing list/tuple/pd.Index input")
        return pd.Series(list(x)).apply(_one)
    # Only pass scalars (not sequences) to _one
    if isinstance(x, (pd.Timestamp, dt, str)) or x is None:
        # logger.debug("to_tz_mixed: processing scalar input")
        return _one(x)

    logger.error("to_tz_mixed: unsupported input type %s", type(x))
    raise TypeError(f"Unsupported input type for to_tz_mixed: {type(x)}")


def ensure_date_only(
    x: Union[DateLike, SeqLike],
    errors: Literal["coerce"] = "coerce",
    **to_pd_datetime_kwargs: Any,
) -> Union[pd.Timestamp, pd.Series, None]:
    """
    Ensure that the input is a date-like value (pd.Timestamp or str) and
    convert it to a date-only format (YYYY-MM-DD).

    Parameters
    ----------
    x : Union[DateLike, SeqLike]
        The input value(s) to convert.
    errors : Literal["coerce"], optional
        How to handle errors during conversion. Defaults to "coerce".
    **to_datetime_kwargs : Any
        Additional arguments passed to pd.to_datetime.

    Returns
    -------
    Union[pd.Timestamp, pd.Series, None]
        The converted date-only value(s), or None if conversion fails.
    """

    # Helper function to apply date-only conversion
    def _to_date_only(v: DateLike) -> Optional[pd.Timestamp]:
        if v is None:
            return None
        ts = pd.to_datetime(v, errors=errors, **to_pd_datetime_kwargs)
        if pd.isna(ts):
            return None
        return ts.normalize()

    if isinstance(x, pd.Series):
        # logger.debug("ensure_date_only: processing pd.Series input")
        return x.apply(_to_date_only)

    if isinstance(x, (list, tuple, pd.Index)):
        # logger.debug("ensure_date_only: processing list/tuple/pd.Index input")
        return pd.Series(list(x)).apply(_to_date_only)

    # Only pass scalars (not sequences) to _to_date_only
    if isinstance(x, (pd.Timestamp, dt, str)) or x is None:
        # logger.debug("ensure_date_only: processing scalar input")
        return _to_date_only(x)

    logger.error("ensure_date_only: unsupported input type %s", type(x))
    raise TypeError(f"Unsupported input type for ensure_date_only: {type(x)}")


def cagr(
    start_value: float,
    end_value: float,
    periods: float,
) -> float:
    """
    Calculate the Compound Annual Growth Rate (CAGR).

    Uses the formula to handle negative start values:
        numerator = end_value - start_value + abs(start_value)
        CAGR = (numerator / abs(start_value))^(1 / periods) - 1

    Parameters
    ----------
    start_value : float
        The initial value at the start of the period.
    end_value : float
        The final value at the end of the period.
    periods : float
        The number of periods (typically years).

    Returns
    -------
    float
        The CAGR as a decimal (e.g., 0.05 for 5%)

    Raises
    ------
    ValueError
        If `periods` <= 0.
    """
    if periods <= 0:
        raise ValueError("CAGR calculation requires periods > 0")

    numerator = end_value - start_value + abs(start_value)
    result = (numerator / abs(start_value)) ** (1 / periods) - 1
    return result


class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy data types."""

    def default(self, o):
        """Convert NumPy data types to native Python types for JSON
        serialization."""
        if isinstance(o, (np.integer,)):
            return int(o)
        elif isinstance(o, (np.floating,)):
            return float(o)
        elif isinstance(o, (np.ndarray,)):
            return o.tolist()
        return super(NpEncoder, self).default(o)


def aggregate_dividends(group):
    """
    A helper function to correctly aggregate dividend transactions for a single
    pay date. It sums the total value, gets the dividend-per-share value only
    from 'DIVIDEND' type transactions, and combines the dividend types.
    """
    # Filter for the cash dividend portion to get the correct per-share value
    dividend_rows = group[
        group["transaction_type"] == TransactionTypes.DIVIDEND.value
    ]

    if not dividend_rows.empty:
        # If dividend exists, use its per-share value.
        # .mean() handles cases where it might appear multiple times...
        # though it shouldn't.
        value_per_share = dividend_rows["value_per_share"].mean()
    else:
        # If only reinvestment, there is no 'dividend per share' to report.
        value_per_share = float("nan")

    # Combine all dividend types for that day (e.g., "DIVIDEND,DIVIDEND_REINVEST")
    if "div_type" in group.columns:
        div_types = ",".join(sorted(set(group["div_type"].dropna())))
    else:
        div_types = ""

    # Return a pandas Series. Pandas will assemble these into the new DataFrame.
    return pd.Series(
        {
            "total_value": group["total_value"].sum(),
            "value_per_share": value_per_share,
            "div_type": div_types,
        }
    )
