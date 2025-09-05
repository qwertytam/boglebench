"""
Collection of utility functions and classes for Boglebench

"""

from __future__ import annotations

import json
from datetime import datetime as dt
from datetime import timedelta, tzinfo
from typing import Any, Iterable, Literal, Optional, Union
from zoneinfo import ZoneInfo  # pylint: disable=wrong-import-order

import numpy as np
import pandas as pd
from pandas import Timedelta

from ..utils.logging_config import get_logger

# Custom type aliases
DateLike = Union[pd.Timestamp, dt, str, None]
SeqLike = Union[pd.Series, Iterable[DateLike], pd.Index]
NonExistentTime = Union[
    Literal["shift_forward", "shift_backward", "NaT", "raise"],
    Timedelta,
    timedelta,
]

logger = get_logger()


def ensure_timestamp(
    value: DateLike,
    default: DateLike = None,
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
    - If both `value` and `default` are unusable, returns None.
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
    Optional[pd.Timestamp]
    """

    def _coerce(x: DateLike) -> Optional[pd.Timestamp]:
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
    except Exception:
        # fall through to default if parsing fails
        pass

    # Fallback to default
    return _coerce(default)


def to_tz_mixed(
    x: Union[DateLike, SeqLike],
    tz: Union[str, tzinfo] = "America/New_York",
    *,
    nonexistent: NonExistentTime = "shift_forward",  # DST spring-forward gaps
    ambiguous: Any = "NaT",  # DST fall-back duplicates
    errors: str = "coerce",  # default for pd.to_datetime
    **to_datetime_kwargs: Any,  # passed through to pd.to_datetime
):
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
            v if isinstance(v, pd.Timestamp) else pd.to_datetime(v, **dt_kwargs)
        )
        if pd.isna(ts):
            return None
        if ts.tz is None:
            return ts.tz_localize(
                tz_obj, nonexistent=nonexistent, ambiguous=ambiguous
            )
        return ts.tz_convert(tz_obj)

    if isinstance(x, pd.Series):
        logger.debug("to_tz_mixed: processing pd.Series input")
        return x.apply(_one)
    if isinstance(x, (list, tuple, pd.Index)):
        logger.debug("to_tz_mixed: processing list/tuple/pd.Index input")
        return pd.Series(list(x)).apply(_one)
    # Only pass scalars (not sequences) to _one
    if isinstance(x, (pd.Timestamp, dt, str)) or x is None:
        logger.debug("to_tz_mixed: processing scalar input")
        return _one(x)

    logger.error("to_tz_mixed: unsupported input type %s", type(x))
    raise TypeError(f"Unsupported input type for to_tz_mixed: {type(x)}")


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
