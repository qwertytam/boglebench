"""
Collection of utility functions and classes for Boglebench

"""

from __future__ import annotations

from datetime import timedelta, tzinfo
from typing import Any, Iterable, Literal, Optional, Union

import pandas as pd
from pandas import Timedelta
from zoneinfo import ZoneInfo

DateLike = Union[pd.Timestamp, str, None]
SeqLike = Union[pd.Series, Iterable[DateLike], pd.Index]
NonExistentTime = Union[
    Literal["shift_forward", "shift_backward", "NaT", "raise"],
    Timedelta,
    timedelta,
]


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
        return x.apply(_one)
    if isinstance(x, (list, tuple, pd.Index)):
        return pd.Series(list(x)).apply(_one)
    # Only pass scalars (not sequences) to _one
    if isinstance(x, (pd.Timestamp, str)) or x is None:
        return _one(x)
    raise TypeError(f"Unsupported input type for to_tz_mixed: {type(x)}")
