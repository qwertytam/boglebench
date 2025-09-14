"""
Custom type aliases used across the BogleBench project.
"""

from datetime import datetime as dt
from datetime import timedelta
from typing import Iterable, Literal, TypeAlias, Union

import pandas as pd
from pandas import Timedelta

DateLike: TypeAlias = Union[pd.Timestamp, dt, str]
SeqLike: TypeAlias = Union[pd.Series, Iterable[DateLike], pd.Index]
NonExistentTime: TypeAlias = Union[
    Literal["shift_forward", "shift_backward", "NaT", "raise"],
    Timedelta,
    timedelta,
]
