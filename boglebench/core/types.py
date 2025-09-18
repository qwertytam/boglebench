"""
Custom type aliases used across the BogleBench project.
"""

from datetime import date, datetime, timedelta
from typing import Iterable, Literal, TypeAlias, Union

import numpy as np
import pandas as pd
from pandas import Timedelta

DateLike: TypeAlias = Union[
    pd.Timestamp, datetime, date, np.datetime64, int, float, str
]
SeqLike: TypeAlias = Union[pd.Series, Iterable[DateLike], pd.Index]
NonExistentTime: TypeAlias = Union[
    Literal["shift_forward", "shift_backward", "NaT", "raise"],
    Timedelta,
    timedelta,
]
