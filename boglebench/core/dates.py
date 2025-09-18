"""
Manages the start and end dates for the analysis period.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal  # type: ignore

from ..core.constants import DateAndTimeConstants, Defaults
from ..utils.logging_config import get_logger
from ..utils.tools import to_tzts_scaler

logger = get_logger()


class AnalysisPeriod:
    """Handles the logic for determining the analysis start and end dates."""

    def __init__(self, config, transactions_df=None):
        self.config = config
        self.transactions_df = transactions_df
        self.start_date = self._determine_start_date()
        self.end_date = self._determine_end_date()

        if (
            self.start_date
            and self.end_date
            and self.start_date > self.end_date
        ):
            raise ValueError(
                f"Start date ({self.start_date.date()}) "
                f"cannot be after end date ({self.end_date.date()})."
            )

    def _determine_start_date(self):
        """Uses config or falls back to the first transaction date."""
        start_date_val = self.config.get("analysis.start_date")
        if start_date_val:
            logger.info("Using start date from config: %s", start_date_val)
            return to_tzts_scaler(
                start_date_val, tz=DateAndTimeConstants.TZ_UTC.value
            )

        if self.transactions_df is not None and not self.transactions_df.empty:
            min_date = self.transactions_df["date"].min()
            logger.info(
                "Using first transaction date as start date: %s", min_date
            )
            return min_date

        return None

    def _determine_end_date(self):
        """Uses config, or falls back to the last closed market day or now."""
        end_date_val = self.config.get("analysis.end_date")
        if end_date_val:
            logger.info("Using end date from config: %s", end_date_val)
            return to_tzts_scaler(
                end_date_val, tz=DateAndTimeConstants.TZ_UTC.value
            )

        if self._is_market_currently_open():
            return self._get_last_closed_market_day()

        logger.info("Market is closed, using current time as end date.")
        return to_tzts_scaler(
            datetime.now(ZoneInfo(DateAndTimeConstants.TZ_UTC.value))
        )

    def _is_market_currently_open(self) -> bool:
        """Check if the market is currently open."""
        nyse = mcal.get_calendar("NYSE")
        now = datetime.now(tz=ZoneInfo(DateAndTimeConstants.TZ_UTC.value))
        schedule = nyse.schedule(start_date=now.date(), end_date=now.date())

        if schedule.empty:
            return False  # Market is closed today (holiday or weekend)

        market_open = schedule.iloc[0]["market_open"]
        market_close = schedule.iloc[0]["market_close"]
        return market_open <= now <= market_close

    def _get_last_closed_market_day(self) -> pd.Timestamp:
        """Get the most recent market day that has closed."""
        nyse = mcal.get_calendar("NYSE")
        # Use a timezone-aware 'today' for comparison
        today = datetime.now(tz=ZoneInfo("America/New_York"))
        schedule = nyse.schedule(
            start_date=today
            - timedelta(days=Defaults.DEFAULT_LOOK_FORWARD_PRICE_DATA),
            end_date=today,
        )

        if schedule.empty:
            raise ValueError("No recent market days found.")

        # Find the last day where the market close time is before the current
        # time
        closed_days = schedule[schedule["market_close"] < today]
        last_closed_day = closed_days["market_close"].max()

        logger.info("Last closed market day is %s", last_closed_day)
        return pd.to_datetime(last_closed_day, utc=True)
