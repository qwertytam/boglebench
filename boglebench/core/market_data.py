"""
Provides market data for BogleBench, handling API calls and caching.
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from alpha_vantage.timeseries import TimeSeries  # type: ignore

from ..core.constants import DateAndTimeConstants, Defaults
from ..utils.logging_config import get_logger

logger = get_logger()


class MarketDataProvider:
    """
    Handles fetching and caching of market data from Alpha Vantage.
    """

    def __init__(
        self,
        cache_dir: Path,
        api_key: str = Defaults.DEFAULT_API_KEY,
        cache_enabled: bool = True,
        force_cache_refresh: bool = False,
    ):
        """
        Initializes the MarketDataProvider.

        Args:
            api_key: The Alpha Vantage API key.
            cache_dir: The directory to use for caching market data.
        """
        self._validate_api_key(api_key)
        self.api_key = api_key
        self.ts = TimeSeries(key=self.api_key, output_format="pandas")
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_enabled = cache_enabled
        self.force_cache_refresh = force_cache_refresh

    @staticmethod
    def _validate_api_key(api_key: Optional[str]):
        """Check if the Alpha Vantage API key is provided."""
        if not api_key or api_key == Defaults.DEFAULT_API_KEY:
            msg = "Alpha Vantage API key is not set. Please set it in your config."
            logger.error(msg)
            raise ValueError(msg)

    def _get_cached_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Attempt to retrieve market data from the cache."""
        cache_file = self.cache_dir / f"{ticker}.parquet"
        if not cache_file.exists():
            logger.debug("No cache file for %s", ticker)
            return None

        try:
            cached_df = pd.read_parquet(cache_file)

            logger.debug("✅ Loaded market data from cache for %s.", ticker)
            return cached_df

        except (pd.errors.EmptyDataError, ValueError) as e:
            logger.warning("⚠️ Could not read cache file %s: %s", cache_file, e)
            return None

    def _cache_data(self, data: pd.DataFrame, ticker: str):
        """Cache the given market data DataFrame."""
        cache_file = self.cache_dir / f"{ticker}.parquet"
        try:
            data.to_parquet(cache_file, index=False)
            logger.info("💾 Saved market data to cache: %s", cache_file)
        except IOError as e:
            logger.error(
                "❌ Failed to write to cache file %s: %s", cache_file, e
            )

    def get_market_data(
        self,
        tickers: List[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch market data for a list of tickers, using cache if available.

        Args:
            tickers: A list of stock tickers.
            start_date: The start date for the data in 'YYYY-MM-DD' format.
            end_date: The end date for the data in 'YYYY-MM-DD' format.

        Returns:
            A dictionary mapping each ticker to its market data DataFrame.
        """
        logger.info(
            "⬇️  Fetching market data for %d unique tickers: %s",
            len(tickers),
            ", ".join(tickers),
        )

        all_data = {}
        failed_tickers = []

        s_date = start_date.date()
        e_date = end_date.date()

        for ticker in tickers:
            if self.cache_enabled and not self.force_cache_refresh:
                logger.info("🔍 Checking cache for %s...", ticker)
                cached_data = self._get_cached_data(ticker)

                if cached_data is not None:
                    logger.info(
                        "✅ Loaded market data from cache for %s.", ticker
                    )
                    cache_start_date = pd.to_datetime(
                        cached_data["date"].min()
                    ).date()
                    cache_end_date = pd.to_datetime(
                        cached_data["date"].max()
                    ).date()
                    if cache_start_date <= s_date and cache_end_date >= e_date:
                        logger.debug(
                            "✅ Using cached data for %s from %s to %s.",
                            ticker,
                            cache_start_date,
                            cache_end_date,
                        )

                        all_data[ticker] = cached_data[
                            [
                                "date",
                                "close",
                                "adj_close",
                                "dividend",
                                "split_coefficient",
                            ]
                        ]

                        continue

                    else:
                        logger.info(
                            "⚠️ Cached data for %s is out of range (%s to %s). "
                            "Fetching fresh data.",
                            ticker,
                            cache_start_date,
                            cache_end_date,
                        )

            try:
                logger.info(
                    "⏳ Cache miss - downloading market data for %s...", ticker
                )
                # pylint: disable-next=unbalanced-tuple-unpacking
                data, _ = self.ts.get_daily_adjusted(  # type: ignore
                    symbol=ticker, outputsize="full"
                )
                # Alpha Vantage column names:
                # '1. open', '2. high', '3. low', '4. close',
                # '5. adjusted close', '6. volume', '7. dividend amount',
                # '8. split coefficient'

                # pylint: disable-next=no-member
                data = data.rename(  # type: ignore
                    columns={
                        "1. open": "open",
                        "2. high": "high",
                        "3. low": "low",
                        "4. close": "close",
                        "5. adjusted close": "adj_close",
                        "6. volume": "volume",
                        "7. dividend amount": "dividend",
                        "8. split coefficient": "split_coefficient",
                    }
                )

                keep_cols = [
                    "close",
                    "adj_close",
                    "dividend",
                    "split_coefficient",
                ]

                data = data[keep_cols]
                data.index = pd.to_datetime(data.index, utc=True)
                data.sort_index(inplace=True)
                data.reset_index(inplace=True, names="date")

                try:
                    # logger.debug("data column dtypes:\n%s", data.dtypes)
                    date_tz = data["date"].dt.tz
                    # logger.debug(
                    #     "Market data 'date' column tz info: %s", date_tz
                    # )
                except TypeError as e:
                    logger.debug(
                        "Market data 'date' column tz info could not be determined: %s",
                        e,
                    )
                    date_tz = None

                if date_tz is None:
                    data["date"] = data["date"].dt.tz_localize(
                        DateAndTimeConstants.TZ_UTC.value
                    )
                    logger.debug(
                        "Localized 'date' column to %s.", data["date"].dt.tz
                    )

                # Filter by date range
                data_start_mask = data["date"] >= pd.to_datetime(
                    s_date
                ).tz_localize(DateAndTimeConstants.TZ_UTC.value)
                if not data_start_mask.any():
                    msg = f"No market data available on or after start date: {s_date}"
                    logger.error(msg)
                    raise ValueError(msg)

                data_end_mask = data["date"] <= pd.to_datetime(
                    e_date
                ).tz_localize(DateAndTimeConstants.TZ_UTC.value)
                if not data_end_mask.any():
                    msg = f"No market data available on or before end date: {e_date}"
                    logger.error(msg)
                    raise ValueError(msg)

                data = data[data_start_mask & data_end_mask]

                logger.info(
                    "✅ Downloaded market data for %s: %d rows from %s to %s.",
                    ticker,
                    len(data),
                    data["date"].dt.date.min(),
                    data["date"].dt.date.max(),
                )

                self._cache_data(data, ticker)

                all_data[ticker] = data

            except (OSError, ValueError) as e:
                logger.error("  ❌ Failed to download %s: %s", ticker, e)
                failed_tickers.append(ticker)

        if failed_tickers:
            logger.error("⚠️  Failed to download data for: %s", failed_tickers)

        if not all_data:
            return {}

        # Return data in the same format as the cached version
        return all_data
