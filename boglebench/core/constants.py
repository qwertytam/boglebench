"""Core constants used throughout the BogleBench library."""

from enum import Enum


class DatesAndTimeFormats(str, Enum):
    """Enumeration for date and time formats used in BogleBench."""

    ISO8601 = "%Y-%m-%d"  # Standard date format (e.g., 2023-01-15)
    ISO8601_FULL = "%Y-%m-%dT%H:%M:%S%z"  # Full datetime with timezone
    MONTHLY = "%Y-%m"  # Monthly format (e.g., 2023-01)
    YEARLY = "%Y"  # Yearly format (e.g., 2023)
    TIME_24HR = "%H:%M:%S"  # 24-hour time format (e.g., 14:30:00)


class DateAndTimeConstants(str, Enum):
    """Enumeration for date and time constants used in BogleBench."""

    TZ_UTC = "UTC"  # Coordinated Universal Time
    TZ_NY = "America/New_York"  # New York timezone
    TZ_LONDON = "Europe/London"  # London timezone
    TZ_TOKYO = "Asia/Tokyo"  # Tokyo timezone
    TZ_DEFAULT = TZ_UTC  # Default timezone for the library

    DAYS_IN_CALENDAR_YEAR = 365  # Days in a year
    DAYS_IN_CALENDAR_LEAP_YEAR = 366  # Days in a leap year

    DAYS_IN_TRADING_YEAR = 252  # Typical trading days in a year
    TRADING_DAYS_IN_MONTH = 21  # Approximate trading days in a month
    TRADING_DAYS_IN_WEEK = 5  # Trading days in a week

    MONTHS_IN_YEAR = 12  # Months in a year
    WEEKS_IN_YEAR = 52  # Weeks in a year
    HOURS_IN_DAY = 24  # Hours in a day
    MINUTES_IN_HOUR = 60  # Minutes in an hour
    SECONDS_IN_MINUTE = 60  # Seconds in a minute
    SECONDS_IN_HOUR = MINUTES_IN_HOUR * SECONDS_IN_MINUTE  # Seconds in an hour
    SECONDS_IN_DAY = SECONDS_IN_HOUR * HOURS_IN_DAY  # Seconds in a day


class ConversionFactors(float, Enum):
    """Enumeration for common conversion factors used in financial calculations."""

    PERCENT_TO_DECIMAL = 0.01
    DECIMAL_TO_PERCENT = 100.0


class Defaults:
    """Default values used throughout the BogleBench library."""

    DEFAULT_VALUE = float(0.0)
    DEFAULT_ZERO = float(0.0)

    ZERO_ASSET_VALUE = float(0.0)
    ZERO_CASH_FLOW = float(0.0)
    ZERO_DIVIDEND = float(0.0)
    ZERO_PRICE = float(0.0)
    ZERO_RETURN = float(0.0)

    DEFAULT_CASH_FLOW_WEIGHT = float(0.5)
    DEFAULT_RISK_FREE_RATE = float(0.02)  # 2% risk-free rate

    DEFAULT_LOOK_FORWARD_PRICE_DATA = 10  # days

    DEFAULT_API_KEY = "YOUR_API_KEY"
    DEFAULT_CACHE_DIR = "market_data/"


class TransactionTypes(str, Enum):
    """Enumeration for transaction types."""

    BUY = "BUY"
    SELL = "SELL"
    DIVIDEND = "DIVIDEND"
    DIVIDEND_REINVEST = "DIVIDEND_REINVEST"
    FEE = "FEE"
    SHARE_TRANSFER_IN = "SHARE_TRANSFER_IN"
    SHARE_TRANSFER_OUT = "SHARE_TRANSFER_OUT"
    SPLIT = "SPLIT"
    OTHER = "OTHER"

    @classmethod
    def all_types(cls) -> list[str]:
        """Return a list of all transaction types."""
        return [item.value for item in cls]

    @classmethod
    def all_dividend_types(cls) -> list[str]:
        """Return a list of all dividend-related transaction types."""
        return [cls.DIVIDEND, cls.DIVIDEND_REINVEST]

    @classmethod
    def is_valid(cls, tx_type: str) -> bool:
        """Check if a transaction type is valid."""
        return tx_type in cls.all_types()

    @classmethod
    def is_buy_or_sell(cls, tx_type: str) -> bool:
        """Check if a transaction type is BUY or SELL."""
        return tx_type in (cls.BUY, cls.SELL)

    @classmethod
    def is_dividend(cls, tx_type: str) -> bool:
        """Check if a transaction type is DIVIDEND or DIVIDEND_REINVEST."""
        return tx_type in (cls.DIVIDEND, cls.DIVIDEND_REINVEST)

    @classmethod
    def is_dividend_reinvest(cls, tx_type: str) -> bool:
        """Check if a transaction type is DIVIDEND_REINVEST."""
        return tx_type == cls.DIVIDEND_REINVEST

    @classmethod
    def is_quantity_changing(cls, tx_type: str) -> bool:
        """Check if a transaction type changes quantity (BUY, SELL,
        SHARE_TRANSFER_IN, SHARE_TRANSFER_OUT)."""
        return tx_type in (
            cls.BUY,
            cls.SELL,
            cls.DIVIDEND_REINVEST,
            cls.SHARE_TRANSFER_IN,
            cls.SHARE_TRANSFER_OUT,
        )

    @classmethod
    def is_fee(cls, tx_type: str) -> bool:
        """Check if a transaction type is FEE."""
        return tx_type == cls.FEE

    @classmethod
    def is_share_transfer(cls, tx_type: str) -> bool:
        """Check if a transaction type is SHARE_TRANSFER_IN or
        SHARE_TRANSFER_OUT."""
        return tx_type in (cls.SHARE_TRANSFER_IN, cls.SHARE_TRANSFER_OUT)

    @classmethod
    def is_split(cls, tx_type: str) -> bool:
        """Check if a transaction type is SPLIT."""
        return tx_type == cls.SPLIT

    @classmethod
    def is_other(cls, tx_type: str) -> bool:
        """Check if a transaction type is OTHER."""
        return tx_type == cls.OTHER


class DividendTypes(str, Enum):
    """Enumeration for dividend types."""

    CASH = "CASH"
    STOCK = "STOCK"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"
    DEFAULT = CASH

    @classmethod
    def all_types(cls) -> list[str]:
        """Return a list of all dividend types."""
        return [item.value for item in cls]

    @classmethod
    def is_valid(cls, div_type: str) -> bool:
        """Check if a dividend type is valid."""
        return div_type in cls.all_types()


class FileExtensions(str, Enum):
    """Enumeration for common file extensions used in BogleBench."""

    CSV = ".csv"
    JSON = ".json"
    XLSX = ".xlsx"
    PARQUET = ".parquet"
    PICKLE = ".pkl"
    TXT = ".txt"
    YAML = ".yaml"
    YML = ".yml"

    @classmethod
    def all_extensions(cls) -> list[str]:
        """Return a list of all file extensions."""
        return [item.value for item in cls]
