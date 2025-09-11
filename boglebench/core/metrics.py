"""
Module for calculating portfolio performance metrics.
"""

import numpy as np
import numpy_financial as npf  # type: ignore
import pandas as pd

from ..utils.config import ConfigManager
from ..utils.logging_config import get_logger
from .constants import DateAndTimeConstants, Defaults

logger = get_logger()


def calculate_modified_dietz_returns(portfolio_df: pd.DataFrame) -> pd.Series:
    """Calculate portfolio returns using Modified Dietz method."""
    returns = []

    for i in range(len(portfolio_df)):
        if i == 0:
            # First day - beginning_value is zero
            beginning_value = Defaults.DEFAULT_ZERO
        else:
            beginning_value = portfolio_df.iloc[i - 1]["total_value"]

        # Dividends are considered internal cash flows and do not directly
        # affect the Modified Dietz calculation. Instead, the method
        # focuses on external cash flows, which are movements of value
        # into or out of the portfolio that are not related to investment
        # income.
        # Value change = total change minus external cash flow impact
        # Get values
        ending_value = portfolio_df.iloc[i]["total_value"]
        external_cash_flow = (
            portfolio_df.iloc[i]["investment_cash_flow"]
            + portfolio_df.iloc[i]["income_cash_flow"]
        )
        weighted_cash_flow = portfolio_df.iloc[i]["weighted_cash_flow"]

        # Modified Dietz formula
        denominator = beginning_value + weighted_cash_flow

        if denominator <= 0:
            # Handle edge case: no beginning value or negative denominator
            returns.append(Defaults.ZERO_RETURN)
        else:
            daily_return = (
                ending_value - beginning_value - external_cash_flow
            ) / denominator
            returns.append(daily_return)

    return pd.Series(returns)


def calculate_account_modified_dietz_returns(
    portfolio_df: pd.DataFrame, account: str
) -> pd.Series:
    """Calculate account-level returns using Modified Dietz method."""
    returns = []
    account_total_col = f"{account}_total"
    account_cash_flow_col = f"{account}_cash_flow"
    account_weighted_cash_flow_col = f"{account}_weighted_cash_flow"

    for i in range(len(portfolio_df)):
        if i == 0:
            returns.append(Defaults.ZERO_RETURN)
            continue

        beginning_value = portfolio_df.iloc[i - 1][account_total_col]
        ending_value = portfolio_df.iloc[i][account_total_col]
        net_cash_flow = portfolio_df.iloc[i][account_cash_flow_col]
        weighted_cash_flow = portfolio_df.iloc[i][
            account_weighted_cash_flow_col
        ]

        denominator = beginning_value + weighted_cash_flow

        if denominator <= 0:
            returns.append(Defaults.ZERO_RETURN)
        else:
            daily_return = (
                ending_value - beginning_value - net_cash_flow
            ) / denominator
            returns.append(daily_return)

    return pd.Series(returns)


def calculate_twr_daily_returns(portfolio_df: pd.DataFrame) -> pd.Series:
    """
    Calculate daily portfolio returns using the Time-Weighted Return (TWR)
    method. This method removes the effects of cash flows to measure the
    performance of the underlying assets.
    """
    returns = []
    for i in range(len(portfolio_df)):
        if i == 0:
            # No return can be calculated on the first day.
            returns.append(Defaults.ZERO_RETURN)
            continue

        beginning_value = portfolio_df.iloc[i - 1]["total_value"]
        ending_value = portfolio_df.iloc[i]["total_value"]
        net_cash_flow = portfolio_df.iloc[i]["net_cash_flow"]

        if beginning_value <= 0:
            # If the starting value is zero or negative, the return for the
            # period is considered zero as there's no initial investment
            # base to measure performance against.
            returns.append(Defaults.ZERO_RETURN)
        else:
            # The TWR formula for a single period (in this case, one day).
            # We are using the assumption that cash flows occur just before
            # the ending value is measured.
            # TWR = (Ending Value - Net Cash Flow) / Beginning Value - 1
            daily_return = (ending_value - net_cash_flow) / beginning_value - 1
            returns.append(daily_return)

    return pd.Series(returns)


def calculate_account_twr_daily_returns(
    portfolio_df: pd.DataFrame, account: str
) -> pd.Series:
    """Calculate account-level returns using Time-Weighted Return method."""
    returns = []
    account_total_col = f"{account}_total"
    account_cash_flow_col = f"{account}_cash_flow"

    for i in range(len(portfolio_df)):
        if i == 0:
            returns.append(Defaults.ZERO_RETURN)
            continue

        beginning_value = portfolio_df.iloc[i - 1][account_total_col]
        ending_value = portfolio_df.iloc[i][account_total_col]
        net_cash_flow = portfolio_df.iloc[i][account_cash_flow_col]

        if beginning_value <= 0:
            returns.append(Defaults.ZERO_RETURN)
        else:
            daily_return = (ending_value - net_cash_flow) / beginning_value - 1
            returns.append(daily_return)

    return pd.Series(returns)


def calculate_irr(
    portfolio_history: pd.DataFrame, config: ConfigManager
) -> float:
    """
    Calculate the Internal Rate of Return (IRR) for the entire portfolio period.
    """
    if portfolio_history is None or portfolio_history.empty:
        return Defaults.ZERO_RETURN

    # IRR calculation requires a series of cash flows over time.
    # The convention for numpy.irr is:
    # - Investments (cash inflows to the portfolio) are negative.
    # - Withdrawals (cash outflows from the portfolio) are positive.
    # - The final value is the terminal market value of the portfolio.

    # Your 'net_cash_flow' is positive for BUYs and negative for SELLs.
    # We need to invert this for the IRR calculation.
    cash_flows = np.array(portfolio_history["net_cash_flow"].values) * -1

    # The first cash flow is the initial investment at time 0.
    # We replace the first day's cash flow with the starting value.
    # cash_flows[0] = -portfolio_history["total_value"].iloc[0]

    # The last entry in the series must be include the final market value
    # of the portfolio >> so the net last cash flow is selling the entire
    # portfolio less any other cash flow actions on that day.
    final_value = portfolio_history["total_value"].iloc[-1]

    all_flows = cash_flows.copy()
    all_flows[-1] += final_value  # Add final value to last cash flow

    try:
        # numpy.irr calculates the IRR for the given period (daily in this case).
        daily_irr = npf.irr(all_flows)
        if np.isnan(daily_irr) or np.isinf(daily_irr):
            return Defaults.ZERO_RETURN

        # Annualize the daily IRR.
        annual_trading_days = config.get(
            "settings.annual_trading_days",
            DateAndTimeConstants.DAYS_IN_TRADING_YEAR,
        )

        if annual_trading_days is None:
            annual_trading_days = DateAndTimeConstants.DAYS_IN_TRADING_YEAR
        elif isinstance(annual_trading_days, dict):
            annual_trading_days = annual_trading_days.get(
                "value", DateAndTimeConstants.DAYS_IN_TRADING_YEAR
            )

        annual_trading_days = int(annual_trading_days)
        annualized_irr = (1 + daily_irr) ** annual_trading_days - 1
        return annualized_irr

    except ValueError:
        # numpy.irr raises a ValueError if it cannot find a solution.
        if logger is not None:
            logger.warning(
                "⚠️  Internal Rate of Return (IRR) calculation did not converge. "
                "This can happen with unconventional cash flow patterns."
            )
        return Defaults.ZERO_RETURN
