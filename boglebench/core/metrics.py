"""
Module for calculating portfolio performance metrics.
"""

from typing import Dict, Optional

import numpy as np
import numpy_financial as npf  # type: ignore
import pandas as pd

from ..utils.config import ConfigManager
from ..utils.logging_config import get_logger
from ..utils.tools import cagr
from .constants import ConversionFactors, DateAndTimeConstants, Defaults

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


def calculate_metrics(
    returns: pd.Series,
    name: str,
    annual_trading_days: int = DateAndTimeConstants.DAYS_IN_TRADING_YEAR.value,
    annual_risk_free_rate: float = Defaults.DEFAULT_RISK_FREE_RATE,
) -> Dict:
    """Calculate performance metrics for a return series."""
    if returns.empty or returns.isna().all():
        return {}

    # Basic statistics
    total_periods = len(returns)
    year_fraction = total_periods / annual_trading_days

    # Ensure returns are numeric to avoid type errors
    logger.debug(
        "Returns series %s before numeric conversion:\n%s",
        name,
        returns.head(10),
    )
    returns = pd.to_numeric(returns, errors="coerce")
    if returns.isna().any():
        logger.warning(
            "⚠️  Non-numeric returns found in series '%s'. These are set to zero:\n%s",
            name,
            returns[returns.isna()],
        )
        returns = returns.fillna(0)

    logger.debug(
        "%s: First 10 daily returns:\n%s",
        name,
        returns.head(n=10) * ConversionFactors.DECIMAL_TO_PERCENT,
    )

    prod_result = (1 + returns).prod()
    if isinstance(prod_result, (int, float, np.number)):
        total_return = float(prod_result) - 1.0
    else:
        logger.error(
            "%s: Unexpected product result type: %s", name, type(prod_result)
        )
        return {}

    annualized_return = cagr(1, 1 + total_return, year_fraction)
    if isinstance(annualized_return, complex):
        logger.error(
            "%s: Annualized return calculation resulted in complex number: %s",
            name,
            annualized_return,
        )
        annualized_return = Defaults.ZERO_RETURN

    logger.debug(
        "%s: Periods: %d, Years: %.2f",
        name,
        total_periods,
        year_fraction,
    )

    logger.debug(
        "%s: First date: %s, Last date: %s",
        name,
        returns.index[0] if len(returns) > 0 else "N/A",
        returns.index[-1] if len(returns) > 0 else "N/A",
    )

    logger.debug(
        "%s: Total Return: %.2f%%, Annualized Return: %.2f%%",
        name,
        total_return * ConversionFactors.DECIMAL_TO_PERCENT,
        annualized_return * ConversionFactors.DECIMAL_TO_PERCENT,
    )

    volatility = returns.std(ddof=1)  # Daily volatility, sample stddev
    annual_volatility = volatility * np.sqrt(
        annual_trading_days
    )  # Annualized volatility

    logger.debug(
        "%s: Volatility: %.2f%% (period) %.2f%% (annualized)",
        name,
        volatility * ConversionFactors.DECIMAL_TO_PERCENT,
        annual_volatility * ConversionFactors.DECIMAL_TO_PERCENT,
    )

    # Risk-adjusted metrics
    daily_risk_free_rate = cagr(
        1, 1 + annual_risk_free_rate, annual_trading_days
    )
    excess_returns = returns - daily_risk_free_rate
    excess_mean_returns = excess_returns.mean()
    sharpe_ratio = (
        excess_mean_returns / volatility * np.sqrt(annual_trading_days)
        if returns.std() > 0
        else 0
    )

    # Drawdown analysis
    cumulative_returns: np.number = (1 + returns).cumprod()
    if not isinstance(cumulative_returns, (int, float, np.number, pd.Series)):
        logger.error(
            "%s: Unexpected cumulative product result type: %s",
            name,
            type(cumulative_returns),
        )
        return {}

    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    # Additional metrics
    positive_periods = (returns > 0).sum()
    win_rate = positive_periods / total_periods if total_periods > 0 else 0

    return {
        "name": name,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "total_periods": total_periods,
        "start_date": returns.index[0] if len(returns) > 0 else None,
        "end_date": returns.index[-1] if len(returns) > 0 else None,
    }


def calculate_relative_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series],
    annual_trading_days: int = DateAndTimeConstants.DAYS_IN_TRADING_YEAR.value,
    annual_risk_free_rate: float = Defaults.DEFAULT_RISK_FREE_RATE,
) -> Dict:
    """Calculate relative performance metrics vs benchmark."""
    if benchmark_returns is None or portfolio_returns.empty:
        logger.warning("⚠️  No benchmark data available for relative metrics.")
        return {}

    # Align the series by index. This is crucial for non-continuous date
    # ranges.
    logger.debug("Aligning portfolio and benchmark returns...")
    logger.debug("Portfolio returns:\n%s", portfolio_returns.head(10))
    logger.debug("Benchmark returns:\n%s", benchmark_returns.head(10))

    aligned_portfolio, aligned_benchmark = portfolio_returns.align(
        benchmark_returns, join="inner"
    )

    if aligned_portfolio.empty:
        logger.warning(
            "⚠️  No overlapping dates between portfolio and benchmark returns."
        )
        return {}

    # Calculate excess returns (alpha)
    excess_returns = aligned_portfolio - aligned_benchmark

    # Tracking error (volatility of excess returns)
    tracking_error = excess_returns.std() * np.sqrt(annual_trading_days)

    # Information ratio (excess return / tracking error)
    information_ratio = (
        (excess_returns.mean() * annual_trading_days) / tracking_error
        if tracking_error > 0
        else 0
    )

    # Beta calculation
    covariance = np.cov(aligned_portfolio, aligned_benchmark, ddof=1)[0, 1]
    benchmark_variance = np.var(aligned_benchmark, ddof=1)
    beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

    # Jensen's Alpha (risk-adjusted excess return
    daily_risk_free_rate = cagr(
        1, 1 + annual_risk_free_rate, annual_trading_days
    )
    portfolio_excess = aligned_portfolio.mean() - daily_risk_free_rate
    benchmark_excess = aligned_benchmark.mean() - daily_risk_free_rate
    jensens_alpha = portfolio_excess - (beta * benchmark_excess)
    jensens_alpha_annualized = jensens_alpha * annual_trading_days

    return {
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "beta": beta,
        "jensens_alpha": jensens_alpha_annualized,
        "correlation": np.corrcoef(aligned_portfolio, aligned_benchmark)[0, 1],
    }
