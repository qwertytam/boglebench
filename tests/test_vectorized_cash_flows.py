"""
Tests for vectorized cash flow computation optimization.

Verifies that the vectorized _compute_cash_flows_vectorized method
produces the same results as the original _process_daily_transactions approach.
"""

import pandas as pd
import pytest

from boglebench.core.constants import TransactionTypes
from boglebench.core.history_builder import PortfolioHistoryBuilder
from boglebench.utils.config import ConfigManager


@pytest.fixture
def sample_transactions():
    """Create sample transaction data."""
    data = {
        "date": pd.to_datetime(
            [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
                "2024-01-05",
            ]
        ),
        "account": ["Brokerage", "Brokerage", "IRA", "Brokerage", "IRA"],
        "symbol": ["SPY", "VTI", "SPY", "VTI", "SPY"],
        "transaction_type": [
            TransactionTypes.BUY.value,
            TransactionTypes.BUY.value,
            TransactionTypes.SELL.value,
            TransactionTypes.DIVIDEND.value,
            TransactionTypes.BUY.value,
        ],
        "quantity": [10, 20, -5, 0, 15],
        "price": [400, 200, 410, 0, 405],
        "total_value": [-4000, -4000, 2050, 50, -6075],
        "fees": [0, 0, 0, 0, 0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_market_data():
    """Create sample market data."""
    dates = pd.date_range("2024-01-01", "2024-01-10", freq="D")

    spy_data = pd.DataFrame(
        {
            "date": dates,
            "close": [400, 402, 405, 408, 410, 412, 415, 418, 420, 422],
            "adjusted_close": [
                400,
                402,
                405,
                408,
                410,
                412,
                415,
                418,
                420,
                422,
            ],
        }
    )

    vti_data = pd.DataFrame(
        {
            "date": dates,
            "close": [200, 201, 202, 203, 204, 205, 206, 207, 208, 209],
            "adjusted_close": [
                200,
                201,
                202,
                203,
                204,
                205,
                206,
                207,
                208,
                209,
            ],
        }
    )

    return {"SPY": spy_data, "VTI": vti_data}


@pytest.fixture
def config():
    """Create test config."""
    # Use minimal config
    config_dict = {
        "settings": {
            "annual_trading_days": 252,
            "annual_risk_free_rate": 0.03,
        }
    }
    config = ConfigManager()  # pylint: disable=redefined-outer-name
    config.config = config_dict
    return config


# pylint: disable=redefined-outer-name
def test_vectorized_cash_flows_basic(
    sample_transactions, sample_market_data, config
):
    """Test that vectorized cash flow computation produces correct results."""

    builder = PortfolioHistoryBuilder(
        config=config,
        transactions=sample_transactions,
        market_data=sample_market_data,
        start_date=pd.Timestamp("2024-01-01"),
        end_date=pd.Timestamp("2024-01-10"),
    )

    # Create a simple portfolio dataframe with dates
    dates = pd.date_range("2024-01-01", "2024-01-10", freq="D")
    portfolio_df = pd.DataFrame({"date": dates})

    # Test the vectorized method
    # pylint: disable=protected-access
    result_df = builder._compute_cash_flows_vectorized(portfolio_df)

    # Verify columns exist
    assert "investment_cash_flow" in result_df.columns
    assert "income_cash_flow" in result_df.columns
    assert "net_cash_flow" in result_df.columns

    # Verify account cash flows
    assert "Brokerage_cash_flow" in result_df.columns
    assert "IRA_cash_flow" in result_df.columns

    # Verify symbol cash flows
    assert "SPY_cash_flow" in result_df.columns
    assert "VTI_cash_flow" in result_df.columns

    # Check specific values
    # 2024-01-01: SPY buy -4000 (investment)
    jan_1 = result_df[result_df["date"] == pd.Timestamp("2024-01-01")].iloc[0]
    assert jan_1["investment_cash_flow"] == -4000
    assert jan_1["income_cash_flow"] == 0
    assert jan_1["net_cash_flow"] == -4000
    assert jan_1["SPY_cash_flow"] == -4000
    assert jan_1["Brokerage_cash_flow"] == -4000

    # 2024-01-02: VTI buy -4000 (investment)
    jan_2 = result_df[result_df["date"] == pd.Timestamp("2024-01-02")].iloc[0]
    assert jan_2["investment_cash_flow"] == -4000
    assert jan_2["income_cash_flow"] == 0
    assert jan_2["VTI_cash_flow"] == -4000

    # 2024-01-03: SPY sell 2050 (investment), VTI dividend 50 (income)
    jan_3 = result_df[result_df["date"] == pd.Timestamp("2024-01-03")].iloc[0]
    assert jan_3["investment_cash_flow"] == 2050
    assert jan_3["income_cash_flow"] == 50
    assert jan_3["net_cash_flow"] == 2100
    assert jan_3["SPY_cash_flow"] == 2050
    assert jan_3["VTI_cash_flow"] == 50

    # 2024-01-04: No transactions
    jan_4 = result_df[result_df["date"] == pd.Timestamp("2024-01-04")].iloc[0]
    assert jan_4["investment_cash_flow"] == 0
    assert jan_4["income_cash_flow"] == 0

    # 2024-01-05: SPY buy -6075 (investment)
    jan_5 = result_df[result_df["date"] == pd.Timestamp("2024-01-05")].iloc[0]
    assert jan_5["investment_cash_flow"] == -6075
    assert jan_5["income_cash_flow"] == 0


def test_vectorized_cash_flows_empty_transactions(config):
    """Test vectorized cash flows with no transactions."""

    empty_transactions = pd.DataFrame(
        {
            "date": pd.to_datetime([]),
            "account": [],
            "symbol": [],
            "transaction_type": [],
            "quantity": [],
            "price": [],
            "total_value": [],
            "fees": [],
        }
    )

    market_data = {
        "SPY": pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", "2024-01-05"),
                "close": [400, 401, 402, 403, 404],
                "adjusted_close": [400, 401, 402, 403, 404],
            }
        )
    }

    builder = PortfolioHistoryBuilder(
        config=config,
        transactions=empty_transactions,
        market_data=market_data,
        start_date=pd.Timestamp("2024-01-01"),
        end_date=pd.Timestamp("2024-01-05"),
    )

    dates = pd.date_range("2024-01-01", "2024-01-05", freq="D")
    portfolio_df = pd.DataFrame({"date": dates})

    # pylint: disable=protected-access
    result_df = builder._compute_cash_flows_vectorized(portfolio_df)

    # All cash flows should be zero
    assert (result_df["investment_cash_flow"] == 0).all()
    assert (result_df["income_cash_flow"] == 0).all()
    assert (result_df["net_cash_flow"] == 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
