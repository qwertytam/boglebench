"""
Tests for future dividend handling functionality in DividendProcessor.
"""

import tempfile
from pathlib import Path
from typing import Dict

import pandas as pd
import pytest

from boglebench.core.dividend_processor import DividendProcessor
from boglebench.utils.config import ConfigManager


@pytest.fixture
def sample_config() -> ConfigManager:
    """Create a minimal test configuration."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
dividend:
  auto_validate: true
  auto_calculate_div_per_share: true
  warn_missing_dividends: true
  default_div_type: "CASH"
  comparison_error_margin: 0.01
  handle_future_dividends: "ignore"
""")
        config_path = f.name
    
    config = ConfigManager(config_path)
    yield config
    
    # Cleanup
    Path(config_path).unlink()


@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    """
    Create sample transactions with last transaction on 2023-03-20.
    - Bought 100 shares of VTI on 2023-01-01
    - Bought 50 shares of BND on 2023-01-01
    - Last transaction (BUY) on 2023-03-20
    """
    data = [
        {
            "date": "2023-01-01",
            "symbol": "VTI",
            "transaction_type": "BUY",
            "quantity": 100,
            "value_per_share": 100.00,
            "total_value": 10000.00,
            "account": "Taxable",
            "div_type": "CASH",
            "div_pay_date": pd.NaT,
            "div_record_date": pd.NaT,
            "div_ex_date": pd.NaT,
            "split_ratio": 0,
            "notes": "",
        },
        {
            "date": "2023-01-01",
            "symbol": "BND",
            "transaction_type": "BUY",
            "quantity": 50,
            "value_per_share": 75.00,
            "total_value": 3750.00,
            "account": "IRA",
            "div_type": "CASH",
            "div_pay_date": pd.NaT,
            "div_record_date": pd.NaT,
            "div_ex_date": pd.NaT,
            "split_ratio": 0,
            "notes": "",
        },
        {
            "date": "2023-03-20",
            "symbol": "VTI",
            "transaction_type": "BUY",
            "quantity": 50,
            "value_per_share": 101.00,
            "total_value": 5050.00,
            "account": "Taxable",
            "div_type": "CASH",
            "div_pay_date": pd.NaT,
            "div_record_date": pd.NaT,
            "div_ex_date": pd.NaT,
            "split_ratio": 0,
            "notes": "",
        },
    ]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


@pytest.fixture
def sample_market_data_with_future_dividends() -> Dict[str, pd.DataFrame]:
    """
    Create market data with dividends after last transaction (2023-03-20):
    - VTI: Pays $0.50 on 2023-03-25 (FUTURE)
    - VTI: Pays $0.52 on 2023-06-20 (FUTURE)
    - BND: Pays $0.20 on 2023-04-15 (FUTURE)
    """
    dates_vti = pd.to_datetime(
        ["2023-03-20", "2023-03-21", "2023-03-25", "2023-06-20", "2023-06-21"],
        utc=True
    )
    dates_bnd = pd.to_datetime(
        ["2023-03-20", "2023-04-14", "2023-04-15", "2023-04-16"],
        utc=True
    )

    data = {
        "VTI": pd.DataFrame(
            {
                "date": dates_vti,
                "close": [100, 101, 102, 103, 104],
                "adj_close": [100, 101, 102, 103, 104],
                "dividend": [0.0, 0.0, 0.50, 0.52, 0.0],
                "split_coefficient": [0.0, 0.0, 0.0, 0.0, 0.0],
            },
        ),
        "BND": pd.DataFrame(
            {
                "date": dates_bnd,
                "close": [75, 76, 77, 78],
                "adj_close": [75, 76, 77, 78],
                "dividend": [0.0, 0.0, 0.20, 0.0],
                "split_coefficient": [0.0, 0.0, 0.0, 0.0],
            },
        ),
    }

    return data


def test_ignore_mode_does_not_add_future_dividends(
    sample_config, sample_transactions, sample_market_data_with_future_dividends
):
    """
    Test that 'ignore' mode does not add any future dividends.
    """
    # Set mode to ignore
    sample_config.config["dividend"]["handle_future_dividends"] = "ignore"
    
    start_date = pd.Timestamp("2023-01-01", tz="UTC")
    end_date = pd.Timestamp("2023-12-31", tz="UTC")
    
    processor = DividendProcessor(
        config=sample_config,
        transactions_df=sample_transactions,
        market_data=sample_market_data_with_future_dividends,
        start_date=start_date,
        end_date=end_date,
    )
    
    result_df = processor.run()
    
    # Should have same number of transactions (no dividends added)
    assert len(result_df) == len(sample_transactions)
    
    # No dividend transactions should exist
    dividend_txns = result_df[result_df["transaction_type"] == "DIVIDEND"]
    assert len(dividend_txns) == 0


def test_add_to_all_accounts_mode_adds_future_dividends(
    sample_config, sample_transactions, sample_market_data_with_future_dividends
):
    """
    Test that 'add_to_all_accounts' mode adds future dividends to accounts holding the symbol.
    """
    # Set mode to add_to_all_accounts
    sample_config.config["dividend"]["handle_future_dividends"] = "add_to_all_accounts"
    
    start_date = pd.Timestamp("2023-01-01", tz="UTC")
    end_date = pd.Timestamp("2023-12-31", tz="UTC")
    
    processor = DividendProcessor(
        config=sample_config,
        transactions_df=sample_transactions,
        market_data=sample_market_data_with_future_dividends,
        start_date=start_date,
        end_date=end_date,
    )
    
    result_df = processor.run()
    
    # Should have additional transactions (dividends added)
    assert len(result_df) > len(sample_transactions)
    
    # Check that dividends were added
    dividend_txns = result_df[result_df["transaction_type"] == "DIVIDEND"]
    assert len(dividend_txns) > 0
    
    # Check VTI dividends - should have 2 (March 25 and June 20)
    vti_divs = dividend_txns[dividend_txns["symbol"] == "VTI"]
    assert len(vti_divs) == 2
    
    # Check BND dividends - should have 1 (April 15)
    bnd_divs = dividend_txns[dividend_txns["symbol"] == "BND"]
    assert len(bnd_divs) == 1
    
    # Verify VTI dividend on 2023-03-25
    vti_march_div = vti_divs[vti_divs["date"].dt.date == pd.Timestamp("2023-03-25").date()]
    assert len(vti_march_div) == 1
    # 150 shares (100 + 50) * $0.50 = $75.00 (negative because cash inflow)
    assert abs(vti_march_div.iloc[0]["total_value"] + 75.00) < 0.01
    assert vti_march_div.iloc[0]["value_per_share"] == 0.50
    assert vti_march_div.iloc[0]["account"] == "Taxable"
    
    # Verify BND dividend on 2023-04-15
    bnd_april_div = bnd_divs[bnd_divs["date"].dt.date == pd.Timestamp("2023-04-15").date()]
    assert len(bnd_april_div) == 1
    # 50 shares * $0.20 = $10.00 (negative because cash inflow)
    assert abs(bnd_april_div.iloc[0]["total_value"] + 10.00) < 0.01
    assert bnd_april_div.iloc[0]["value_per_share"] == 0.20
    assert bnd_april_div.iloc[0]["account"] == "IRA"


def test_respects_end_date_boundary(
    sample_config, sample_transactions, sample_market_data_with_future_dividends
):
    """
    Test that dividends after end_date are NOT added.
    """
    # Set mode to add_to_all_accounts
    sample_config.config["dividend"]["handle_future_dividends"] = "add_to_all_accounts"
    
    start_date = pd.Timestamp("2023-01-01", tz="UTC")
    # End date is before the June dividend
    end_date = pd.Timestamp("2023-05-31", tz="UTC")
    
    processor = DividendProcessor(
        config=sample_config,
        transactions_df=sample_transactions,
        market_data=sample_market_data_with_future_dividends,
        start_date=start_date,
        end_date=end_date,
    )
    
    result_df = processor.run()
    
    dividend_txns = result_df[result_df["transaction_type"] == "DIVIDEND"]
    
    # Should only have March and April dividends, not June
    vti_divs = dividend_txns[dividend_txns["symbol"] == "VTI"]
    assert len(vti_divs) == 1  # Only March 25, not June 20
    
    # Check that June dividend is NOT present
    june_divs = dividend_txns[dividend_txns["date"].dt.month == 6]
    assert len(june_divs) == 0


def test_only_adds_to_accounts_with_holdings(sample_config):
    """
    Test that dividends are only added to accounts that actually hold the symbol.
    """
    # Create transactions where only one account holds VTI
    transactions = pd.DataFrame([
        {
            "date": pd.Timestamp("2023-01-01", tz="UTC"),
            "symbol": "VTI",
            "transaction_type": "BUY",
            "quantity": 100,
            "value_per_share": 100.00,
            "total_value": 10000.00,
            "account": "Taxable",
            "div_type": "CASH",
            "div_pay_date": pd.NaT,
            "div_record_date": pd.NaT,
            "div_ex_date": pd.NaT,
            "split_ratio": 0,
            "notes": "",
        },
        {
            "date": pd.Timestamp("2023-01-01", tz="UTC"),
            "symbol": "BND",
            "transaction_type": "BUY",
            "quantity": 50,
            "value_per_share": 75.00,
            "total_value": 3750.00,
            "account": "IRA",
            "div_type": "CASH",
            "div_pay_date": pd.NaT,
            "div_record_date": pd.NaT,
            "div_ex_date": pd.NaT,
            "split_ratio": 0,
            "notes": "",
        },
    ])
    
    # Market data with future dividend for VTI
    market_data = {
        "VTI": pd.DataFrame({
            "date": pd.to_datetime(["2023-03-25"], utc=True),
            "close": [102],
            "adj_close": [102],
            "dividend": [0.50],
            "split_coefficient": [0.0],
        }),
        "BND": pd.DataFrame({
            "date": pd.to_datetime(["2023-03-25"], utc=True),
            "close": [77],
            "adj_close": [77],
            "dividend": [0.20],
            "split_coefficient": [0.0],
        }),
    }
    
    sample_config.config["dividend"]["handle_future_dividends"] = "add_to_all_accounts"
    
    processor = DividendProcessor(
        config=sample_config,
        transactions_df=transactions,
        market_data=market_data,
        start_date=pd.Timestamp("2023-01-01", tz="UTC"),
        end_date=pd.Timestamp("2023-12-31", tz="UTC"),
    )
    
    result_df = processor.run()
    
    dividend_txns = result_df[result_df["transaction_type"] == "DIVIDEND"]
    
    # VTI dividend should only be in Taxable account
    vti_divs = dividend_txns[dividend_txns["symbol"] == "VTI"]
    assert len(vti_divs) == 1
    assert vti_divs.iloc[0]["account"] == "Taxable"
    
    # BND dividend should only be in IRA account
    bnd_divs = dividend_txns[dividend_txns["symbol"] == "BND"]
    assert len(bnd_divs) == 1
    assert bnd_divs.iloc[0]["account"] == "IRA"


def test_does_not_duplicate_existing_dividends(
    sample_config, sample_market_data_with_future_dividends
):
    """
    Test that if user already recorded a future dividend, it's not added again.
    """
    # Transactions with VTI dividend already recorded
    transactions = pd.DataFrame([
        {
            "date": pd.Timestamp("2023-01-01", tz="UTC"),
            "symbol": "VTI",
            "transaction_type": "BUY",
            "quantity": 100,
            "value_per_share": 100.00,
            "total_value": 10000.00,
            "account": "Taxable",
            "div_type": "CASH",
            "div_pay_date": pd.NaT,
            "div_record_date": pd.NaT,
            "div_ex_date": pd.NaT,
            "split_ratio": 0,
            "notes": "",
        },
        {
            "date": pd.Timestamp("2023-03-25", tz="UTC"),
            "symbol": "VTI",
            "transaction_type": "DIVIDEND",
            "quantity": 0,
            "value_per_share": 0.50,
            "total_value": -50.00,  # Already recorded
            "account": "Taxable",
            "div_type": "CASH",
            "div_pay_date": pd.Timestamp("2023-03-25", tz="UTC"),
            "div_record_date": pd.NaT,
            "div_ex_date": pd.NaT,
            "split_ratio": 0,
            "notes": "",
        },
    ])
    
    sample_config.config["dividend"]["handle_future_dividends"] = "add_to_all_accounts"
    
    processor = DividendProcessor(
        config=sample_config,
        transactions_df=transactions,
        market_data=sample_market_data_with_future_dividends,
        start_date=pd.Timestamp("2023-01-01", tz="UTC"),
        end_date=pd.Timestamp("2023-12-31", tz="UTC"),
    )
    
    result_df = processor.run()
    
    # Check VTI dividends on 2023-03-25 - should only have 1 (not duplicated)
    vti_march_divs = result_df[
        (result_df["symbol"] == "VTI")
        & (result_df["transaction_type"] == "DIVIDEND")
        & (result_df["date"].dt.date == pd.Timestamp("2023-03-25").date())
    ]
    assert len(vti_march_divs) == 1
    
    # But June dividend should be added (not already recorded)
    vti_june_divs = result_df[
        (result_df["symbol"] == "VTI")
        & (result_df["transaction_type"] == "DIVIDEND")
        & (result_df["date"].dt.date == pd.Timestamp("2023-06-20").date())
    ]
    assert len(vti_june_divs) == 1


def test_handles_multiple_accounts_holding_same_symbol(sample_config):
    """
    Test that when multiple accounts hold the same symbol, dividends are added to each.
    """
    # Both accounts hold VTI
    transactions = pd.DataFrame([
        {
            "date": pd.Timestamp("2023-01-01", tz="UTC"),
            "symbol": "VTI",
            "transaction_type": "BUY",
            "quantity": 100,
            "value_per_share": 100.00,
            "total_value": 10000.00,
            "account": "Taxable",
            "div_type": "CASH",
            "div_pay_date": pd.NaT,
            "div_record_date": pd.NaT,
            "div_ex_date": pd.NaT,
            "split_ratio": 0,
            "notes": "",
        },
        {
            "date": pd.Timestamp("2023-01-01", tz="UTC"),
            "symbol": "VTI",
            "transaction_type": "BUY",
            "quantity": 50,
            "value_per_share": 100.00,
            "total_value": 5000.00,
            "account": "IRA",
            "div_type": "CASH",
            "div_pay_date": pd.NaT,
            "div_record_date": pd.NaT,
            "div_ex_date": pd.NaT,
            "split_ratio": 0,
            "notes": "",
        },
    ])
    
    market_data = {
        "VTI": pd.DataFrame({
            "date": pd.to_datetime(["2023-03-25"], utc=True),
            "close": [102],
            "adj_close": [102],
            "dividend": [0.50],
            "split_coefficient": [0.0],
        }),
    }
    
    sample_config.config["dividend"]["handle_future_dividends"] = "add_to_all_accounts"
    
    processor = DividendProcessor(
        config=sample_config,
        transactions_df=transactions,
        market_data=market_data,
        start_date=pd.Timestamp("2023-01-01", tz="UTC"),
        end_date=pd.Timestamp("2023-12-31", tz="UTC"),
    )
    
    result_df = processor.run()
    
    dividend_txns = result_df[result_df["transaction_type"] == "DIVIDEND"]
    
    # Should have 2 dividend transactions (one for each account)
    assert len(dividend_txns) == 2
    
    # Check Taxable account dividend: 100 shares * $0.50
    taxable_div = dividend_txns[dividend_txns["account"] == "Taxable"]
    assert len(taxable_div) == 1
    assert abs(taxable_div.iloc[0]["total_value"] + 50.00) < 0.01
    
    # Check IRA account dividend: 50 shares * $0.50
    ira_div = dividend_txns[dividend_txns["account"] == "IRA"]
    assert len(ira_div) == 1
    assert abs(ira_div.iloc[0]["total_value"] + 25.00) < 0.01
