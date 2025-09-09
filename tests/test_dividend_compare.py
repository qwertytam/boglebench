import pandas as pd
import pytest

from boglebench.core.portfolio import BogleBenchAnalyzer


class DummyConfig:
    def get(self, *args, **kwargs):
        return None


@pytest.fixture
def analyzer():
    analyzer = BogleBenchAnalyzer()
    analyzer.config = DummyConfig()
    return analyzer


def make_market_data():
    # Simulate AlphaVantage market data with dividends
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "open": [100, 102, 101],
            "close": [101, 103, 104],
            "dividend": [0.0, 1.23, 0.0],
        }
    )
    return {"VTI": df}


def make_transactions_div_and_reinvest(match=True):
    # Simulate user transactions for VTI, both DIVIDEND and DIVIDEND_REINVEST
    # on same date
    if match:
        amount_div = 1.23
        amount_reinvest = -1.00
    else:
        amount_div = 0.23
        amount_reinvest = -2.00
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-02"]),
            "ticker": ["VTI", "VTI", "VTI"],
            "transaction_type": ["BUY", "DIVIDEND", "DIVIDEND_REINVEST"],
            "quantity": [1, 0, 0.01],
            "value_per_share": [100, amount_div, amount_reinvest],
            "total_value": [0, amount_div, amount_reinvest],
            "account": ["Test", "Test", "Test"],
        }
    )
    return df


def test_dividend_and_reinvest_match(analyzer):
    analyzer.market_data = make_market_data()
    analyzer.transactions = make_transactions_div_and_reinvest(match=True)
    messages = analyzer.compare_user_dividends_to_market(
        "VTI", warn_missing_dividends=True, auto_calculate_div_per_share=False
    )
    messages_str = "\n".join(messages)

    assert "No user dividends" not in messages_str
    assert "Cannot calculate" not in messages_str
    assert "Dividend mismatch" not in messages_str


def test_dividend_and_reinvest_mismatch(analyzer):
    analyzer.market_data = make_market_data()
    analyzer.transactions = make_transactions_div_and_reinvest(match=False)
    messages = analyzer.compare_user_dividends_to_market(
        "VTI", warn_missing_dividends=True, auto_calculate_div_per_share=False
    )
    messages_str = "\n".join(messages)

    assert "Dividend mismatch" in messages_str
    assert "user: $0.2300, market: $1.2300" in messages_str
