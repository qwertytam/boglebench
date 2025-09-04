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
        amount_div = 0.23
        amount_reinvest = 1.00
    else:
        amount_div = 0.23
        amount_reinvest = 2.00
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-02", "2023-01-02"]),
            "ticker": ["VTI", "VTI"],
            "transaction_type": ["DIVIDEND", "DIVIDEND_REINVEST"],
            "shares": [0, 0.01],
            "price_per_share": [0, 100],
            "amount": [amount_div, amount_reinvest],
            "account": ["Test", "Test"],
        }
    )
    return df


def test_dividend_and_reinvest_match(analyzer, capsys):
    analyzer.market_data = make_market_data()
    analyzer.transactions = make_transactions_div_and_reinvest(match=True)
    analyzer.compare_user_dividends_to_alphavantage("VTI")
    captured = capsys.readouterr()
    assert "Dividend mismatch" not in captured.out


def test_dividend_and_reinvest_mismatch(analyzer, capsys):
    analyzer.market_data = make_market_data()
    analyzer.transactions = make_transactions_div_and_reinvest(match=False)
    analyzer.compare_user_dividends_to_alphavantage("VTI")
    captured = capsys.readouterr()
    assert "Dividend mismatch" in captured.out
    assert "user total=2.23 vs market data=1.23" in captured.out
