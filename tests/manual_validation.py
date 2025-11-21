"""
Manual validation script to test short position handling with sample transactions.

This script demonstrates:
1. Loading transactions in REJECT mode (should fail)
2. Loading transactions in CAP mode (should adjust)
"""

import sys
import tempfile
import traceback
from pathlib import Path

import pandas as pd

from boglebench import BogleBenchAnalyzer, ShortPositionError

# Test with the sample transactions that has a short position
sample_csv = (
    Path(__file__).parent.parent
    / "boglebench"
    / "templates"
    / "sample_transactions.csv"
)


def create_test_config(config_path: Path, short_handling: str):
    """Create a test configuration file."""
    config_content = f"""
data:
  base_path: "{config_path.parent}"
  transactions_file: "{sample_csv}"

validation:
  short_position_handling: "{short_handling}"

settings:
  cache_market_data: false

api:
  alpha_vantage_key: "demo"
"""
    config_path.write_text(config_content)


def test_reject_mode():
    """Test REJECT mode - should fail on short position."""
    print("\n" + "=" * 70)
    print("TEST 1: REJECT mode (should detect and reject short position)")
    print("=" * 70)

    try:

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            config_path = tmppath / "config.yaml"
            create_test_config(config_path, "reject")

            analyzer = BogleBenchAnalyzer(config_path=str(config_path))
            _ = analyzer.load_transactions()

            print("❌ FAILED: Should have raised ShortPositionError")
            return False

    except ShortPositionError as e:
        print("✅ SUCCESS: Short position detected and rejected")
        print("\nError details:")
        print(str(e))
        return True
    except Exception as e:  # pylint: disable=broad-except
        print(f"❌ FAILED: Unexpected error: {e}")

        traceback.print_exc()
        return False


def test_cap_mode():
    """Test CAP mode - should adjust short position."""
    print("\n" + "=" * 70)
    print("TEST 2: CAP mode (should detect and cap short position)")
    print("=" * 70)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            config_path = tmppath / "config.yaml"
            create_test_config(config_path, "cap")

            analyzer = BogleBenchAnalyzer(config_path=str(config_path))
            transactions = analyzer.load_transactions()

            # Find the adjusted transaction (2023-11-15, AAPL, SELL)
            aapl_sells = transactions[
                (transactions["symbol"] == "AAPL")
                & (transactions["transaction_type"] == "SELL")
                & (
                    transactions["date"].dt.date
                    == pd.Timestamp("2023-11-15").date()
                )
            ]

            if len(aapl_sells) == 0:
                print("❌ FAILED: Could not find the AAPL SELL transaction")
                return False

            adjusted_qty = aapl_sells.iloc[0]["quantity"]

            # Should be capped to -125 (all available shares)
            if abs(adjusted_qty - (-125.0)) < 0.01:
                print("✅ SUCCESS: Short position detected and capped")
                print("\nOriginal transaction: SELL 200 shares")
                print(
                    f"Adjusted transaction: SELL {abs(adjusted_qty):.0f} shares"
                )
                print("Result: 0 shares remaining (prevented short position)")
                return True
            else:
                print(f"❌ FAILED: Expected quantity -125, got {adjusted_qty}")
                return False

    except Exception as e:  # pylint: disable=broad-except
        print(f"❌ FAILED: Unexpected error: {e}")

        traceback.print_exc()
        return False


def test_ignore_mode():
    """Test IGNORE mode - should allow short position with warning."""
    print("\n" + "=" * 70)
    print("TEST 3: IGNORE mode (should detect and allow short position)")
    print("=" * 70)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            config_path = tmppath / "config.yaml"
            create_test_config(config_path, "ignore")

            analyzer = BogleBenchAnalyzer(config_path=str(config_path))
            transactions = analyzer.load_transactions()

            # Find the transaction (2023-11-15, AAPL, SELL)
            aapl_sells = transactions[
                (transactions["symbol"] == "AAPL")
                & (transactions["transaction_type"] == "SELL")
                & (
                    transactions["date"].dt.date
                    == pd.Timestamp("2023-11-15").date()
                )
            ]

            if len(aapl_sells) == 0:
                print("❌ FAILED: Could not find the AAPL SELL transaction")
                return False

            actual_qty = aapl_sells.iloc[0]["quantity"]

            # Should remain unchanged at -200
            if abs(actual_qty - (-200.0)) < 0.01:
                print("✅ SUCCESS: Short position detected and allowed")
                print("\nOriginal transaction: SELL 200 shares")
                print(f"Actual transaction: SELL {abs(actual_qty):.0f} shares")
                print(
                    "Result: -75 shares (short position allowed with warning)"
                )
                return True
            else:
                print(f"❌ FAILED: Expected quantity -200, got {actual_qty}")
                return False

    except Exception as e:  # pylint: disable=broad-except
        print(f"❌ FAILED: Unexpected error: {e}")

        traceback.print_exc()
        return False


if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("SHORT POSITION HANDLING - MANUAL VALIDATION")
    print("=" * 70)
    print(f"\nUsing sample transactions from: {sample_csv}")

    if not sample_csv.exists():
        print(f"❌ ERROR: Sample transactions file not found: {sample_csv}")
        sys.exit(1)

    # Run tests
    TEST1_PASS = test_reject_mode()
    TEST2_PASS = test_cap_mode()
    TEST3_PASS = test_ignore_mode()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"Test 1 (REJECT mode): {'✅ PASSED' if TEST1_PASS else '❌ FAILED'}"
    )
    print(
        f"Test 2 (CAP mode):    {'✅ PASSED' if TEST2_PASS else '❌ FAILED'}"
    )
    print(
        f"Test 3 (IGNORE mode): {'✅ PASSED' if TEST3_PASS else '❌ FAILED'}"
    )

    if TEST1_PASS and TEST2_PASS and TEST3_PASS:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)
