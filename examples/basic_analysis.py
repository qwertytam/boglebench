#!/usr/bin/env python3
"""
Basic BogleBench Analysis Example

This script demonstrates how to use BogleBench for portfolio performance analysis.
It shows the complete workflow from loading transactions to generating performance reports.

Usage:
    python examples/basic_analysis.py

Make sure you have:
1. Initialized a BogleBench workspace: boglebench-init --path ~/my_data
2. Added your transaction data to ~/my_data/transactions/
3. Updated the config file path below
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import boglebench
sys.path.insert(0, str(Path(__file__).parent.parent))

from boglebench import BogleBenchAnalyzer


def main():
    """Run a basic portfolio analysis example."""

    print("🚀 BogleBench Portfolio Analysis Example")
    print("=" * 50)

    # Initialize the analyzer
    # Update this path to your actual data directory
    config_path = "~/boglebench_data/config/config.yaml"

    try:
        analyzer = BogleBenchAnalyzer(config_path=config_path)
        print(f"✅ Initialized BogleBench analyzer")

        # Step 1: Load transaction data
        print("\n📊 Step 1: Loading transaction data...")
        transactions = analyzer.load_transactions()

        print(f"   Loaded {len(transactions)} transactions")
        print(f"   Assets: {', '.join(transactions['ticker'].unique())}")
        print(f"   Accounts: {', '.join(transactions['account'].unique())}")
        print(
            f"   Date range: {transactions['date'].min().date()} to {transactions['date'].max().date()}"
        )

        # Step 2: Build portfolio history
        print("\n🏗️  Step 3: Building portfolio history...")
        portfolio_history = analyzer.build_portfolio_history()

        print(f"   Built history over {len(portfolio_history)} trading days")

        # Step 3: Calculate performance metrics
        print("\n📊 Step 4: Calculating performance metrics...")
        results = analyzer.calculate_performance()

        # Step 4: Display results
        print("\n" + "=" * 60)
        print(results.summary())
        print("=" * 60)

        # Step 5: Export results
        print("\n💾 Step 5: Exporting results...")
        export_path = results.export_to_csv()
        print(f"   Results exported to: {export_path}")

        # Additional analysis examples
        print("\n🔍 Additional Analysis:")

        # Account breakdown
        account_summary = results.get_account_summary()
        if not account_summary.empty:
            print("\n🏦 Account Breakdown:")
            for _, account in account_summary.iterrows():
                print(
                    f"   {account['account']}: ${account['current_value']:,.2f} "
                    f"({account['weight_of_portfolio']:.1%} of portfolio)"
                )

        # Current holdings by account
        holdings = results.get_account_holdings()
        if not holdings.empty:
            print("\n📊 Current Holdings by Account:")
            for account in holdings["account"].unique():
                account_holdings = holdings[holdings["account"] == account]
                print(f"\n   {account}:")
                for _, holding in account_holdings.iterrows():
                    print(
                        f"     {holding['ticker']}: {holding['shares']:.2f} shares "
                        f"(${holding['value']:,.2f}, {holding['weight']:.1%})"
                    )

        # Get portfolio returns for further analysis
        returns = results.get_portfolio_returns()
        print(f"\n📈 Portfolio Performance:")
        print(
            f"   Average daily return: {returns.mean():.4f} ({returns.mean()*252:.2%} annualized)"
        )
        print(f"   Best day: +{returns.max():.2%}")
        print(f"   Worst day: {returns.min():.2%}")

        # Get cumulative returns
        cum_returns = results.get_cumulative_returns()
        print(f"   Total return: {cum_returns.iloc[-1]:.2%}")

        print("\n✅ Analysis complete!")
        print("\n💡 Remember John Bogle's wisdom:")
        print("   - Stay the course with long-term investing")
        print("   - Keep costs low")
        print("   - Diversify broadly")
        print("   - Don't try to time the market")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\n💡 Quick setup:")
        print("   1. Run: boglebench-init --path ~/my_data")
        print("   2. Add your transactions to ~/my_data/transactions/")
        print("   3. Update the config_path in this script")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("\n🐛 If you need help, check:")
        print(
            "   - Transaction CSV format (date, ticker, transaction_type, shares, price_per_share)"
        )
        print("   - Internet connection for market data")
        print("   - Configuration file settings")


def create_sample_data():
    """
    Helper function to create sample transaction data for testing.

    This creates a sample CSV file that you can use to test BogleBench.
    """
    import pandas as pd

    # Try to read from template first
    try:
        template_path = (
            Path(__file__).parent.parent
            / "boglebench"
            / "templates"
            / "sample_transactions.csv"
        )
        if template_path.exists():
            sample_transactions = pd.read_csv(template_path)
        else:
            raise FileNotFoundError("Template not found")
    except (FileNotFoundError, ImportError):
        # Fallback to minimal programmatic generation
        sample_transactions = pd.DataFrame(
            {
                "date": ["2023-01-15", "2023-02-15", "2023-03-15"],
                "ticker": ["AAPL", "SPY", "MSFT"],
                "transaction_type": ["BUY", "BUY", "BUY"],
                "quantity": [100, 50, 25],
                "value_per_share": [150.50, 380.00, 240.25],
                "account": ["Test", "Test", "Test"],
            }
        )

    # Save to current directory
    output_file = Path("sample_transactions.csv")
    sample_transactions.to_csv(output_file, index=False)

    print(f"📄 Created sample transaction file: {output_file}")
    print("   You can copy this to your BogleBench transactions directory")
    print("📊 Sample includes multiple broker accounts:")
    print("   - Schwab_401k: 401(k) retirement account")
    print("   - Fidelity_IRA: IRA retirement account")
    print("   - Personal_Brokerage: Taxable brokerage account")

    return sample_transactions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BogleBench Analysis Example")
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create sample transaction data file",
    )
    parser.add_argument(
        "--config", type=str, help="Path to BogleBench config file"
    )

    args = parser.parse_args()

    if args.create_sample:
        create_sample_data()
    else:
        if args.config:
            # Update the config path if provided
            main.__globals__["config_path"] = args.config
        main()
