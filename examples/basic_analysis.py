"""
Example of basic portfolio analysis using BogleBench.

This example demonstrates:
1. Loading transaction data
2. Building portfolio history (now in database)
3. Loading symbol attributes
4. Calculating performance metrics
5. Querying normalized data
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import boglebench
sys.path.insert(0, str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position
from boglebench import BogleBenchAnalyzer


def main():
    """Run a basic portfolio analysis example."""

    print("🚀 BogleBench Portfolio Analysis Example (with SQL Database)")
    print("=" * 60)

    # Initialize the analyzer
    config_path = "~/boglebench_data/config/config.yaml"

    try:
        analyzer = BogleBenchAnalyzer(config_path=config_path)
        print("✅ Initialized BogleBench analyzer")

        # Step 1: Load transaction data
        print("\n📊 Step 1: Loading transaction data...")
        transactions = analyzer.load_transactions()

        print(f"   Loaded {len(transactions)} transactions")
        print(f"   Assets: {', '.join(transactions['symbol'].unique())}")
        print(f"   Accounts: {', '.join(transactions['account'].unique())}")
        print(
            f"   Date range: {transactions['date'].min().date()} "
            f"to {transactions['date'].max().date()}"
        )

        # Step 2: Build portfolio history (writes to database)
        print("\n🏗️  Step 2: Building portfolio history to database...")
        portfolio_db = analyzer.build_portfolio_history()

        print("   ✅ Portfolio history stored in database")
        portfolio_db.print_stats()

        # Step 3: Load symbol attributes (separate from transactions)
        print("\n📋 Step 3: Loading symbol attributes...")
        # Note: Attributes are now loaded separately, not from transactions
        # analyzer.load_symbol_attributes(csv_path='path/to/attributes.csv')
        print("   ℹ️  Attributes not loaded (optional)")
        print("   ℹ️  To enable attribution analysis, create an attributes CSV and load it")

        # Show current attributes
        attributes = portfolio_db.get_symbol_attributes()
        if not attributes.empty:
            print(f"   ✅ Found attributes for {len(attributes)} symbols")
            print("\n   Attribute Summary:")
            if "geography" in attributes.columns:
                print(
                    f"   Geographies: {attributes['geography'].unique().tolist()}"
                )
            if "sector" in attributes.columns:
                print(f"   Sectors: {attributes['sector'].unique().tolist()}")

        # Step 4: Query normalized data
        print("\n🔍 Step 4: Querying normalized data...")

        # Get latest portfolio summary
        latest = portfolio_db.get_latest_portfolio()
        print(f"\n   Latest Portfolio Value: ${latest['total_value']:,.2f}")
        print(f"   Latest Date: {latest['date'].date()}")

        # Get current holdings
        holdings = portfolio_db.get_latest_holdings()
        print(f"\n   Current Holdings: {len(holdings)} positions")
        for _, holding in holdings.head(5).iterrows():
            print(
                f"      {holding['account']}/{holding['symbol']}: "
                f"{holding['quantity']:.2f} shares, ${holding['value']:,.2f} "
                f"({holding['weight']:.1%})"
            )

        # Get allocation by geography (if attributes loaded)
        if not attributes.empty and "geography" in attributes.columns:
            print("\n   Allocation by Geography:")
            geo_alloc = portfolio_db.get_allocation_by_attribute("geography")
            for _, row in geo_alloc.iterrows():
                print(
                    f"      {row['category']}: {row['total_weight']:.1%} "
                    f"(${row['total_value']:,.2f})"
                )

        # Get account breakdown
        accounts = portfolio_db.get_accounts()
        print("\n   Account Breakdown:")
        for account in accounts:
            account_data = portfolio_db.get_account_data(account=account)
            if not account_data.empty:
                latest_account = account_data.iloc[-1]
                print(
                    f"      {account}: ${latest_account['total_value']:,.2f} "
                    f"({latest_account['weight']:.1%})"
                )

        # Step 5: Calculate performance metrics (using legacy method)
        print("\n📊 Step 5: Calculating performance metrics...")
        results = analyzer.calculate_performance()

        print("\n" + "=" * 60)
        print(results.summary())
        print("=" * 60)

        # Step 6: Export results
        print("\n💾 Step 6: Exporting results...")
        export_path = results.export_to_csv()
        print(f"   Results exported to: {export_path}")

        print("\n✅ Analysis complete!")

    # pylint: disable=broad-except
    except Exception as e:
        print(f"\n❌ Error: {e}")

        # pylint: disable=import-outside-toplevel
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
