"""
Command line interface for BogleBench portfolio analyzer.
"""

import shutil
from pathlib import Path

import click

from ..utils.config import ConfigManager


@click.command()
@click.option(
    "--path",
    default="~/boglebench_data",
    help="Path where to create the BogleBench data workspace",
)
@click.option("--force", is_flag=True, help="Overwrite existing files")
def init_workspace(path: str, force: bool):
    """Initialize a new BogleBench portfolio analysis workspace."""
    workspace_path = Path(path).expanduser()

    click.echo(f"Initializing BogleBench workspace at: {workspace_path}")
    click.echo(
        "📊 In the spirit of John Bogle: Simple, low-cost, long-term investing analysis"
    )

    # Create directory structure
    directories = [
        "config",
        "transactions",
        "market_data",
        "output",
        "output/reports",
    ]

    for dir_name in directories:
        dir_path = workspace_path / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        click.echo(f"Created directory: {dir_path}")

    # Create configuration file
    config_manager = ConfigManager()
    config_path = workspace_path / "config" / "config.yaml"

    if config_path.exists() and not force:
        click.echo(f"Configuration file already exists: {config_path}")
        click.echo("Use --force to overwrite")
    else:
        config_manager.create_config_file(str(config_path))

    # Copy template files
    _copy_templates(workspace_path, force)

    # Create sample transactions file
    _create_sample_transactions(
        workspace_path / "transactions" / "sample_transactions.csv", force
    )

    click.echo(f"\n✅ BogleBench workspace initialized successfully!")
    click.echo(f"\nNext steps:")
    click.echo(f"1. Edit your configuration: {config_path}")
    click.echo(
        f"2. Add your transaction data to: {workspace_path}/transactions/"
    )
    click.echo(f"3. Run analysis: boglebench-analyze --config {config_path}")
    click.echo(
        f"\n💡 Remember Bogle's wisdom: 'Stay the course' and focus on long-term results!"
    )


def _copy_templates(workspace_path: Path, force: bool):
    """Copy template files to workspace."""
    templates_source = Path(__file__).parent.parent / "templates"

    if not templates_source.exists():
        click.echo("Warning: Template files not found in package")
        return

    for template_file in templates_source.glob("*"):
        dest_file = workspace_path / "output" / template_file.name

        if dest_file.exists() and not force:
            click.echo(f"Template already exists: {dest_file}")
        else:
            shutil.copy2(template_file, dest_file)
            click.echo(f"Copied template: {dest_file}")


def _create_sample_transactions(file_path: Path, force: bool):
    """Create a sample transactions CSV file."""
    if file_path.exists() and not force:
        click.echo(f"Sample transactions file already exists: {file_path}")
        return

    sample_data = """date,ticker,transaction_type,shares,price_per_share,account
2023-01-17,AAPL,BUY,100,150.50,Schwab_401k
2023-01-17,MSFT,BUY,50,240.25,Schwab_401k
2023-01-17,SPY,BUY,25,380.00,Fidelity_IRA
2023-02-15,AAPL,BUY,50,155.75,Schwab_401k
2023-02-15,VTI,BUY,100,200.00,Fidelity_IRA
2023-03-15,SPY,BUY,25,385.00,Personal_Brokerage
2023-04-13,AAPL,SELL,25,165.25,Schwab_401k
2023-05-15,GOOGL,BUY,10,105.50,Personal_Brokerage
"""

    with open(file_path, "w") as f:
        f.write(sample_data)

    click.echo(f"Created sample transactions: {file_path}")
    click.echo("📊 Sample includes transactions across multiple accounts:")
    click.echo("   - Schwab_401k (401k retirement account)")
    click.echo("   - Fidelity_IRA (IRA retirement account)")
    click.echo("   - Personal_Brokerage (taxable brokerage account)")


@click.command()
@click.option("--config", help="Path to configuration file")
@click.option(
    "--output-format",
    default="jupyter",
    type=click.Choice(["jupyter", "html", "pdf"]),
    help="Output format for analysis",
)
@click.option(
    "--create-charts", is_flag=True, help="Generate performance charts"
)
@click.option("--benchmark", help="Override benchmark ticker (e.g., SPY, VTI)")
def run_analysis(
    config: str, output_format: str, create_charts: bool, benchmark: str
):
    """Run BogleBench portfolio analysis."""
    click.echo("🚀 Running BogleBench portfolio analysis...")
    click.echo("📈 Analyzing your portfolio with Bogle's principles in mind...")

    # This would be implemented in your main analyzer
    from ..core.portfolio import BogleBenchAnalyzer

    try:
        analyzer = BogleBenchAnalyzer(config_path=config)

        # Override benchmark if specified
        if benchmark:
            analyzer.config.config["settings"]["benchmark_ticker"] = benchmark
            click.echo(f"📊 Using custom benchmark: {benchmark}")

        click.echo("Loading transaction data...")
        analyzer.load_transactions()

        click.echo("Fetching market data...")
        analyzer.fetch_market_data()

        click.echo("Calculating performance metrics...")
        results = analyzer.calculate_performance()

        # Display results
        click.echo("\n" + results.summary())

        # Export results
        output_dir = analyzer.config.get_output_path()
        results.export_to_csv(str(output_dir))

        # Create charts if requested
        if create_charts:
            click.echo("📊 Creating performance charts...")
            from ..visualization.charts import BogleBenchCharts

            charts = BogleBenchCharts(results)

            # Create dashboard
            chart_path = Path(output_dir) / "performance_dashboard.png"
            charts.create_performance_dashboard(str(chart_path))

            # Create account comparison if multiple accounts
            account_summary = results.get_account_summary()
            if len(account_summary) > 1:
                account_chart_path = output_dir / "account_comparison.png"
                charts.create_account_comparison(str(account_chart_path))

        click.echo(f"✅ Analysis complete! Results saved to {output_dir}")

    except Exception as e:
        click.echo(f"❌ Error running analysis: {e}")
        raise click.Abort()


@click.command()
@click.option("--config", help="Path to configuration file")
@click.option("--account", help="Show holdings for specific account only")
def show_holdings(config: str, account: str):
    """Show current portfolio holdings."""
    from ..core.portfolio import BogleBenchAnalyzer

    try:
        analyzer = BogleBenchAnalyzer(config_path=config)
        analyzer.load_transactions()
        analyzer.fetch_market_data()
        analyzer.build_portfolio_history()
        results = analyzer.calculate_performance()

        holdings = results.get_account_holdings(account)

        if holdings.empty:
            click.echo("No holdings found.")
            return

        click.echo("\n📊 Current Holdings:")
        click.echo("=" * 80)

        if account:
            click.echo(f"Account: {account}")

        for _, holding in holdings.iterrows():
            click.echo(
                f"{holding['ticker']:6} | "
                f"{holding['account']:15} | "
                f"{holding['shares']:>8.2f} shares | "
                f"${holding['price']:>8.2f} | "
                f"${holding['value']:>10,.2f} | "
                f"{holding['weight']:>6.1%}"
            )

        total_value = holdings["value"].sum()
        click.echo("=" * 80)
        click.echo(f"Total Value: ${total_value:,.2f}")

    except Exception as e:
        click.echo(f"❌ Error: {e}")


@click.command()
@click.option("--path", required=True, help="Path to CSV file to validate")
def validate_transactions(path: str):
    """Validate transaction CSV file format."""
    from ..core.portfolio import BogleBenchAnalyzer

    try:
        click.echo(f"🔍 Validating transaction file: {path}")

        analyzer = BogleBenchAnalyzer()
        transactions = analyzer.load_transactions(path)

        click.echo("✅ Transaction file is valid!")
        click.echo(f"   📊 {len(transactions)} transactions")
        click.echo(f"   🏦 {transactions['account'].nunique()} accounts")
        click.echo(f"   📈 {transactions['ticker'].nunique()} assets")
        click.echo(
            f"   📅 {transactions['date'].min().date()} to {transactions['date'].max().date()}"
        )

    except Exception as e:
        click.echo(f"❌ Validation failed: {e}")
        raise click.Abort()


if __name__ == "__main__":
    init_workspace()
