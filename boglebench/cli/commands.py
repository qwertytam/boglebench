"""
Command line interface for BogleBench portfolio analyzer.
"""

import shutil
from pathlib import Path

import click

from ..utils.config import ConfigManager
from ..utils.logging_config import get_logger, setup_logging
from ..utils.workspace import WorkspaceContext


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

    click.echo(f"Initializing BogleBench workspace at: {path}")
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
        "logs",
    ]

    for dir_name in directories:
        dir_path = workspace_path / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        # print(f"DEBUG: Created directory: {dir_path}")

    print(f"INFO: Created workspace directories under: {workspace_path}")

    # Copy template files
    _copy_templates(workspace_path, force)

    # Initialize logging (will use correct workspace)
    # print("DEBUG: init_workspace Setting up logging...")
    setup_logging()
    # logger = get_logger("cli.init")

    # Create configuration file
    config_manager = ConfigManager()
    config_path = workspace_path / "config" / "config.yaml"

    if config_path.exists() and not force:
        click.echo(f"Configuration file already exists: {config_path}")
        click.echo("Use --force to overwrite")
    else:
        config_manager.create_config_file(str(config_path))

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

    # Copy notebook template to output directory
    notebook_template = templates_source / "analysis_template.ipynb"
    if notebook_template.exists():
        dest_file = workspace_path / "output" / "analysis_template.ipynb"
        if not dest_file.exists() or force:
            shutil.copy2(notebook_template, dest_file)
            click.echo(f"INFO: Copied template: {dest_file}")

    # Copy logging config to config directory
    logging_template = templates_source / "logging_config_template.yaml"
    if logging_template.exists():
        dest_file = workspace_path / "config" / "logging.yaml"
        if not dest_file.exists() or force:
            shutil.copy2(logging_template, dest_file)
            click.echo(f"INFO: Created logging configuration: {dest_file}")


def _create_sample_transactions(file_path: Path, force: bool):
    """Create a sample transactions CSV file from template."""
    if file_path.exists() and not force:
        click.echo(
            f"INFO: Sample transactions file already exists: {file_path}"
        )
        return

    # Get template file path
    templates_source = Path(__file__).parent.parent / "templates"
    sample_template = templates_source / "sample_transactions.csv"

    if sample_template.exists():
        # Copy template to destination
        import shutil

        shutil.copy2(sample_template, file_path)
        click.echo(f"INFO: Created sample transactions: {file_path}")
    else:
        # Fallback if template not found
        click.echo(f"WARNING: Sample template not found at {sample_template}")
        click.echo("INFO: Creating minimal sample file")

        minimal_sample = """date,ticker,transaction_type,shares,price_per_share,account
2023-01-15,AAPL,BUY,100,150.50,Default
2023-02-15,SPY,BUY,50,380.00,Default
"""
        with open(file_path, "w") as f:
            f.write(minimal_sample)
        click.echo(f"Created minimal sample transactions: {file_path}")

    click.echo("Sample includes transactions across multiple accounts:")
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

    # Set workspace context early
    if config:
        config_path = Path(config).expanduser()
        if config_path.exists():
            workspace = WorkspaceContext.discover_workspace(config_path.parent)

    # Now initialize logging (will use correct workspace)
    setup_logging()
    logger = get_logger("cli.analyze")

    logger.info("🚀 Running BogleBench portfolio analysis...")
    logger.info(
        "📈 Analyzing your portfolio with Bogle's principles in mind..."
    )

    # This would be implemented in your main analyzer
    from ..core.portfolio import BogleBenchAnalyzer

    try:
        analyzer = BogleBenchAnalyzer(config_path=config)

        # Override benchmark if specified
        if benchmark:
            analyzer.config.config["settings"]["benchmark_ticker"] = benchmark
            logger.info(f"📊 Using custom benchmark: {benchmark}")

        logger.info("Loading transaction data...")
        analyzer.load_transactions()

        logger.info("Fetching market data...")
        analyzer.fetch_market_data()

        logger.info("Calculating performance metrics...")
        results = analyzer.calculate_performance()

        # Display results
        logger.info("\n" + results.summary())

        # Export results
        output_dir = analyzer.config.get_output_path()
        results.export_to_csv(str(output_dir))

        # Create charts if requested
        if create_charts:
            logger.info("📊 Creating performance charts...")
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

        logger.info(f"✅ Analysis complete! Results saved to {output_dir}")

    except Exception as e:
        logger.error(f"❌ Error running analysis: {e}")
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

    # Discover workspace from transaction file path
    WorkspaceContext.discover_workspace(path)

    # Initialize logging (will now use correct workspace)
    setup_logging()
    logger = get_logger("cli.validate")

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


@click.command()
@click.option("--days", default=30, help="Days of logs to keep")
@click.option("--config", help="Path to configuration file")
def cleanup_logs(days: int, config: str):
    """Clean up old log files."""
    if config:
        WorkspaceContext.discover_workspace(Path(config).parent)

    setup_logging()
    logger_instance = BogleBenchLogger()
    logger_instance.cleanup_old_logs(days)

    click.echo(f"Log cleanup completed - kept logs from last {days} days")


if __name__ == "__main__":
    init_workspace()
