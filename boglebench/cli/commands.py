"""
Command-line interface for BogleBench.

This module implements CLI commands for portfolio analysis including workspace
initialization, transaction validation, portfolio analysis, and holdings display.
Uses Click framework for command-line parsing and user interaction.
"""

import shutil
from pathlib import Path

import click

from ..core.portfolio import BogleBenchAnalyzer
from ..utils.config import ConfigManager
from ..utils.logging_config import BogleBenchLogger, get_logger, setup_logging
from ..utils.workspace import WorkspaceContext
from ..visualization.charts import BogleBenchCharts


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

    click.echo(f"Created workspace directories under: {workspace_path}")

    # Copy template files
    _copy_templates(workspace_path, force)

    # Initialize logging (will use correct workspace)
    setup_logging()

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

    # Create sample attributions file
    _create_sample_attributions(
        workspace_path / "transactions" / "sample_attributions.csv", force
    )

    click.echo("\n✅ BogleBench workspace initialized successfully!")
    click.echo("\nNext steps:")
    click.echo(f"1. Edit your configuration: {config_path}")
    click.echo(
        f"2. Add your transaction data to: {workspace_path}/transactions/"
    )
    click.echo(f"3. Run analysis: boglebench-analyze --config {config_path}")
    click.echo(
        "\n💡 Remember Bogle's wisdom: 'Stay the course' "
        "and focus on long-term results!"
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
            click.echo(f"Copied template: {dest_file}")

    # Copy logging config to config directory
    logging_template = templates_source / "logging_config_template.yaml"
    if logging_template.exists():
        dest_file = workspace_path / "config" / "logging.yaml"
        if not dest_file.exists() or force:
            shutil.copy2(logging_template, dest_file)
            click.echo(f"Created logging configuration: {dest_file}")


def _create_sample_transactions(file_path: Path, force: bool):
    """Create a sample transactions CSV file from template."""
    if file_path.exists() and not force:
        click.echo(f"Sample transactions file already exists: {file_path}")
        return

    # Get template file path
    templates_source = Path(__file__).parent.parent / "templates"
    sample_template = templates_source / "sample_transactions.csv"

    if sample_template.exists():
        # Copy template to destination
        shutil.copy2(sample_template, file_path)
        click.echo(f"Created sample transactions: {file_path}")
    else:
        # Fallback if template not found
        click.echo(f"WARNING: Sample template not found at {sample_template}")
        click.echo("Creating minimal sample file")

        minimal_sample = """date,symbol,transaction_type,quantity,value_per_share,account
2023-01-15,AAPL,BUY,100,150.50,Default
2023-02-15,SPY,BUY,50,380.00,Default
"""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(minimal_sample)
        click.echo(f"Created minimal sample transactions: {file_path}")

    click.echo("Sample includes transactions across multiple accounts:")
    click.echo("   - Schwab_401k (401k retirement account)")
    click.echo("   - Fidelity_IRA (IRA retirement account)")
    click.echo("   - Personal_Brokerage (taxable brokerage account)")


def _create_sample_attributions(file_path: Path, force: bool):
    """Create a sample attributions CSV file from template."""
    if file_path.exists() and not force:
        click.echo(f"Sample attributions file already exists: {file_path}")
        return

    # Get template file path
    templates_source = Path(__file__).parent.parent / "templates"
    sample_template = templates_source / "sample_attributions.csv"

    if sample_template.exists():
        # Copy template to destination
        shutil.copy2(sample_template, file_path)
        click.echo(f"Created sample attributions: {file_path}")
    else:
        # Fallback if template not found
        click.echo(
            f"WARNING: Sample attributions template not found at {sample_template}"
        )
        click.echo("Creating minimal sample file")

        minimal_sample = """symbol,effective_date,asset_class,geography,sector,fund_type
AAPL,2023-01-01,Equity,US,Technology,Stock
SPY,2023-01-01,Equity,US,Diversified,ETF
"""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(minimal_sample)
        click.echo(f"Created minimal sample attributions: {file_path}")

    click.echo(
        "Sample includes attributes for symbols used in sample_transactions.csv"
    )


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
@click.option("--benchmark", help="Override benchmark symbol (e.g., SPY, VTI)")
@click.option(
    "--profile",
    is_flag=True,
    default=False,
    help="Enable performance profiling"
)
def run_analysis(
    config: str, output_format: str, create_charts: bool, benchmark: str, profile: bool
):
    """Run BogleBench portfolio analysis."""

    # Set workspace context early
    if config:
        config_path = Path(config).expanduser()
        if config_path.exists():
            _ = WorkspaceContext.discover_workspace(config_path.parent)

    # Now initialize logging (will use correct workspace)
    setup_logging()
    logger = get_logger("cli.analyze")

    logger.info("🚀 Running BogleBench portfolio analysis...")
    logger.info(
        "📈 Analyzing your portfolio with Bogle's principles in mind..."
    )
    
    if profile:
        logger.info("🔍 Profiling enabled - performance stats will be saved")

    try:
        analyzer = BogleBenchAnalyzer(config_path=config)
        
        # Enable profiling if requested
        analyzer.profiling_enabled = profile

        # Override benchmark if specified
        if benchmark:
            analyzer.config.config["settings"]["benchmark_symbol"] = benchmark
            logger.info("📊 Using custom benchmark: %s", benchmark)

        logger.debug("Loading transaction data...")
        analyzer.load_transactions()

        logger.debug("Building portfolio history...")
        analyzer.build_portfolio_history()

        logger.debug("Loading symbol attributes...")
        analyzer.build_symbol_attributes()

        logger.debug("Calculating performance metrics...")
        results = analyzer.calculate_performance()

        # Display results
        logger.info("\n%s", results.summary())

        # Export results
        output_dir = analyzer.config.get_output_path()
        results.export_to_csv(str(output_dir))

        logger.debug(
            "Exporting analysis report with format: %s", output_format
        )

        # Create charts if requested
        if create_charts:
            logger.info("📊 Creating performance charts...")
            charts = BogleBenchCharts(results)

            # Create dashboard
            chart_path = Path(output_dir) / "performance_dashboard.png"
            charts.create_performance_dashboard(str(chart_path))

            # Create account comparison if multiple accounts
            account_summary = results.get_account_summary()
            if len(account_summary) > 1:
                account_chart_path = output_dir / "account_comparison.png"
                charts.create_account_comparison(str(account_chart_path))

        logger.info("✅ Analysis complete! Results saved to %s", output_dir)

    except ValueError as e:
        logger.error("❌ Error running analysis: %s", e)
        raise click.Abort()
    # except AttributeError as e:
    #     logger.error("❌ Error running analysis: %s", e)
    #     raise click.Abort()


@click.command()
@click.option("--config", help="Path to configuration file")
@click.option("--account", help="Show holdings for specific account only")
def show_holdings(config: str, account: str):
    """Show current portfolio holdings."""
    try:
        analyzer = BogleBenchAnalyzer(config_path=config)
        analyzer.load_transactions()
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
                f"{holding['symbol']:6} | "
                f"{holding['account']:15} | "
                f"{holding['shares']:>8.2f} shares | "
                f"${holding['price']:>8.2f} | "
                f"${holding['value']:>10,.2f} | "
                f"{holding['weight']:>6.1%}"
            )

        total_value = holdings["value"].sum()
        click.echo("=" * 80)
        click.echo(f"Total Value: ${total_value:,.2f}")

    except Exception as e:  # pylint: disable=broad-except
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
    logger.info("🔍 Validating transaction file: %s", path)

    try:
        click.echo(f"🔍 Validating transaction file: {path}")

        analyzer = BogleBenchAnalyzer()
        transactions = analyzer.load_transactions(path)

        click.echo("✅ Transaction file is valid!")
        click.echo(f"   📊 {len(transactions)} transactions")
        click.echo(f"   🏦 {transactions['account'].nunique()} accounts")
        click.echo(f"   📈 {transactions['symbol'].nunique()} assets")
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
