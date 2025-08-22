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

    click.echo("\n✅ BogleBench workspace initialized successfully!")
    click.echo("\nNext steps:")
    click.echo(f"1. Edit your configuration: {config_path}")
    msg = f"2. Add your transaction data to: {workspace_path}/transactions/"
    click.echo(msg)
    click.echo(f"3. Run analysis: boglebench-analyze --config {config_path}")
    msg = "\n💡 Remember Bogle's wisdom: 'Stay the course' "
    msg = msg + "and focus on long-term results!"
    click.echo(msg)


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

    sample_data = """date,ticker,transaction_type,shares,price_per_share
2023-01-15,AAPL,BUY,100,150.50
2023-01-15,MSFT,BUY,50,240.25
2023-02-15,AAPL,BUY,50,155.75
2023-03-15,SPY,BUY,25,380.00
2023-04-15,AAPL,SELL,25,165.25
2023-05-15,GOOGL,BUY,10,105.50
"""

    with open(file_path, "w") as f:
        f.write(sample_data)

    click.echo(f"Created sample transactions: {file_path}")


@click.command()
@click.option("--config", help="Path to configuration file")
@click.option(
    "--output-format",
    default="jupyter",
    type=click.Choice(["jupyter", "html", "pdf"]),
    help="Output format for analysis",
)
def run_analysis(config: str, output_format: str):
    """Run BogleBench portfolio analysis."""
    click.echo("🚀 Running BogleBench portfolio analysis...")
    click.echo("📈 Analyzing your portfolio with Bogle's principles in mind...")

    # This would be implemented in your main analyzer
    from ..core.portfolio import BogleBenchAnalyzer

    try:
        analyzer = BogleBenchAnalyzer(config_path=config)

        click.echo("Loading transaction data...")
        analyzer.load_transactions()

        click.echo("Fetching market data...")
        analyzer.fetch_market_data()

        click.echo("Calculating performance metrics...")
        results = analyzer.calculate_performance()

        click.echo(f"Analysis complete! Results saved to output directory.")

    except Exception as e:
        click.echo(f"❌ Error running analysis: {e}")
        raise click.Abort()


if __name__ == "__main__":
    init_workspace()
