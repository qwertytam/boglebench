"""
Tests for parallel CSV export optimization.

Verifies that the parallelized export methods produce correct results
and that all files are exported successfully.
"""

import tempfile

import pandas as pd
import pytest

from boglebench.core.results import PerformanceResults
from boglebench.utils.config import ConfigManager


@pytest.fixture
def simple_results():
    """Create simple performance results without database."""

    # Create mock portfolio metrics
    portfolio_metrics = {
        "mod_dietz": {
            "total_return": 0.15,
            "annualized_return": 0.12,
            "volatility": 0.18,
        },
        "twr": {
            "total_return": 0.16,
            "annualized_return": 0.13,
            "volatility": 0.18,
        },
        "irr": {
            "annualized_return": 0.125,
        },
    }

    # Create mock holding attribution
    holding_attribution = pd.DataFrame(
        {
            "symbol": ["SPY", "VTI", "BND"],
            "contribution": [0.05, 0.08, 0.02],
            "weight": [0.40, 0.40, 0.20],
        }
    )

    # Create mock account attribution
    account_attribution = pd.DataFrame(
        {
            "account": ["Brokerage", "IRA"],
            "contribution": [0.08, 0.07],
            "weight": [0.60, 0.40],
        }
    )

    config = ConfigManager()

    results = PerformanceResults(
        portfolio_metrics=portfolio_metrics,
        holding_attribution=holding_attribution,
        account_attribution=account_attribution,
        portfolio_db=None,  # No database for simple test
        config=config,
    )

    return results


# pylint: disable=redefined-outer-name
def test_parallel_export_metrics_and_attribution(simple_results):
    """Test that parallel export creates metrics and attribution files."""

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = simple_results.export_to_csv(output_dir=tmpdir)

        # Check that the path exists
        assert output_path.exists()
        assert output_path.is_dir()

        # List all CSV files created
        csv_files = list(output_path.glob("*.csv"))

        # We should have at least 3 files (metrics, holding_attribution, account_attribution)
        assert len(csv_files) >= 3

        # Check for specific files
        file_patterns = [
            "metrics",
            "holding_attribution",
            "account_attribution",
        ]

        for pattern in file_patterns:
            matching_files = [f for f in csv_files if pattern in f.name]
            assert (
                len(matching_files) > 0
            ), f"Missing file with pattern: {pattern}"


def test_parallel_export_metrics_content(simple_results):
    """Test that exported metrics file has correct content."""

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = simple_results.export_to_csv(output_dir=tmpdir)

        # Find metrics file
        metrics_files = list(output_path.glob("*metrics*.csv"))
        assert len(metrics_files) == 1

        # Read and verify content
        metrics_df = pd.read_csv(metrics_files[0])

        # Check columns
        assert "method" in metrics_df.columns
        assert "metric" in metrics_df.columns
        assert "value" in metrics_df.columns

        # Check that we have metrics for all methods
        methods = metrics_df["method"].unique()
        assert "mod_dietz" in methods
        assert "twr" in methods
        assert "irr" in methods

        # Check specific values
        total_return_md = metrics_df[
            (metrics_df["method"] == "mod_dietz")
            & (metrics_df["metric"] == "total_return")
        ]["value"].iloc[0]
        assert abs(total_return_md - 0.15) < 0.001


def test_parallel_export_attribution_content(simple_results):
    """Test that exported attribution files have correct content."""

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = simple_results.export_to_csv(output_dir=tmpdir)

        # Find holding attribution file
        holding_files = list(output_path.glob("*holding_attribution*.csv"))
        assert len(holding_files) == 1

        # Read and verify content
        holding_df = pd.read_csv(holding_files[0], index_col=0)

        # Check that we have data
        assert len(holding_df) == 3  # SPY, VTI, BND

        # Check expected columns
        assert "symbol" in holding_df.columns
        assert "contribution" in holding_df.columns

        # Find account attribution file
        account_files = list(output_path.glob("*account_attribution*.csv"))
        assert len(account_files) == 1

        # Read and verify content
        account_df = pd.read_csv(account_files[0], index_col=0)

        # Check that we have data
        assert len(account_df) == 2  # Brokerage, IRA


def test_parallel_export_empty_attribution():
    """Test parallel export handles missing attribution gracefully."""

    # Create results without attribution
    results = PerformanceResults(
        portfolio_metrics={
            "mod_dietz": {"total_return": 0.10},
        },
        holding_attribution=None,  # No attribution
        account_attribution=pd.DataFrame(),  # Empty attribution
        portfolio_db=None,
        config=ConfigManager(),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Should not raise an error
        output_path = results.export_to_csv(output_dir=tmpdir)

        # Should still create metrics file
        csv_files = list(output_path.glob("*.csv"))
        assert len(csv_files) >= 1

        # Should have metrics file
        metrics_files = [f for f in csv_files if "metrics" in f.name]
        assert len(metrics_files) == 1

        # Should not have attribution files
        attribution_files = [f for f in csv_files if "attribution" in f.name]
        assert len(attribution_files) == 0


def test_parallel_export_thread_safety():
    """Test that parallel export doesn't cause data corruption."""

    # Create results with multiple dataframes
    results = PerformanceResults(
        portfolio_metrics={
            "mod_dietz": {f"metric_{i}": float(i) for i in range(10)},
            "twr": {f"metric_{i}": float(i * 2) for i in range(10)},
        },
        holding_attribution=pd.DataFrame(
            {
                "symbol": [f"SYM_{i}" for i in range(100)],
                "contribution": [float(i) / 100 for i in range(100)],
            }
        ),
        account_attribution=pd.DataFrame(
            {
                "account": [f"ACC_{i}" for i in range(50)],
                "contribution": [float(i) / 50 for i in range(50)],
            }
        ),
        portfolio_db=None,
        config=ConfigManager(),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = results.export_to_csv(output_dir=tmpdir)

        # Verify all files are complete
        csv_files = list(output_path.glob("*.csv"))
        assert len(csv_files) == 3  # metrics, holding, account

        # Verify each file has correct row counts
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            assert len(df) > 0, f"File {csv_file.name} is empty"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
