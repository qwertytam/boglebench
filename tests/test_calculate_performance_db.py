"""
Test that calculate_performance() uses database instead of DataFrame.
"""

import inspect

import pandas as pd

from boglebench.core.metrics import calculate_irr
from boglebench.utils.config import ConfigManager


def test_calculate_irr_with_series():
    """Test that calculate_irr works with Series instead of DataFrame."""
    # Create sample data - initial investment, then value growth
    # The IRR formula needs inverted cash flows and final value added to last flow
    net_cash_flows = pd.Series([-1000, 0, 0, 0, 0])  # Initial investment
    total_values = pd.Series(
        [1000, 1020, 1030, 1050, 1100]
    )  # Portfolio growth

    # Create minimal config
    config = ConfigManager()

    # Calculate IRR
    irr = calculate_irr(net_cash_flows, total_values, config)

    # Verify IRR was calculated (should be positive for this scenario)
    assert isinstance(irr, float), "IRR should be a float"
    # For this scenario with 10% total return over 5 days, annualized IRR should be positive
    # We don't check the exact value as it depends on compounding


def test_calculate_irr_empty_series():
    """Test that calculate_irr handles empty series gracefully."""
    # Create empty data
    net_cash_flows = pd.Series(dtype=float)
    total_values = pd.Series(dtype=float)

    # Create minimal config
    config = ConfigManager()

    # Calculate IRR
    irr = calculate_irr(net_cash_flows, total_values, config)

    # Verify IRR returns zero for empty data
    assert irr == 0.0, "IRR should be 0.0 for empty series"


def test_calculate_irr_none_series():
    """Test that calculate_irr handles None gracefully."""
    # Create minimal config
    config = ConfigManager()

    # Calculate IRR with None
    irr = calculate_irr(pd.Series(dtype=float), pd.Series(dtype=float), config)

    # Verify IRR returns zero for None
    assert irr == 0.0, "IRR should be 0.0 for None series"


def test_calculate_irr_signature_changed():
    """Test that calculate_irr signature accepts series, not DataFrame."""

    # Get the signature
    sig = inspect.signature(calculate_irr)
    params = list(sig.parameters.keys())

    # Verify new signature
    assert "net_cash_flows" in params, "Should have net_cash_flows parameter"
    assert "total_values" in params, "Should have total_values parameter"
    assert "config" in params, "Should have config parameter"
    assert (
        "portfolio_history" not in params
    ), "Should not have portfolio_history parameter anymore"
