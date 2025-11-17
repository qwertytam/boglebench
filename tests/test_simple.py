"""
Simple tests to verify BogleBench setup.
"""


def test_imports():
    """Test that we can import the main modules."""
    # pylint: disable=import-outside-toplevel
    from boglebench import BogleBenchAnalyzer, ConfigManager

    assert BogleBenchAnalyzer is not None
    assert ConfigManager is not None


def test_analyzer_creation():
    """Test that we can create an analyzer instance."""
    # pylint: disable=import-outside-toplevel
    from boglebench import BogleBenchAnalyzer

    analyzer = BogleBenchAnalyzer()
    assert analyzer is not None
    assert analyzer.config is not None
    assert analyzer.transactions.empty is True


def test_config_manager():
    """Test that ConfigManager works."""
    # pylint: disable=import-outside-toplevel
    from boglebench.utils.config import ConfigManager

    config = ConfigManager()
    assert config is not None
    assert config.config is not None

    # Test getting a default value
    base_path = config.get("data.base_path")
    assert base_path is not None
