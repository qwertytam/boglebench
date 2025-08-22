"""
BogleBench - Portfolio performance analysis and benchmarking in the spirit of
John Bogle.

Named after John C. Bogle, founder of Vanguard and champion of low-cost index
investing. This package helps investors analyze their portfolio performance
against  benchmarks using the principles Bogle advocated: long-term investing,
low costs, and broad diversification.
"""

__version__ = "0.1.0"
__author__ = "Tom Marshall"
__email__ = "tom.q.marshall@gmail.com"

# Import main classes for easy access
from .core.portfolio import BogleBenchAnalyzer
from .utils.config import ConfigManager

# Define what gets imported with "from boglebench import *"
__all__ = [
    "BogleBenchAnalyzer",
    "ConfigManager",
]
