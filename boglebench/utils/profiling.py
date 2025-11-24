"""
Performance profiling utilities.

Provides decorators and context managers for profiling code execution
and identifying performance bottlenecks.
"""

import cProfile
import functools
import pstats
import time
from io import StringIO
from pathlib import Path
from typing import Optional

from .logging_config import get_logger

logger = get_logger(__name__)


class Profiler:
    """Context manager for profiling code blocks."""

    def __init__(
        self, name: str = "profile", output_file: Optional[Path] = None
    ):
        """
        Initialize profiler.

        Args:
            name: Name for the profiling session
            output_file: Optional file path to save profile stats
        """
        self.name = name
        self.output_file = output_file
        self.profiler = cProfile.Profile()

    def __enter__(self):
        """Start profiling."""
        logger.info("🔍 Starting profiling: %s", self.name)
        self.profiler.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop profiling and print stats."""
        self.profiler.disable()

        # Create stats
        stream = StringIO()
        stats = pstats.Stats(self.profiler, stream=stream)
        stats.sort_stats("cumulative")

        # Print top 20 functions
        logger.info("📊 Profile results for %s:", self.name)
        stats.print_stats(20)

        # Log the output
        logger.info("\n%s", stream.getvalue())

        # Save to file if requested
        if self.output_file:
            stats.dump_stats(str(self.output_file))
            logger.info("💾 Profile saved to: %s", self.output_file)


def profile_function(func):
    """
    Decorator to profile a function's execution.

    Usage:
        @profile_function
        def my_slow_function():
            ...
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()

        start_time = time.time()
        result = profiler.runcall(func, *args, **kwargs)
        elapsed = time.time() - start_time

        profiler.disable()

        # Print stats
        stream = StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats("cumulative")

        logger.info("⏱️  %s completed in %.2fs", func.__name__, elapsed)
        logger.info("📊 Top 10 functions in %s:", func.__name__)
        stats.print_stats(10)
        logger.info("\n%s", stream.getvalue())

        return result

    return wrapper
