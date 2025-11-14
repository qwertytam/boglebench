"""
Timing utilities for performance measurement.

This module provides decorators and context managers for measuring and logging
execution time of operations. Useful for profiling and optimization of portfolio
analysis workflows.
"""

import functools
import logging
import time
from contextlib import contextmanager
from typing import Optional

from .logging_config import get_logger


@contextmanager
def timed_operation(
    operation_name: str, logger: Optional[logging.Logger] = None
):
    """
    Context manager to time operations and log results.

    Args:
        operation_name: Description of the operation being timed
        logger: Logger instance. If None, uses module logger.
    """
    if logger is None:

        logger = get_logger("timing")

    start_time = time.perf_counter()
    logger.debug("Starting %s", operation_name)

    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        logger.debug("%s completed in %.4f seconds", operation_name, elapsed)


def time_function(logger: Optional[logging.Logger] = None):
    """Decorator to time function execution."""

    if logger is None:
        logger = get_logger("timing")

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with timed_operation(func.__name__, logger):
                return func(*args, **kwargs)

        return wrapper

    return decorator
