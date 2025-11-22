"""Performance monitoring utilities."""

import functools
import time
from typing import Callable

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


def timeit(func: Callable) -> Callable:
    """
    Decorator to measure and log function execution time.
    
    Usage:
        @timeit
        def my_function():
            # code here
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        logger.debug(
            "⏱️  %s completed in %.2f seconds",
            func.__name__,
            elapsed
        )
        
        return result
    
    return wrapper
