"""
Miscelaneous utilities.
"""

import inspect
import time
from functools import wraps
from typing import Callable

from app.logger import logger


def timed(func) -> Callable:
    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def timed_func(*args, **kwargs):
            init = time.perf_counter()
            out = await func(*args, **kwargs)
            end = time.perf_counter() - init
            logger.info(f"{func.__name__} finished in {1000 * end:.2f} ms")
            return out

    else:

        def timed_func(*args, **kwargs):
            init = time.perf_counter()
            out = func(*args, **kwargs)
            end = time.perf_counter() - init
            logger.info(f"{func.__name__} finished in {1000 * end:.2f} ms")
            return out

    return timed_func
