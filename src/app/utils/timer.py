# Standard Library
import asyncio
import logging
import time
from contextlib import contextmanager
from functools import wraps


def wrap_timer(name: str, logger: logging.Logger):
    def timeit(func):
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def timeit_wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = await func(*args, **kwargs)
                total = time.perf_counter() - start
                logger.info(f"{name} took {total:.4f} seconds")
                return result

        else:

            @wraps(func)
            def timeit_wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                total = time.perf_counter() - start
                logger.info(f"{name} took {total:.4f} seconds")
                return result

        return timeit_wrapper

    return timeit


@contextmanager
def ctx_timer(name: str, logger: logging.Logger) -> None:
    start = time.perf_counter()
    yield
    total = time.perf_counter() - start
    logger.info(f"{name} took {total:.4f} seconds")
