"""Retry utility with exponential backoff for transient API failures."""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Callable, TypeVar

logger = logging.getLogger("nope")

T = TypeVar("T")

# Exception types considered transient (worth retrying)
TRANSIENT_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    transient_exceptions: tuple = TRANSIENT_EXCEPTIONS,
) -> Callable:
    """Decorator that retries a function on transient failures with exponential backoff."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except transient_exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(
                            "%s failed after %d attempts: %s",
                            func.__name__, max_attempts, e,
                        )
                        raise
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    logger.warning(
                        "%s attempt %d/%d failed (%s: %s), retrying in %.1fs",
                        func.__name__, attempt, max_attempts,
                        type(e).__name__, e, delay,
                    )
                    time.sleep(delay)
            raise last_exception  # unreachable, but satisfies type checker

        return wrapper

    return decorator
