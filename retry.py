"""Retry Decorator Module

This module provides a decorator for implementing retry logic with exponential backoff.
It's useful for handling transient failures in network calls, database operations,
or any other operations that might temporarily fail.

Example:
    @retry_with_backoff(retries=3, exceptions=(RequestException,))
    def make_api_call():
        return requests.get('https://api.example.com/data')

The decorated function will retry up to 3 times with exponential backoff
if a RequestException occurs.
"""

import time
from functools import wraps

def retry_with_backoff(retries: int, exceptions: tuple, backoff_in_seconds: int = 1):
    """Decorator that implements retry logic with exponential backoff.

    This decorator will retry the decorated function if it raises any of the specified
    exceptions. The wait time between retries increases exponentially, following
    the formula: backoff_time = backoff_in_seconds * (2 ^ (attempt - 1))

    Args:
        retries: Maximum number of retry attempts before giving up.
        exceptions: Tuple of exception classes that should trigger a retry.
        backoff_in_seconds: Initial backoff time in seconds (default: 1).
                          This value is doubled after each retry attempt.

    Returns:
        function: Decorated function that implements retry logic.

    Raises:
        The last exception encountered after all retry attempts are exhausted.

    Example:
        >>> @retry_with_backoff(retries=3, exceptions=(ValueError,))
        ... def unstable_function():
        ...     # Function that might raise ValueError
        ...     pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    # Calculate exponential backoff time
                    backoff_time = backoff_in_seconds * (2 ** (attempt - 1))
                    time.sleep(backoff_time)
                    # If this was the last attempt, re-raise the exception
                    if attempt == retries:
                        raise e
        return wrapper
    return decorator
