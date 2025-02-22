import asyncio
from functools import wraps
from typing import TypeVar, Callable, Any
from MAX.utils import Logger
from .exceptions import LLMRateLimitError, LLMProviderError

T = TypeVar("T")

logger = Logger.get_logger(__name__)


def async_retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    max_delay: float = 30.0,
):
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            last_exception = None

            for retry in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except LLMRateLimitError as e:
                    last_exception = e
                    delay = min(delay * exponential_base, max_delay)
                    logger.warn(
                        f"Rate limit hit, attempt {retry + 1}/{max_retries}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                except LLMProviderError as e:
                    last_exception = e
                    if "timeout" in str(e).lower():
                        delay = min(delay * exponential_base, max_delay)
                        logger.warn(
                            f"Timeout error, attempt {retry + 1}/{max_retries}. "
                            f"Retrying in {delay:.2f} seconds..."
                        )
                    else:
                        raise  # Don't retry other provider errors

                await asyncio.sleep(delay)

            raise last_exception or LLMProviderError("Max retries exceeded")

        return wrapper

    return decorator
