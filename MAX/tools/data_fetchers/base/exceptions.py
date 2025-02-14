class FetcherError(Exception):
    """Base exception for fetcher errors."""

    pass


class RateLimitError(FetcherError):
    """Raised when rate limit is exceeded."""

    def __init__(self, retry_after: float):
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded. Retry after {retry_after} seconds."
        )


class FetchError(FetcherError):
    """Raised when fetch operation fails."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"Fetch failed with status {status_code}: {message}")


class ResponseParseError(FetcherError):
    """Raised when response parsing fails."""

    pass


class CircuitBreakerError(FetcherError):
    """Raised when circuit breaker is open."""

    def __init__(self, recovery_time: float):
        self.recovery_time = recovery_time
        super().__init__(
            f"Circuit breaker is open. Recovery in {recovery_time} seconds."
        )
