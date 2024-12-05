# This is base fetcher class for all fetchers
from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    AsyncGenerator,
    AsyncIterable,
    Dict,
    Generic,
    Literal,
    Optional,
    Protocol,
    TypeVar,
    Union,
    List,
    Final,
    runtime_checkable,
)
from urllib.parse import urljoin

import aiohttp
from pydantic import HttpUrl, PositiveFloat, PositiveInt
import backoff
from prometheus_client import Counter, Histogram
from opentelemetry import trace
from cachetools import TTLCache
from circuitbreaker import circuit
from pydantic_settings import BaseSettings

# Type definitions
T = TypeVar("T", covariant=True)
HTTPMethod = Literal["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"]

class FetcherError(Exception):
    """Base exception for fetcher errors."""
    pass

class RateLimitError(FetcherError):
    """Raised when rate limit is exceeded."""
    def __init__(self, retry_after: float):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after} seconds.")

class FetchError(FetcherError):
    """Raised when fetch operation fails."""
    pass

class ResponseParseError(FetcherError):
    """Raised when response parsing fails."""
    pass

class FetchStatus(Enum):
    SUCCESS = auto()
    FAILED = auto()
    RATE_LIMITED = auto()
    CACHED = auto()

@dataclass(frozen=True)
class FetchResult(Generic[T]):
    """Immutable result of a fetch operation."""
    status: FetchStatus
    data: Optional[T] = None
    error: Optional[Exception] = None
    retry_after: Optional[float] = None
    metadata: Dict[str, any] = field(default_factory=dict)

class FetcherConfig(BaseSettings):
    """Configuration with validation and environment variable support."""
    base_url: HttpUrl
    requests_per_second: PositiveFloat = 1.0
    max_retries: PositiveInt = 3
    retry_delay: PositiveFloat = 1.0
    timeout: PositiveFloat = 30.0
    cache_ttl: PositiveInt = 300  # 5 minutes
    circuit_breaker_threshold: PositiveInt = 5
    circuit_breaker_timeout: PositiveInt = 60

    class Config:
        env_prefix = "FETCHER_"

@runtime_checkable
class ResponseProcessor(Protocol[T]):
    """Protocol for response processors."""
    async def process(self, response: aiohttp.ClientResponse) -> T:
        ...

class Metrics:
    """Prometheus metrics."""
    REQUESTS = Counter(
        "fetcher_requests_total",
        "Total number of requests",
        ["method", "status"]
    )
    LATENCY = Histogram(
        "fetcher_request_duration_seconds",
        "Request duration in seconds",
        ["method"]
    )

class RateLimiter:
    """Token bucket rate limiter with precise timing."""
    def __init__(self, requests_per_second: float):
        self.rate: Final[float] = requests_per_second
        self.tokens: float = requests_per_second
        self.last_update: float = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        async with self._lock:
            now = time.monotonic()
            time_passed = now - self.last_update
            self.tokens = min(self.rate, self.tokens + time_passed * self.rate)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

class AbstractFetcher(ABC, Generic[T]):
    """Enhanced abstract base class for data fetching with built-in rate limiting,
    caching, circuit breaking, and telemetry.
    
    Generic type T represents the expected response data type after processing.
    """

    def __init__(self, config: FetcherConfig):
        """Initialize fetcher with configuration.
        
        Args:
            config: FetcherConfig instance containing all necessary settings
        """
        self.config = config
         # Rate limiter ensures we don't exceed configured requests per second
        self.rate_limiter = RateLimiter(config.requests_per_second)
        # Session is initialized lazily in connect()
        self.session: Optional[aiohttp.ClientSession] = None
                # In-memory cache with TTL for GET requests
        self.cache: TTLCache = TTLCache(maxsize=100, ttl=config.cache_ttl)
        # OpenTelemetry tracer for request tracking
        self.tracer = trace.get_tracer(__name__)

    async def __aenter__(self) -> AbstractFetcher[T]:
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()

    async def connect(self) -> None:
        """Initialize connection with timeout configuration."""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self.default_headers
            )

    async def disconnect(self) -> None:
        """Clean up resources properly."""
        if self.session:
            await self.session.close()
            self.session = None

    @property
    @abstractmethod
    def default_headers(self) -> Dict[str, str]:
        """Default headers for requests."""
        pass

    @abstractmethod
    async def process_response(self, response: aiohttp.ClientResponse) -> T:
        """Process the raw response into the desired format."""
        pass

    @abstractmethod
    async def handle_error(
        self, 
        error: Exception,
        context: Dict[str, any]
    ) -> FetchResult[T]:
        """Make an HTTP request with automatic retries, caching, and instrumentation.
        
        Args:
            endpoint: The API endpoint to call (will be joined with base_url)
            method: HTTP method to use
            params: Optional query parameters
            data: Optional request body (will be JSON-encoded)
            headers: Optional additional headers
            use_cache: Whether to use cache for GET requests
            **kwargs: Additional arguments passed to aiohttp.ClientSession.request()
            
        Returns:
            FetchResult containing status and processed data or error information
            
        Raises:
            RateLimitError: When rate limit is exceeded
            CircuitBreakerError: When circuit breaker is open
        """
        pass

    def _get_cache_key(self, method: str, url: str, params: Optional[Dict] = None) -> str:
        """Generate cache key for request."""
        return f"{method}:{url}:{str(sorted(params.items()) if params else '')}"

    async def _should_cache(
        self,
        method: HTTPMethod,
        endpoint: str,
        params: Optional[Dict] = None
    ) -> bool:
        """Determine if request should be cached."""
        return method == "GET" and not params

    @circuit(
        failure_threshold=lambda self: self.config.circuit_breaker_threshold,
        recovery_timeout=lambda self: self.config.circuit_breaker_timeout
    )
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, RateLimitError),
        max_tries=lambda self: self.config.max_retries
    )
    async def fetch(
        self,
        endpoint: str,
        method: HTTPMethod = "GET",
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        use_cache: bool = True,
        **kwargs
    ) -> FetchResult[T]:
        """Enhanced fetch with caching, circuit breaker, and proper instrumentation."""
        
        if not self.session:
            await self.connect()

        url = urljoin(str(self.config.base_url), endpoint.lstrip('/'))
        cache_key = self._get_cache_key(method, url, params)

        # Check cache
        if use_cache and await self._should_cache(method, endpoint, params):
            if cached := self.cache.get(cache_key):
                return FetchResult(status=FetchStatus.CACHED, data=cached)

        # Rate limiting check
        if not await self.rate_limiter.acquire():
            raise RateLimitError(1/self.rate_limiter.rate)

        # Tracing
        with self.tracer.start_as_current_span(
            f"fetch_{method.lower()}",
            attributes={
                "http.method": method,
                "http.url": url,
            }
        ) as span:
            try:
                async with Metrics.LATENCY.time():
                    async with self.session.request(
                        method=method,
                        url=url,
                        params=params,
                        json=data,
                        headers={**self.default_headers, **(headers or {})},
                        **kwargs
                    ) as response:
                        
                        if response.status == 429:
                            retry_after = float(response.headers.get('Retry-After', self.config.retry_delay))
                            raise RateLimitError(retry_after)

                        response.raise_for_status()
                        processed_data = await self.process_response(response)

                        # Update cache if applicable
                        if use_cache and await self._should_cache(method, endpoint, params):
                            self.cache[cache_key] = processed_data

                        Metrics.REQUESTS.labels(method=method, status="success").inc()
                        return FetchResult(
                            status=FetchStatus.SUCCESS,
                            data=processed_data,
                            metadata={'headers': dict(response.headers)}
                        )

            except Exception as e:
                Metrics.REQUESTS.labels(method=method, status="error").inc()
                span.record_exception(e)
                return await self.handle_error(e, {
                    'url': url,
                    'method': method,
                    'params': params,
                    'headers': headers
                })

    async def stream(
        self,
        endpoint: str,
        method: HTTPMethod = "GET",
        **kwargs
    ) -> AsyncGenerator[T, None]:
        """Stream responses for large datasets."""
        async for result in self._stream_helper(endpoint, method, **kwargs):
            if result.status == FetchStatus.SUCCESS and result.data is not None:
                yield result.data

    async def _stream_helper(
        self,
        endpoint: str,
        method: HTTPMethod,
        **kwargs
    ) -> AsyncGenerator[FetchResult[T], None]:
        """Helper method for streaming with proper error handling."""
        if not self.session:
            await self.connect()

        url = urljoin(str(self.config.base_url), endpoint.lstrip('/'))

        try:
            async with self.session.request(
                method=method,
                url=url,
                **kwargs
            ) as response:
                async for chunk in response.content.iter_chunked(8192):
                    try:
                        processed = await self.process_response(
                            aiohttp.ClientResponse(
                                method,
                                url,
                                writer=None,
                                continue100=None,
                                timer=None,
                                request_info=None,
                                traces=None,
                                loop=asyncio.get_event_loop(),
                                session=self.session
                            )
                        )
                        yield FetchResult(status=FetchStatus.SUCCESS, data=processed)
                    except Exception as e:
                        yield FetchResult(status=FetchStatus.FAILED, error=e)

        except Exception as e:
            yield FetchResult(status=FetchStatus.FAILED, error=e)

    async def bulk_fetch(
        self,
        requests: List[Dict],
        concurrency: int = 5,
        preserve_order: bool = True
    ) -> List[FetchResult[T]]:
        """Enhanced bulk fetch with order preservation option."""
        semaphore = asyncio.Semaphore(concurrency)
        
        async def fetch_with_semaphore(index: int, request: Dict) -> tuple[int, FetchResult[T]]:
            async with semaphore:
                result = await self.fetch(**request)
                return index, result

        tasks = [
            fetch_with_semaphore(i, req) 
            for i, req in enumerate(requests)
        ]
        
        results = await asyncio.gather(*tasks)
        
        if preserve_order:
            return [result for _, result in sorted(results, key=lambda x: x[0])]
        return [result for _, result in results]

    @property
    def stats(self) -> Dict[str, any]:
        """Comprehensive statistics including cache and circuit breaker."""
        return {
            'cache_info': {
                'size': len(self.cache),
                'maxsize': self.cache.maxsize,
                'hits': self.cache.hits,
                'misses': self.cache.misses,
            },
            'rate_limit_info': {
                'current_tokens': self.rate_limiter.tokens,
                'rate': self.rate_limiter.rate,
            },
        }