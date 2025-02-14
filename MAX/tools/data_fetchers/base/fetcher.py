from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Generic, List, Optional, TypeVar, Any
from urllib.parse import urljoin

import aiohttp
import backoff
from cachetools import TTLCache
from circuitbreaker import circuit

from .types import HTTPMethod, FetchStatus, FetchResult, T
from .exceptions import RateLimitError, CircuitBreakerError
from ..monitoring.metrics import FetcherMetrics
from ..monitoring.health import HealthMonitor
from ..monitoring.telemetry import FetcherTelemetry


@dataclass
class FetcherConfig:
    """Configuration with validation."""

    base_url: str
    requests_per_second: float = 1.0
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    cache_ttl: int = 300
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    enable_telemetry: bool = True
    enable_metrics: bool = True


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, requests_per_second: float):
        self.rate = requests_per_second
        self.tokens = requests_per_second
        self.last_update = time.monotonic()
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
    """Enhanced base fetcher with comprehensive monitoring."""

    def __init__(self, config: FetcherConfig, name: str):
        self._name = name
        self.config = config
        self.rate_limiter = RateLimiter(config.requests_per_second)
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache = TTLCache(maxsize=100, ttl=config.cache_ttl)

        # Initialize monitoring components
        if config.enable_metrics:
            self._metrics = FetcherMetrics(name)
        if config.enable_telemetry:
            self.telemetry = FetcherTelemetry(name)

        # Register with health monitor
        self.health_monitor = HealthMonitor()
        self.health_monitor.register_component(self)

        # Initialize statistics
        self._reset_stats()

        # Apply circuit breaker at instance level
        self.fetch = circuit(
            failure_threshold=self.config.circuit_breaker_threshold,
            recovery_timeout=self.config.circuit_breaker_timeout,
            expected_exception=CircuitBreakerError,
        )(self.fetch)

    def _reset_stats(self) -> None:
        """Initialize/reset statistics."""
        self._stats = {
            "requests": 0,
            "successes": 0,
            "failures": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "rate_limits": 0,
            "circuit_breaks": 0,
            "total_latency": 0.0,
        }

    async def __aenter__(self) -> AbstractFetcher[T]:
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()

    async def connect(self) -> None:
        """Initialize connection with timeout."""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(
                timeout=timeout, headers=self.default_headers
            )

    async def disconnect(self) -> None:
        """Clean up resources."""
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

    async def handle_error(
        self, error: Exception, context: Dict[str, any]
    ) -> FetchResult[T]:
        """Enhanced error handling with monitoring."""
        error_type = type(error).__name__
        self._stats["failures"] += 1

        if isinstance(error, RateLimitError):
            self._stats["rate_limits"] += 1
            return FetchResult(
                status=FetchStatus.RATE_LIMITED,
                error=error,
                retry_after=error.retry_after,
            )

        if isinstance(error, CircuitBreakerError):
            self._stats["circuit_breaks"] += 1
            return FetchResult(status=FetchStatus.CIRCUIT_BROKEN, error=error)

        # Record error metrics
        if hasattr(self, "_metrics"):
            self._metrics.record_error(error_type, context["url"])

        return FetchResult(
            status=FetchStatus.FAILED,
            error=error,
            metadata={"error_type": error_type, "context": context},
        )

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, RateLimitError),
        max_tries=lambda self: self.config.max_retries,
    )
    async def fetch(
        self,
        endpoint: str,
        method: HTTPMethod = "GET",
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> FetchResult[T]:
        """Enhanced fetch with monitoring."""
        if not self.session:
            await self.connect()

        self._stats["requests"] += 1
        url = urljoin(str(self.config.base_url), endpoint.lstrip("/"))
        cache_key = (
            f"{method}:{url}:{str(sorted(params.items()) if params else '')}"
        )

        # Cache check
        if use_cache and method == "GET":
            if cached := self.cache.get(cache_key):
                self._stats["cache_hits"] += 1
                if hasattr(self, "_metrics"):
                    self._metrics.cache_stats.labels(
                        operation="get", result="hit"
                    ).inc()
                return FetchResult(status=FetchStatus.CACHED, data=cached)
            self._stats["cache_misses"] += 1

        # Rate limiting
        if not await self.rate_limiter.acquire():
            raise RateLimitError(1 / self.rate_limiter.rate)

        # Request execution with monitoring
        start_time = time.monotonic()
        span = None

        try:
            if hasattr(self, "telemetry"):
                span = self.telemetry.span(
                    f"fetch_{method.lower()}",
                    {
                        "http.method": method,
                        "http.url": url,
                        "http.params": str(params),
                    },
                )
            if span:
                span.start()

            async with self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers={**self.default_headers, **(headers or {})},
                **kwargs,
            ) as response:
                duration = time.monotonic() - start_time
                self._stats["total_latency"] += duration

                if hasattr(self, "_metrics"):
                    self._metrics.record_latency(method, endpoint, duration)

                if response.status == 429:
                    retry_after = float(
                        response.headers.get(
                            "Retry-After", self.config.retry_delay
                        )
                    )
                    raise RateLimitError(retry_after)

                response.raise_for_status()
                processed_data = await self.process_response(response)

                # Cache update
                if use_cache and method == "GET":
                    self.cache[cache_key] = processed_data

                self._stats["successes"] += 1
                if hasattr(self, "_metrics"):
                    self._metrics.record_request(method, "success", endpoint)

                return FetchResult(
                    status=FetchStatus.SUCCESS,
                    data=processed_data,
                    metadata={
                        "headers": dict(response.headers),
                        "duration": duration,
                    },
                )

        except Exception as e:
            if span and hasattr(self, "telemetry"):
                self.telemetry.record_exception(span, e)
            return await self.handle_error(
                e,
                {
                    "url": url,
                    "method": method,
                    "params": params,
                    "headers": headers,
                },
            )

        finally:
            if span:
                span.end()

    async def bulk_fetch(
        self,
        requests: List[Dict],
        concurrency: int = 5,
        preserve_order: bool = True,
    ) -> List[FetchResult[T]]:
        """Enhanced bulk fetch with monitoring."""
        semaphore = asyncio.Semaphore(concurrency)

        async def fetch_with_semaphore(
            index: int, request: Dict
        ) -> tuple[int, FetchResult[T]]:
            async with semaphore:
                if hasattr(self, "telemetry"):
                    with self.telemetry.span(
                        "bulk_fetch_item",
                        {"index": index, "request": str(request)},
                    ):
                        result = await self.fetch(**request)
                        return index, result
                else:
                    result = await self.fetch(**request)
                    return index, result

        tasks = [
            fetch_with_semaphore(i, req) for i, req in enumerate(requests)
        ]

        results = await asyncio.gather(*tasks)

        if preserve_order:
            return [
                result for _, result in sorted(results, key=lambda x: x[0])
            ]
        return [result for _, result in results]

    @property
    def health_stats(self) -> Dict[str, any]:
        """Get health-related statistics."""
        total_requests = self._stats["requests"]
        if total_requests == 0:
            return {
                "error_rate": 0.0,
                "success_rate": 0.0,
                "average_latency": 0.0,
                "cache_hit_rate": 0.0,
            }

        return {
            "error_rate": self._stats["failures"] / total_requests,
            "success_rate": self._stats["successes"] / total_requests,
            "average_latency": self._stats["total_latency"] / total_requests,
            "cache_hit_rate": (
                self._stats["cache_hits"]
                / (self._stats["cache_hits"] + self._stats["cache_misses"])
                if (self._stats["cache_hits"] + self._stats["cache_misses"])
                > 0
                else 0.0
            ),
            "rate_limit_rate": self._stats["rate_limits"] / total_requests,
            "circuit_break_rate": self._stats["circuit_breaks"]
            / total_requests,
        }

    @property
    def name(self) -> str:
        """Name of the monitorable component."""
        return self._name

    @property
    def stats(self) -> Dict[str, Any]:
        """Current statistics of the component."""
        return self._stats

    @property
    def metrics(self) -> Any:
        """Metrics collector for the component."""
        return self._metrics if hasattr(self, "_metrics") else None
