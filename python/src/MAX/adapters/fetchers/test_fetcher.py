from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Generic, Optional, TypeVar, List
import asyncio
import aiohttp
from enum import Enum
import logging

# Type definitions
T = TypeVar('T')

class FetchStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"

@dataclass
class FetchResult(Generic[T]):
    status: FetchStatus
    data: Optional[T] = None
    error: Optional[str] = None
    retry_after: Optional[float] = None
    metadata: Dict[str, any] = None

class RateLimiter:
    def __init__(self, requests_per_second: float):
        self.rate = requests_per_second
        self.tokens = requests_per_second
        self.last_update = asyncio.get_event_loop().time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            time_passed = now - self.last_update
            self.tokens = min(self.rate, self.tokens + time_passed * self.rate)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

class AbstractFetcher(ABC, Generic[T]):
    """Simple abstract base class for data fetching with rate limiting."""
    
    def __init__(
        self,
        base_url: str,
        requests_per_second: float = 1.0,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.base_url = base_url.rstrip('/')
        self.rate_limiter = RateLimiter(requests_per_second)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Basic statistics
        self.stats = {
            'requests': 0,
            'successes': 0,
            'failures': 0,
            'rate_limits': 0
        }

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    async def connect(self):
        """Initialize connection."""
        if not self.session:
            self.session = aiohttp.ClientSession(
                headers=self.default_headers
            )

    async def disconnect(self):
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

    @abstractmethod
    async def handle_error(self, error: Exception, context: Dict[str, any]) -> FetchResult[T]:
        """Handle specific error cases."""
        pass

    async def fetch(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        **kwargs
    ) -> FetchResult[T]:
        """Main fetch method with retry logic."""
        if not self.session:
            await self.connect()

        self.stats['requests'] += 1
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        for attempt in range(self.max_retries + 1):
            try:
                if not await self.rate_limiter.acquire():
                    self.stats['rate_limits'] += 1
                    return FetchResult(
                        status=FetchStatus.RATE_LIMITED,
                        retry_after=1/self.rate_limiter.rate
                    )

                async with self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    headers={**self.default_headers, **(headers or {})},
                    **kwargs
                ) as response:
                    
                    if response.status == 429:
                        self.stats['rate_limits'] += 1
                        retry_after = float(response.headers.get('Retry-After', self.retry_delay))
                        await asyncio.sleep(retry_after)
                        continue

                    if response.status >= 500 and attempt < self.max_retries:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                        continue

                    if response.status < 400:
                        self.stats['successes'] += 1
                        data = await self.process_response(response)
                        return FetchResult(
                            status=FetchStatus.SUCCESS,
                            data=data,
                            metadata={'headers': dict(response.headers)}
                        )
                    
                    self.stats['failures'] += 1
                    return FetchResult(
                        status=FetchStatus.FAILED,
                        error=f"HTTP {response.status}"
                    )

            except Exception as e:
                context = {
                    'url': url,
                    'method': method,
                    'attempt': attempt,
                    'params': params
                }
                return await self.handle_error(e, context)

        self.stats['failures'] += 1
        return FetchResult(
            status=FetchStatus.FAILED,
            error=f"Max retries ({self.max_retries}) exceeded"
        )

    async def bulk_fetch(
        self,
        requests: List[Dict],
        concurrency: int = 5
    ) -> List[FetchResult[T]]:
        """Execute multiple fetch operations concurrently."""
        semaphore = asyncio.Semaphore(concurrency)
        
        async def fetch_with_semaphore(request: Dict) -> FetchResult[T]:
            async with semaphore:
                return await self.fetch(**request)

        return await asyncio.gather(*[
            fetch_with_semaphore(req) for req in requests
        ])