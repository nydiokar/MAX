import asyncio
import aiohttp
from typing import Optional, Dict, Any
import time

from MAX.adapters.fetchers.base.fetcher import AbstractFetcher, FetcherConfig
from MAX.adapters.fetchers.base.types import FetchResult, FetchStatus

class SimpleCacheFetcher(AbstractFetcher[dict]):
    """Simple fetcher implementation for testing cache behavior."""

    def __init__(self):
        config = FetcherConfig(
            base_url="https://api.example.com",
            requests_per_second=10,
            cache_ttl=5  # Short TTL for easier testing
        )
        super().__init__(config, "simple_cache_fetcher")
        self.cache = {}
        self.cache_timestamps = {}
        self._stats = {
            'requests': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

    @property
    def default_headers(self) -> dict:
        return {"Accept": "application/json"}

    async def process_response(self, response: aiohttp.ClientResponse) -> dict:
        """Process the response from the server"""
        return await response.json()

    def _is_cached(self, endpoint: str) -> bool:
        """Check if endpoint is cached and valid"""
        if endpoint not in self.cache_timestamps:
            return False
        age = time.time() - self.cache_timestamps[endpoint]
        return age < self.config.cache_ttl

    async def simulate_request(self, endpoint: str) -> FetchResult[dict]:
        """Simulate a request with caching"""
        self._stats['requests'] += 1

        # Check cache
        if self._is_cached(endpoint):
            self._stats['cache_hits'] += 1
            return FetchResult(
                status=FetchStatus.CACHED,
                data=self.cache[endpoint]
            )

        # Cache miss
        self._stats['cache_misses'] += 1
        
        # Simulate API request
        await asyncio.sleep(0.1)  # Simulate network delay
        
        # Store in cache
        response_data = {"message": "Success", "endpoint": endpoint}
        self.cache[endpoint] = response_data
        self.cache_timestamps[endpoint] = time.time()

        return FetchResult(
            status=FetchStatus.SUCCESS,
            data=response_data
        )

async def test_cache_behavior():
    """Test the caching behavior of the fetcher"""
    print("\n=== Testing Fetcher Cache Behavior ===")
    
    async with SimpleCacheFetcher() as fetcher:
        endpoint = "/api/test"
        
        print("\nTest 1: Initial request (should miss cache)")
        result1 = await fetcher.simulate_request(endpoint)
        print(f"Cache hit? {result1.status == FetchStatus.CACHED}")
        
        print("\nTest 2: Immediate second request (should hit cache)")
        result2 = await fetcher.simulate_request(endpoint)
        print(f"Cache hit? {result2.status == FetchStatus.CACHED}")
        
        print("\nTest 3: Waiting for cache expiration...")
        await asyncio.sleep(6)  # Wait for TTL to expire
        result3 = await fetcher.simulate_request(endpoint)
        print(f"Cache hit? {result3.status == FetchStatus.CACHED}")
        
        print("\n=== Cache Statistics ===")
        print(f"Total requests: {fetcher._stats['requests']}")
        print(f"Cache hits: {fetcher._stats['cache_hits']}")
        print(f"Cache misses: {fetcher._stats['cache_misses']}")

if __name__ == "__main__":
    asyncio.run(test_cache_behavior())