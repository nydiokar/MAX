import asyncio
import aiohttp
from datetime import datetime, timedelta
import random
from typing import Optional, Dict, Any, List
from contextlib import nullcontext
from enum import Enum, auto
import json
from pathlib import Path

from MAX.adapters.fetchers.base.fetcher import AbstractFetcher, FetcherConfig
from MAX.adapters.fetchers.base.types import FetchResult, FetchStatus
from MAX.adapters.fetchers.base.exceptions import RateLimitError

# Report-related enums
class ReportTimeframe(Enum):
    HOUR = auto()
    DAY = auto()

class ReportFormat(Enum):
    JSON = auto()
    TEXT = auto()
    HTML = auto()

class TestReporter:
    """Simple reporter implementation for tests."""
    def __init__(self, output_dir: str = "test_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    async def generate_summary_report(
        self,
        stats: Dict[str, Any],
        timeframe: ReportTimeframe,
        format: ReportFormat
    ) -> str:
        """Generate a report in the specified format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"report_{timeframe.name.lower()}_{timestamp}"

        # Prepare report data
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "timeframe": timeframe.name,
            "statistics": stats,
            "summary": {
                "total_requests": stats['requests'],
                "success_rate": (stats['successes'] / stats['requests'] * 100) if stats['requests'] > 0 else 0,
                "average_latency": stats['total_latency'] / stats['requests'] if stats['requests'] > 0 else 0,
                "error_rate": (stats['failures'] / stats['requests'] * 100) if stats['requests'] > 0 else 0
            }
        }

        try:
            if format == ReportFormat.JSON:
                filename = filename.with_suffix('.json')
                with open(filename, 'w') as f:
                    json.dump(report_data, f, indent=2)
                return str(filename)

            elif format == ReportFormat.TEXT:
                filename = filename.with_suffix('.txt')
                with open(filename, 'w') as f:
                    f.write(f"Fetcher Report - {timeframe.name}\n")
                    f.write(f"Generated: {report_data['timestamp']}\n\n")
                    f.write("Statistics:\n")
                    for key, value in report_data['summary'].items():
                        f.write(f"{key}: {value}\n")
                return str(filename)

            elif format == ReportFormat.HTML:
                filename = filename.with_suffix('.html')
                with open(filename, 'w') as f:
                    html_content = f"""
                    <html>
                    <head><title>Fetcher Report - {timeframe.name}</title></head>
                    <body>
                        <h1>Fetcher Report - {timeframe.name}</h1>
                        <p>Generated: {report_data['timestamp']}</p>
                        <h2>Summary</h2>
                        <ul>
                        {"".join(f"<li>{k}: {v}</li>" for k, v in report_data['summary'].items())}
                        </ul>
                    </body>
                    </html>
                    """
                    f.write(html_content)
                return str(filename)

        except Exception as e:
            raise Exception(f"Failed to generate report: {str(e)}")

class TestFetcher(AbstractFetcher[dict]):
    """Test fetcher with controllable behavior for comprehensive testing."""

    def __init__(self, simulate_latency: bool = True):
        config = FetcherConfig(
            base_url="https://api.example.com",
            requests_per_second=10,
            max_retries=3,
            cache_ttl=60,
            enable_metrics=False,  # Disable metrics to avoid prometheus errors
            enable_telemetry=False  # Disable telemetry for testing
        )
        super().__init__(config, "test_fetcher")
        self.simulate_latency = simulate_latency
        self.failure_points: Dict[str, str] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._reset_stats()

    def _reset_stats(self):
        """Reset statistics for clean test runs."""
        self._stats = {
            'requests': 0,
            'successes': 0,
            'failures': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'rate_limits': 0,
            'circuit_breaks': 0,
            'total_latency': 0.0,
            'errors': []  # Track specific errors for better reporting
        }

    @property
    def default_headers(self) -> dict:
        return {"Accept": "application/json"}

    async def process_response(self, response: aiohttp.ClientResponse) -> dict:
        return await response.json()

    def set_endpoint_behavior(self, endpoint: str, behavior: str):
        """Configure endpoint to exhibit specific behavior."""
        self.failure_points[endpoint] = behavior

    async def simulate_request(
        self,
        endpoint: str,
        method: str = "GET",
        delay: Optional[float] = None
    ) -> FetchResult[dict]:
        """Simulate request with configurable behavior."""
        self._stats['requests'] += 1
        start_time = asyncio.get_event_loop().time()

        if self.simulate_latency and delay is None:
            delay = random.uniform(0.1, 0.5)

        if delay:
            await asyncio.sleep(delay)

        behavior = self.failure_points.get(endpoint, "success")
        result = None

        try:
            if behavior == "timeout":
                self._stats['failures'] += 1
                self._stats['errors'].append(f"Timeout on {endpoint}")
                raise asyncio.TimeoutError("Request timed out")

            elif behavior == "rate_limit":
                self._stats['rate_limits'] += 1
                self._stats['errors'].append(f"Rate limit on {endpoint}")
                raise RateLimitError(retry_after=0.5)

            elif behavior == "error_5xx":
                self._stats['failures'] += 1
                self._stats['errors'].append(f"5xx error on {endpoint}")
                raise aiohttp.ClientError("Internal Server Error")

            elif behavior == "error_4xx":
                self._stats['failures'] += 1
                self._stats['errors'].append(f"4xx error on {endpoint}")
                raise aiohttp.ClientResponseError(
                    request_info=None,
                    history=None,
                    status=404
                )

            self._stats['successes'] += 1
            result = FetchResult(
                status=FetchStatus.SUCCESS,
                data={"message": "Success", "endpoint": endpoint},
                metadata={"latency": delay if delay else 0}
            )

        finally:
            end_time = asyncio.get_event_loop().time()
            self._stats['total_latency'] += (end_time - start_time)

        return result

    async def bulk_fetch(self, requests: List[Dict], concurrency: int = 3) -> List[FetchResult]:
        """Perform bulk fetch operations with detailed results."""
        semaphore = asyncio.Semaphore(concurrency)
        results = []

        async def fetch_with_semaphore(request: Dict) -> FetchResult:
            async with semaphore:
                try:
                    return await self.simulate_request(**request)
                except Exception as e:
                    return FetchResult(
                        status=FetchStatus.FAILED,
                        error=str(e),
                        metadata={"request": request}
                    )

        tasks = [fetch_with_semaphore(request) for request in requests]
        results = await asyncio.gather(*tasks)
        return results

def print_scenario_results(scenario: str, stats: Dict[str, Any], additional_info: str = ""):
    """Print detailed results for a test scenario."""
    print(f"\n=== Results for {scenario} ===")
    if additional_info:
        print(f"Info: {additional_info}")
    print(f"Requests: {stats['requests']}")
    print(f"Successes: {stats['successes']}")
    print(f"Failures: {stats['failures']}")
    print(f"Rate Limits: {stats['rate_limits']}")
    if stats['errors']:
        print("Errors encountered:")
        for error in stats['errors'][-5:]:  # Show last 5 errors
            print(f"  - {error}")
    if stats['requests'] > 0:
        print(f"Average Latency: {stats['total_latency']/stats['requests']:.3f}s")
    print("=" * 50)

async def run_comprehensive_tests():
    """Run comprehensive test scenarios with detailed reporting."""
    print("\n=== Starting Comprehensive Fetcher Tests ===\n")

    reporter = TestReporter()
    async with TestFetcher() as fetcher:
        # Test Scenario 1: Normal Operation
        print("\n--- Scenario 1: Normal Operation ---")
        for i in range(10):
            await fetcher.simulate_request(f"/api/normal/{i}")
        print_scenario_results("Normal Operation", fetcher.stats)

        # Test Scenario 2: Caching Behavior
        print("\n--- Scenario 2: Caching Behavior ---")
        for _ in range(3):
            for i in range(3):
                await fetcher.simulate_request("/api/cached", delay=0.1)
        print_scenario_results("Caching Behavior", fetcher.stats)

        # Test Scenario 3: Rate Limiting
        print("\n--- Scenario 3: Rate Limiting ---")
        fetcher.set_endpoint_behavior("/api/heavy", "rate_limit")
        try:
            await asyncio.gather(*[
                fetcher.simulate_request("/api/heavy")
                for _ in range(5)
            ])
        except Exception as e:
            print(f"Rate limiting activated: {str(e)}")
        print_scenario_results("Rate Limiting", fetcher.stats)

        # Test Scenario 4: Error Handling
        print("\n--- Scenario 4: Error Handling ---")
        fetcher.set_endpoint_behavior("/api/unstable", "error_5xx")
        for i in range(3):
            try:
                await fetcher.simulate_request("/api/unstable")
            except Exception as e:
                print(f"Error {i+1}: {str(e)}")
        print_scenario_results("Error Handling", fetcher.stats)

        # Test Scenario 5: Mixed Traffic Pattern
        print("\n--- Scenario 5: Mixed Traffic Pattern ---")
        endpoints = [
            ("/api/fast", 0.1),
            ("/api/slow", 0.5),
            ("/api/normal", 0.2)
        ]
        for endpoint, delay in endpoints:
            await fetcher.simulate_request(endpoint, delay=delay)
        print_scenario_results("Mixed Traffic", fetcher.stats)

        # Test Scenario 6: Recovery Behavior
        print("\n--- Scenario 6: Recovery Behavior ---")
        fetcher.set_endpoint_behavior("/api/unstable", "success")  # Reset behavior
        for i in range(5):
            await fetcher.simulate_request("/api/unstable")
        print_scenario_results("Recovery Behavior", fetcher.stats)

        # Test Scenario 7: Bulk Operations
        print("\n--- Scenario 7: Bulk Operations ---")
        requests = [
            {"endpoint": f"/api/bulk/{i}", "method": "GET"}
            for i in range(5)
        ]
        results = await fetcher.bulk_fetch(requests, concurrency=2)
        successful = sum(1 for r in results if r.status == FetchStatus.SUCCESS)
        print_scenario_results("Bulk Operations", fetcher.stats,
                             f"Successful bulk requests: {successful}/{len(requests)}")

        # Generate Reports
        print("\n--- Generating Reports ---")
        for timeframe in [ReportTimeframe.HOUR, ReportTimeframe.DAY]:
            for format in [ReportFormat.JSON, ReportFormat.TEXT, ReportFormat.HTML]:
                try:
                    report_path = await reporter.generate_summary_report(
                        fetcher.stats,
                        timeframe,
                        format
                    )
                    print(f"Generated {format.name} report for {timeframe.name}: {report_path}")
                except Exception as e:
                    print(f"Failed to generate report ({timeframe.name}, {format.name}): {str(e)}")

        # Print Final Summary
        print("\n=== Final Test Summary ===")
        print(f"Total Requests: {fetcher.stats['requests']}")
        print(f"Successful Requests: {fetcher.stats['successes']}")
        print(f"Failed Requests: {fetcher.stats['failures']}")
        print(f"Cache Hits: {fetcher.stats['cache_hits']}")
        print(f"Cache Misses: {fetcher.stats['cache_misses']}")
        print(f"Rate Limits Hit: {fetcher.stats['rate_limits']}")
        print(f"Circuit Breaker Trips: {fetcher.stats['circuit_breaks']}")
        if fetcher.stats['requests'] > 0:
            print(f"Average Latency: {fetcher.stats['total_latency']/fetcher.stats['requests']:.3f}s")

if __name__ == "__main__":
    asyncio.run(run_comprehensive_tests())