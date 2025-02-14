from prometheus_client import Counter, Histogram, Gauge
from typing import Dict


class FetcherMetrics:
    def __init__(self, fetcher_name: str):
        # Core metrics
        self.requests = Counter(
            f"{fetcher_name}_requests_total",
            "Total number of requests",
            ["method", "status", "endpoint"],
        )
        self.latency = Histogram(
            f"{fetcher_name}_request_duration_seconds",
            "Request duration in seconds",
            ["method", "endpoint"],
        )
        self.errors = Counter(
            f"{fetcher_name}_errors_total",
            "Total number of errors",
            ["type", "endpoint"],
        )
        self.cache_stats = Counter(
            f"{fetcher_name}_cache_operations_total",
            "Cache operations",
            ["operation", "result"],
        )

    def record_request(self, method: str, status: str, endpoint: str) -> None:
        self.requests.labels(
            method=method, status=status, endpoint=endpoint
        ).inc()

    def record_latency(
        self, method: str, endpoint: str, duration: float
    ) -> None:
        self.latency.labels(method=method, endpoint=endpoint).observe(duration)

    def record_error(self, error_type: str, endpoint: str) -> None:
        self.errors.labels(type=error_type, endpoint=endpoint).inc()
