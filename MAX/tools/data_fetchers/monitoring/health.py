from typing import Dict, List, Any
from datetime import datetime
from ..base.types import HealthStatus, HealthData, Monitorable, FetchStatus


class HealthMonitor:
    """Independent health monitoring for components."""

    def __init__(self):
        self.components: Dict[str, Monitorable] = {}
        self.health_history: Dict[str, List[HealthData]] = {}

    def register_component(self, component: Monitorable) -> None:
        """Register a component for health monitoring."""
        self.components[component.name] = component
        self.health_history[component.name] = []

    async def check_health(self) -> Dict[str, HealthData]:
        """Generate health report for all registered components."""
        health_status = {}
        for name, component in self.components.items():
            status = await self._evaluate_health(component)
            self.health_history[name].append(status)
            # Keep last 100 status checks
            self.health_history[name] = self.health_history[name][-100:]
            health_status[name] = status
        return health_status

    async def _evaluate_health(self, component: Monitorable) -> HealthData:
        """Evaluate health based on metrics and stats."""
        stats = component.stats
        metrics = component.metrics

        error_rate = self._calculate_error_rate(metrics)
        latency = self._calculate_p95_latency(metrics)

        status = HealthStatus.HEALTHY
        if error_rate > 0.25 or latency > 5.0:
            status = HealthStatus.UNHEALTHY
        elif error_rate > 0.1 or latency > 2.0:
            status = HealthStatus.DEGRADED

        return HealthData(
            status=status,
            last_check=datetime.now(),
            error_rate=error_rate,
            latency_p95=latency,
            circuit_breaker_trips=metrics.circuit_breaker_trips._value.sum(),
            rate_limit_hits=metrics.rate_limit_hits._value.sum(),
            cache_hit_rate=self._calculate_cache_hit_rate(stats),
            details=stats,
        )

    def _calculate_error_rate(self, metrics) -> float:
        total_requests = sum(
            metrics.requests.labels(status=FetchStatus.SUCCESS.name).values()
        )
        failed_requests = sum(
            metrics.requests.labels(status=FetchStatus.FAILED.name).values()
        )
        return failed_requests / max(total_requests, 1)

    def _calculate_p95_latency(self, metrics) -> float:
        # Implement p95 latency calculation from metrics
        return (
            metrics.latency.observe()
        )  # This needs to be implemented based on your metrics structure

    def _calculate_cache_hit_rate(self, stats: Dict[str, Any]) -> float:
        hits = stats.get("cache_hits", 0)
        total = hits + stats.get("cache_misses", 0)
        return hits / max(total, 1)
