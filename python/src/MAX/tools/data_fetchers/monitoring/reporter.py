from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from enum import Enum
import json
from pathlib import Path

from ..base.types import HealthStatus, HealthData, Monitorable, FetchStatus
from .health import HealthMonitor
from .metrics import FetcherMetrics


class ReportFormat(Enum):
    JSON = "json"
    TEXT = "text"
    HTML = "html"


class ReportTimeframe(Enum):
    HOUR = timedelta(hours=1)
    DAY = timedelta(days=1)
    WEEK = timedelta(days=7)
    MONTH = timedelta(days=30)


class FetcherReporter:
    def __init__(
        self, health_monitor: HealthMonitor, report_dir: Optional[str] = None
    ):
        self.health_monitor = health_monitor
        self.report_dir = Path(report_dir) if report_dir else Path("reports")
        self.report_dir.mkdir(parents=True, exist_ok=True)

    async def generate_summary_report(
        self,
        component: Monitorable,
        timeframe: ReportTimeframe,
        format: ReportFormat = ReportFormat.JSON,
    ) -> Dict[str, Any]:
        """Generate a comprehensive summary report for a monitored component."""
        current_time = datetime.now()
        start_time = current_time - timeframe.value

        # Get health history for component
        health_history = self.health_monitor.health_history.get(
            component.name, []
        )
        relevant_history = [
            h for h in health_history if h.last_check >= start_time
        ]

        # Get current health status
        current_health = await self.health_monitor.check_health()
        component_health = current_health.get(component.name)

        if not component_health:
            raise ValueError(f"No health data available for {component.name}")

        # Calculate statistics
        stats = self._calculate_stats(relevant_history)

        report = {
            "timestamp": current_time.isoformat(),
            "component_name": component.name,
            "timeframe": timeframe.name,
            "current_status": component_health.status.value,
            "statistics": stats,
            "metrics": self._get_metrics_summary(component.metrics),
            "performance": {
                "average_latency": stats["avg_latency"],
                "p95_latency": stats["p95_latency"],
                "error_rate": stats["error_rate"],
                "success_rate": stats["success_rate"],
            },
            "issues": self._identify_issues(stats, component_health),
            "recommendations": self._generate_recommendations(
                stats, component_health
            ),
        }

        self._save_report(report, component.name, format)
        return report

    def _calculate_stats(self, history: List[HealthData]) -> Dict[str, float]:
        """Calculate statistics from health history."""
        if not history:
            return {
                "error_rate": 0.0,
                "success_rate": 100.0,
                "avg_latency": 0.0,
                "p95_latency": 0.0,
                "cache_hit_rate": 0.0,
                "rate_limit_percentage": 0.0,
                "circuit_breaker_percentage": 0.0,
            }

        total_records = len(history)
        return {
            "error_rate": sum(h.error_rate for h in history) / total_records,
            "success_rate": 100
            - (sum(h.error_rate for h in history) / total_records * 100),
            "avg_latency": sum(h.latency_p95 for h in history) / total_records,
            "p95_latency": sorted([h.latency_p95 for h in history])[
                int(total_records * 0.95)
            ],
            "cache_hit_rate": sum(h.cache_hit_rate for h in history)
            / total_records,
            "rate_limit_percentage": sum(
                1 for h in history if h.rate_limit_hits > 0
            )
            / total_records
            * 100,
            "circuit_breaker_percentage": sum(
                1 for h in history if h.circuit_breaker_trips > 0
            )
            / total_records
            * 100,
        }

    def _get_metrics_summary(self, metrics: FetcherMetrics) -> Dict[str, Any]:
        """Generate metrics summary."""
        return {
            "total_requests": sum(
                v
                for k, v in metrics.requests._value.items()
                if k[1] == FetchStatus.SUCCESS.name
            ),
            "total_errors": sum(
                v
                for k, v in metrics.requests._value.items()
                if k[1] == FetchStatus.FAILED.name
            ),
            "latency_stats": {
                "avg": metrics.latency._sum.get()
                / max(metrics.latency._count.get(), 1),
                "count": metrics.latency._count.get(),
            },
            "cache_operations": {
                "hits": sum(
                    v
                    for k, v in metrics.cache_stats._value.items()
                    if k[1] == "hit"
                ),
                "misses": sum(
                    v
                    for k, v in metrics.cache_stats._value.items()
                    if k[1] == "miss"
                ),
            },
        }

    def _identify_issues(
        self, stats: Dict[str, float], health_data: HealthData
    ) -> List[str]:
        """Identify potential issues based on statistics and current health."""
        issues = []

        if health_data.status == HealthStatus.UNHEALTHY:
            issues.append("Component is currently UNHEALTHY")

        if health_data.status == HealthStatus.DEGRADED:
            issues.append("Component is currently DEGRADED")

        if stats["error_rate"] > 0.1:
            issues.append(f"High error rate: {stats['error_rate']:.2f}%")

        if stats["rate_limit_percentage"] > 10:
            issues.append(
                "Frequent rate limiting - consider adjusting request rate"
            )

        if stats["circuit_breaker_percentage"] > 5:
            issues.append(
                "Circuit breaker tripping frequently - check endpoint stability"
            )

        if stats["cache_hit_rate"] < 0.3:
            issues.append(
                "Low cache hit rate - consider adjusting caching strategy"
            )

        if stats["p95_latency"] > 1.0:  # 1 second
            issues.append(f"High P95 latency: {stats['p95_latency']:.2f}s")

        return issues

    def _generate_recommendations(
        self, stats: Dict[str, float], health_data: HealthData
    ) -> List[str]:
        """Generate recommendations based on statistics and health data."""
        recommendations = []

        if health_data.status != HealthStatus.HEALTHY:
            recommendations.append(
                f"Address issues causing {health_data.status.value} status"
            )

        if stats["rate_limit_percentage"] > 10:
            recommendations.append(
                "Consider implementing request rate throttling or increasing rate limits"
            )

        if stats["cache_hit_rate"] < 0.3:
            recommendations.append(
                "Consider increasing cache TTL or reviewing cache key strategy"
            )

        if stats["error_rate"] > 0.1:
            recommendations.append(
                "Implement more aggressive retry strategies or review error handling"
            )

        if stats["circuit_breaker_percentage"] > 5:
            recommendations.append(
                "Review circuit breaker thresholds and implement fallback mechanisms"
            )

        return recommendations

    def _save_report(
        self, report: Dict[str, Any], component_name: str, format: ReportFormat
    ) -> None:
        """Save report in specified format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{component_name}_report_{timestamp}"

        if format == ReportFormat.JSON:
            with open(self.report_dir / f"{filename}.json", "w") as f:
                json.dump(report, f, indent=2)

        elif format == ReportFormat.TEXT:
            with open(self.report_dir / f"{filename}.txt", "w") as f:
                self._write_text_report(report, f)

        elif format == ReportFormat.HTML:
            with open(self.report_dir / f"{filename}.html", "w") as f:
                self._write_html_report(report, f)

    def _write_text_report(self, report: Dict[str, Any], file) -> None:
        """Write report in text format."""
        file.write(f"Fetcher Report - {report['timestamp']}\n")
        file.write("=" * 50 + "\n\n")

        file.write(f"Component: {report['component_name']}\n")
        file.write(f"Status: {report['current_status']}\n")
        file.write(f"Timeframe: {report['timeframe']}\n\n")

        file.write("Performance Metrics:\n")
        file.write("-" * 20 + "\n")
        for key, value in report["performance"].items():
            file.write(f"{key}: {value:.2f}\n")

        if report["issues"]:
            file.write("\nIdentified Issues:\n")
            file.write("-" * 20 + "\n")
            for issue in report["issues"]:
                file.write(f"- {issue}\n")

        if report["recommendations"]:
            file.write("\nRecommendations:\n")
            file.write("-" * 20 + "\n")
            for rec in report["recommendations"]:
                file.write(f"- {rec}\n")

    def _write_html_report(self, report: Dict[str, Any], file) -> None:
        """Write report in HTML format."""
        status_class = report["current_status"].lower()

        file.write(
            f"""
        <html>
            <head>
                <title>Fetcher Report - {report['component_name']}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    .section {{ margin: 20px 0; }}
                    .status {{ padding: 5px; border-radius: 3px; }}
                    .healthy {{ background-color: #90EE90; }}
                    .degraded {{ background-color: #FFD700; }}
                    .unhealthy {{ background-color: #FFB6C1; }}
                    .metric {{ margin: 10px 0; }}
                    .issues {{ color: #D8000C; }}
                    .recommendations {{ color: #4F8A10; }}
                </style>
            </head>
            <body>
                <h1>Fetcher Report</h1>
                <div class="section">
                    <h2>Overview</h2>
                    <p><strong>Component:</strong> {report['component_name']}</p>
                    <p><strong>Timestamp:</strong> {report['timestamp']}</p>
                    <p><strong>Status:</strong> 
                        <span class="status {status_class}">
                            {report['current_status']}
                        </span>
                    </p>
                </div>
        """
        )

        # Performance section
        file.write(
            """
                <div class="section">
                    <h2>Performance</h2>
        """
        )
        for key, value in report["performance"].items():
            file.write(
                f"""
                    <div class="metric">
                        <strong>{key}:</strong> {value:.2f}
                    </div>
            """
            )
        file.write("</div>")

        # Issues section
        if report["issues"]:
            file.write(
                """
                <div class="section">
                    <h2>Issues</h2>
                    <ul class="issues">
            """
            )
            for issue in report["issues"]:
                file.write(f"<li>{issue}</li>")
            file.write("</ul></div>")

        # Recommendations section
        if report["recommendations"]:
            file.write(
                """
                <div class="section">
                    <h2>Recommendations</h2>
                    <ul class="recommendations">
            """
            )
            for rec in report["recommendations"]:
                file.write(f"<li>{rec}</li>")
            file.write("</ul></div>")

        file.write(
            """
            </body>
        </html>
        """
        )
