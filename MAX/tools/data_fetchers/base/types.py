from typing import Literal, TypeVar, Dict, Optional, Protocol, Any, Generic
from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime

# Type definitions
T = TypeVar("T", covariant=True)
HTTPMethod = Literal["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"]


class FetchStatus(Enum):
    """Status of a fetch operation."""

    SUCCESS = auto()
    FAILED = auto()
    RATE_LIMITED = auto()
    CACHED = auto()
    CIRCUIT_BROKEN = auto()


class HealthStatus(Enum):
    """Health status of a monitorable component."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True)
class FetchResult(Generic[T]):
    """Immutable result of a fetch operation."""

    status: FetchStatus
    data: Optional[T] = None
    error: Optional[Exception] = None
    retry_after: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthData:
    """Health information for a monitored component."""

    status: HealthStatus
    last_check: datetime
    error_rate: float
    latency_p95: float
    circuit_breaker_trips: int
    rate_limit_hits: int
    cache_hit_rate: float
    details: Dict[str, Any]


class Monitorable(Protocol):
    """Protocol for objects that can be monitored."""

    @property
    def name(self) -> str:
        """Name of the monitorable component."""
        pass

    @property
    def stats(self) -> Dict[str, Any]:
        """Current statistics of the component."""
        pass

    @property
    def metrics(self) -> Any:
        """Metrics collector for the component."""
        pass
