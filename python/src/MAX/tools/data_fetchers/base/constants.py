from enum import Enum, auto
from typing import Final


class FetchStatus(Enum):
    SUCCESS = auto()
    FAILED = auto()
    RATE_LIMITED = auto()
    CACHED = auto()
    CIRCUIT_BROKEN = auto()


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# Default configuration values
DEFAULT_TIMEOUT: Final = 30.0
DEFAULT_RETRIES: Final = 3
DEFAULT_CACHE_TTL: Final = 300
DEFAULT_REPORT_INTERVAL: Final = 3600
