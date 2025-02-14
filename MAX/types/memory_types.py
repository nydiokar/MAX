from enum import Enum
from datetime import datetime
from typing import Dict, Any
from pydantic import BaseModel, Field


class DataCategory(str, Enum):
    """Extensible data categories"""

    USER = "user"
    SYSTEM = "system"
    AGENT = "agent"
    FETCHER = "fetcher"
    ANALYSIS = "analysis"


class DataPriority(str, Enum):
    """Priority levels for processing"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MemoryData(BaseModel):
    """Base memory data structure"""

    id: str
    timestamp: datetime
    category: DataCategory
    priority: DataPriority
    content: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
