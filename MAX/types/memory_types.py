from enum import Enum
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from pydantic import BaseModel, Field
from dataclasses import dataclass
from .base_types import ParticipantRole, MessageType


class DataCategory(str, Enum):
    """Memory data categories"""

    USER = "user"
    SYSTEM = "system"
    AGENT = "agent"
    TASK = "task"


class DataPriority(str, Enum):
    """Priority levels for memory data"""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class MemoryEntry:
    """Individual memory entry"""

    content: str
    role: 'ParticipantRole'  # Forward reference
    timestamp: datetime
    category: DataCategory
    priority: DataPriority = DataPriority.MEDIUM
    relevance_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ConversationMemory:
    """Container for conversation memory"""

    conversation_id: str
    messages: List[MemoryEntry]
    metadata: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)


class MemoryData(BaseModel):
    """Memory data for storage"""

    id: str
    timestamp: datetime
    category: DataCategory
    priority: DataPriority
    content: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
