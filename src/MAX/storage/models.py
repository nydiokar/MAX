from typing import Dict, Any, Optional, List, Protocol
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, field_validator

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    OVERDUE = "overdue"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TaskModel(BaseModel):
    """
    Represents a task in the system.
    """
    id: str
    title: str
    description: str
    assigned_agent: str
    status: TaskStatus
    priority: TaskPriority
    created_at: datetime
    due_date: Optional[datetime]
    dependencies: List[str]
    progress: float
    last_updated: datetime
    metadata: Dict[str, Any]
    created_by: str
    tags: List[str] = []
    estimated_hours: Optional[float] = None

    @field_validator('progress')
    @classmethod
    def validate_progress(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Progress must be between 0 and 100')
        return v