from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum
from pydantic import field_validator


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


class Task(BaseModel):
    """
    Represents a task in the system.
    """

    task_id: str
    title: str
    description: str
    assigned_agent: str
    status: TaskStatus
    priority: TaskPriority
    created_at: datetime
    due_date: Optional[datetime]
    dependencies: List[str] = Field(default_factory=list)
    progress: float = 0
    last_updated: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_by: str
    tags: List[str] = Field(default_factory=list)
    estimated_hours: Optional[float] = None
    priority_score: Optional[float] = None

    @field_validator("progress")
    @classmethod
    def validate_progress(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("Progress must be between 0 and 100")
        return v


class ExecutionHistoryEntry(BaseModel):
    """
    Represents an entry in a task's execution history.
    """

    entry_id: str
    task_id: str
    timestamp: datetime
    status: str
    changed_by: str  # e.g., user ID or system ID
    comment: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

