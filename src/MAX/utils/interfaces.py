from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, field_validator

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    OVERDUE = "overdue"
    BLOCKED = "blocked"

class TaskPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class TaskModel(BaseModel):
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

    @field_validator('progress')
    @classmethod
    def validate_progress(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Progress must be between 0 and 100')
        return v

class TaskStorage:
    """Abstract base class for task storage"""
    async def save_task(self, task: TaskModel) -> None:
        raise NotImplementedError
        
    async def get_task(self, task_id: str) -> Optional[TaskModel]:
        raise NotImplementedError
        
    async def get_tasks(self, filters: Dict[str, Any]) -> List[TaskModel]:
        raise NotImplementedError
        
    async def delete_task(self, task_id: str) -> None:
        raise NotImplementedError
        
    async def save_interaction(self, interaction: Dict[str, Any]) -> None:
        raise NotImplementedError

class NotificationService:
    """Abstract base class for notification service"""
    async def send(self, agent_id: str, notification: Dict[str, Any]) -> None:
        raise NotImplementedError
