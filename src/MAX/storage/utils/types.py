from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
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

    @field_validator('progress')
    @classmethod
    def validate_progress(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Progress must be between 0 and 100')
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

class RequestMetadata(BaseModel):
    """
    Metadata associated with a user request.
    """
    user_input: str
    agent_id: str
    agent_name: str
    user_id: str
    session_id: str
    additional_params: Optional[Dict[str, str]] = None
    error_type: Optional[str] = None


class ParticipantRole(Enum):
    ASSISTANT = "assistant"
    USER = "user"
    SYSTEM = "system"
    STATE = "state"


@dataclass
class AgentResponse:
    """
    Represents a response from an agent.
    """
    metadata: Dict[str, Any]
    output: Any
    streaming: bool = False


class ConversationMessage(BaseModel):
    """
    Represents a single message in a conversation.
    """
    role: ParticipantRole
    content: Union[str, List[Dict[str, str]]]
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = {}


class TimestampedMessage(ConversationMessage):
    timestamp: datetime

    class Config:
        arbitrary_types_allowed = True


TemplateVariables = Dict[str, Union[str, List[str]]]
