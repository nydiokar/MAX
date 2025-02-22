from enum import Enum
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from datetime import datetime
from MAX.types.workflow_types import WorkflowState
from .base_types import AgentResponse, MessageType

class CollaborationRole(str, Enum):
    INITIATOR = "initiator"
    PARTICIPANT = "participant"
    OBSERVER = "observer"

class CollaborationStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class SubTask:
    """Represents a subtask in a collaboration"""
    task_id: str
    parent_task_id: str
    description: str
    assigned_agent: str
    status: str = "pending"
    priority: int = 1
    dependencies: Set[str] = field(default_factory=set)
    completion_criteria: Optional[Dict[str, Any]] = None
    result: Optional[AgentResponse] = None
    created_at: datetime = datetime.now()
    completed_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class CollaborationContext:
    """Context for collaboration between agents"""
    context_id: str
    initiator: str
    participants: List[str]
    subtasks: List[SubTask]
    status: CollaborationStatus = CollaborationStatus.PENDING
    created_at: datetime = datetime.now()
    updated_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class AgentCollaborationConfig:
    supported_roles: List[CollaborationRole]
    max_concurrent_tasks: int
    specialties: List[str]
    collaboration_preferences: Dict[str, Any]  # e.g., preferred agents to work with
    performance_metrics: Dict[str, float]  # Historical performance in different roles

@dataclass
class CollaborationMessage:
    """Message exchanged during collaboration"""
    message_id: str
    sender: str
    content: str
    context_id: str
    recipient: Optional[str] = None
    message_type: MessageType = MessageType.TEXT
    timestamp: datetime = datetime.now()
    metadata: Optional[Dict[str, Any]] = None
