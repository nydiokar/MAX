from enum import Enum
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass
from datetime import datetime
from MAX.types.workflow_types import WorkflowState

class CollaborationRole(Enum):
    COORDINATOR = "coordinator"  # Manages collaboration and integrates results
    CONTRIBUTOR = "contributor"  # Works on assigned subtasks
    VALIDATOR = "validator"      # Validates intermediate results
    OBSERVER = "observer"       # Monitors progress without direct contribution

class CollaborationStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    AWAITING_INPUTS = "awaiting_inputs"
    INTEGRATING = "integrating"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class SubTask:
    id: str
    parent_task_id: str
    assigned_agent: str
    description: str
    dependencies: Set[str]  # IDs of subtasks this depends on
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@dataclass
class CollaborationContext:
    id: str
    workflow_state: WorkflowState
    coordinator_agent: str
    contributors: Dict[str, List[str]]  # Agent ID -> List of assigned subtask IDs
    validators: List[str]  # List of validator agent IDs
    status: CollaborationStatus
    subtasks: Dict[str, SubTask]
    shared_context: Dict[str, Any]  # Shared data between agents
    progress: float  # Overall progress 0-1
    created_at: datetime
    updated_at: datetime
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
    from_agent: str
    to_agent: str
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime
    subtask_id: Optional[str] = None
    requires_response: bool = False
    response_deadline: Optional[datetime] = None
