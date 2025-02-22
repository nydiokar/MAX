from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

class WorkflowStage(str, Enum):
    """Stages in a workflow"""
    INIT = "init"
    PLANNING = "planning"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"

class WorkflowStatus(str, Enum):
    """Status of a workflow"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

class IntrospectionScope(str, Enum):
    """Scope for workflow introspection"""
    LOCAL = "local"
    GLOBAL = "global"
    TEAM = "team"

@dataclass
class WorkflowState:
    """Current state of a workflow"""
    workflow_id: str
    stage: WorkflowStage
    status: WorkflowStatus
    metadata: Dict[str, Any]
    created_at: datetime = datetime.now()
    updated_at: Optional[datetime] = None

@dataclass
class WorkflowTransition:
    """Transition between workflow states"""
    from_state: WorkflowState
    to_state: WorkflowState
    trigger: str
    timestamp: datetime = datetime.now()
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class WorkflowContext:
    """Context for workflow execution"""
    workflow_id: str
    current_state: WorkflowState
    transitions: List[WorkflowTransition]
    scope: IntrospectionScope = IntrospectionScope.LOCAL
    metadata: Optional[Dict[str, Any]] = None
