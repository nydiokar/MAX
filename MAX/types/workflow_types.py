from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

class WorkflowStage(Enum):
    MEMORY = "memory"
    REASONING = "reasoning"
    EXECUTION = "execution"
    INTROSPECTION = "introspection"

class IntrospectionScope(Enum):
    AGENT_PERFORMANCE = "agent_performance"
    WORKFLOW_EFFECTIVENESS = "workflow_effectiveness"
    DECISION_QUALITY = "decision_quality"
    ERROR_ANALYSIS = "error_analysis"
    INTROSPECTION = "introspection"

class WorkflowStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"

@dataclass
class WorkflowState:
    stage: WorkflowStage
    status: WorkflowStatus
    current_agent: Optional[str] = None
    memory_context: Dict[str, Any] = None
    reasoning_result: Dict[str, Any] = None
    execution_result: Dict[str, Any] = None
    introspection_data: Dict[str, Any] = None
    error_details: Optional[str] = None
    performance_metrics: Dict[str, float] = None
    improvement_suggestions: List[str] = None
    
@dataclass
class WorkflowTransition:
    from_stage: WorkflowStage
    to_stage: WorkflowStage
    validation_result: bool
    validation_message: Optional[str] = None

@dataclass
class WorkflowContext:
    session_id: str
    user_id: str
    initial_input: str
    current_state: WorkflowState
    history: List[WorkflowTransition] = None
    metadata: Dict[str, Any] = None
