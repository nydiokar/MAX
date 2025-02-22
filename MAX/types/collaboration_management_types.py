from enum import Enum
from typing import Dict, List, Optional, Set, Union, Any
from dataclasses import dataclass
from datetime import datetime
from .base_types import AgentResponse
from .collaboration_types import SubTask, CollaborationStatus

@dataclass
class TaskDivisionPlan:
    """Plan for dividing a task among team members."""
    parent_task_id: str
    subtasks: List[SubTask]
    dependencies: Dict[str, Set[str]]  # subtask_id -> set of dependent subtask_ids
    estimated_duration: Dict[str, float]  # subtask_id -> estimated hours
    assignment_map: Dict[str, str]  # subtask_id -> agent_id

class AggregationStrategy(Enum):
    SEQUENTIAL = "sequential"  # Combine responses in sequence
    PARALLEL = "parallel"     # Merge parallel responses
    WEIGHTED = "weighted"     # Use weighted scoring
    VOTING = "voting"        # Use consensus/voting
    HYBRID = "hybrid"        # Combine multiple strategies

class ResponseType(Enum):
    TEXT = "text"
    STRUCTURED = "structured"
    CODE = "code"
    DATA = "data"
    ERROR = "error"
