from typing import Dict, Union, List

# Add this type definition
TemplateVariables = Dict[str, Union[str, List[str]]]

"""
Types module for MAX system.
Import all types from their respective modules and expose them at package level.
"""

# Base types
from .base_types import (
    MessageType,
    ParticipantRole,
    BaseMessage,
    ConversationMessage,
    AgentMessage,
    TimestampedMessage,
    MessageContent,
    AgentProviderType,
    AgentTypes,
    AgentMetadata
)

# Memory types
from .memory_types import (
    DataCategory,
    DataPriority,
    MemoryEntry,
    ConversationMemory,
    MemoryData
)

# Collaboration types
from .collaboration_types import (
    SubTask,
    CollaborationRole,
    CollaborationStatus,
    CollaborationContext,
    CollaborationMessage
)

# Collaboration management types
from .collaboration_management_types import (
    TaskDivisionPlan,
    AggregationStrategy,
    ResponseType
)
from MAX.config.base_llm import ResourceConfig

__all__ = [
    # Add TemplateVariables to __all__
    "TemplateVariables",
    
    # Base types
    "MessageType",
    "ParticipantRole",
    "BaseMessage",
    "ConversationMessage",
    "AgentMessage",
    "TimestampedMessage",
    "MessageContent",
    
    # Memory types
    "DataCategory",
    "DataPriority",
    "MemoryEntry",
    "ConversationMemory",
    "MemoryData",
    
    # Collaboration types
    "SubTask",
    "CollaborationRole",
    "CollaborationStatus",
    "CollaborationContext",
    "CollaborationMessage",
    
    # Collaboration management types
    "TaskDivisionPlan",
    "AggregationStrategy",
    "ResponseType",
    
    # Agent types
    "AgentProviderType",
    "AgentTypes",
    "AgentMetadata",
    "ResourceConfig"
]
