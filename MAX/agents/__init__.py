"""
Code for Agents.
"""

from typing import TYPE_CHECKING, Optional, Type, Any

# Base classes and types
from MAX.agents.agent import (
    Agent,
    AgentCallbacks,
    AgentProcessingResult,
    AgentResponse,
    AgentOptions,
)
from MAX.agents.options import (
    BaseAgentOptions,
    RecursiveThinkerOptions
)

# Import the RecursiveThinkerAgent
from MAX.agents.recursive_thinker import RecursiveThinkerAgent

# Options - import these directly from their modules
from MAX.agents.task_expert.options import TaskExpertOptions

# Interfaces
from MAX.storage.utils.protocols import TaskStorage, NotificationService

# Model providers - import base class
from MAX.llms.base import AsyncLLMBase as LLMProvider

# Conditionally import providers to avoid dependency issues
try:
    from MAX.llms.ollama import OllamaLLM as OllamaProvider
except ImportError:
    OllamaProvider = None

try:
    from MAX.llms.anthropic import AnthropicLLM as AnthropicProvider
except ImportError:
    AnthropicProvider = None

try:
    from MAX.llms.openai import OpenAILLM as OpenAIProvider
except ImportError:
    OpenAIProvider = None

if TYPE_CHECKING:
    from MAX.agents.task_expert.task_expert import TaskExpertAgent


# Expert agents - lazy import to avoid circular dependencies
def get_task_expert_agent() -> Optional[Type[Any]]:
    try:
        from MAX.agents.task_expert.task_expert import TaskExpertAgent

        return TaskExpertAgent
    except ImportError as e:
        print(f"Failed to import TaskExpertAgent: {e}")
        return None


__all__ = [
    # Base classes and types
    "Agent",
    "AgentCallbacks",
    "AgentProcessingResult",
    "AgentResponse",
    # Options
    "BaseAgentOptions",
    "AgentOptions",
    "RecursiveThinkerOptions",
    "TaskExpertOptions",
    # Agents
    "RecursiveThinkerAgent",
    # Interfaces
    "TaskStorage",
    "NotificationService",
    # Model providers
    "LLMProvider",
    "OllamaProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    # Expert agents
    "get_task_expert_agent",
]
