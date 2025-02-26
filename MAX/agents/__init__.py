"""Agents module."""

# Base classes and types
from MAX.agents.agent import (
    Agent,
    AgentCallbacks,
    AgentOptions,
    AgentProcessingResult,
    AgentResponse,
)

# Import recursive thinking agent
from MAX.agents.recursive_thinker import (
    RecursiveThinkerAgent,
    RecursiveThinkerOptions
)

__all__ = [
    # Base classes
    "Agent",
    "AgentCallbacks",
    "AgentOptions",
    "AgentProcessingResult",
    "AgentResponse",
    # Recursive thinking
    "RecursiveThinkerAgent",
    "RecursiveThinkerOptions",
]
