"""
Code for Agents.
"""
# Base classes and types
from .agent import Agent, AgentCallbacks, AgentProcessingResult, AgentResponse
# Options
from ..utils.options import (
    BaseAgentOptions,
    AgentOptions,
    AnthropicAgentOptions,
    TaskExpertOptions
)
# Interfaces
from ..utils.interfaces import TaskStorage, NotificationService
# Model providers   
from ..adapters.llm import (
    LLMProvider,
    OllamaProvider,
    AnthropicProvider
)
# Expert agents
from .anthropic_agent import AnthropicAgent
from .task_expert import TaskExpertAgent

__all__ = [
    # Base classes and types
    'Agent',
    'AgentCallbacks',
    'AgentProcessingResult',
    'AgentResponse',
    
    # Options
    'BaseAgentOptions',
    'AgentOptions',
    'AnthropicAgentOptions',
    'TaskExpertOptions',
    
    # Interfaces
    'TaskStorage',
    'NotificationService',
    
    # Model providers
    'LLMProvider',
    'OllamaProvider', 
    'AnthropicProvider',
    
    # Expert agents
    'AnthropicAgent',
    'TaskExpertAgent'
]