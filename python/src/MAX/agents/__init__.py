"""
Code for Agents.
"""
from .options import (
    BaseAgentOptions,
    AgentOptions,
    AnthropicAgentOptions,
    TaskExpertOptions
)
from .agent import Agent, AgentCallbacks, AgentProcessingResult, AgentResponse
from .anthropic_agent import AnthropicAgent
from .task_expert import TaskExpertAgent
from .ollama_agent import OllamaAgent, OllamaAgentOptions
__all__ = [
    'BaseAgentOptions',
    'AgentOptions',
    'AnthropicAgentOptions',
    'TaskExpertOptions',
    'Agent',
    'AnthropicAgent',
    'TaskExpertAgent',
    'ChainAgent',
    'ChainAgentOptions',
    'AgentCallbacks',
    'AgentProcessingResult',
    'AgentResponse',
    'OllamaAgent',
    'OllamaAgentOptions',
]