# base_llm.py
from dataclasses import dataclass, field
from typing import Optional
from MAX.config.models import AnthropicModels


@dataclass
class ResourceConfig:
    """
    Configuration for LLM resource management and execution scheduling.

    Attributes:
        max_parallel_calls (int): Maximum number of parallel inference calls.
        cost_per_token (float): Cost per token for accounting or billing.
        priority (int): Priority level for scheduling requests.
        local_only (bool): Indicates whether the model should only run locally.
    """

    max_parallel_calls: int = 5
    cost_per_token: float = 0.0
    priority: int = 1
    local_only: bool = False


def create_resource_config() -> ResourceConfig:
    try:
        return ResourceConfig()
    except Exception as e:
        raise ValueError(f"Failed to create ResourceConfig: {str(e)}") from e


@dataclass
class BaseLlmConfig:
    """Base configuration for all LLM providers"""
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 1024
    streaming: bool = False
    resources: Optional[ResourceConfig] = None
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None

    def __str__(self) -> str:
        """Safe string representation that hides API key"""
        return f"LLMConfig(model={self.model}, api_key=[REDACTED])"

    def __repr__(self) -> str:
        """Safe repr that hides API key"""
        return self.__str__()
