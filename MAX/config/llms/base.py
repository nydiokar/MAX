# base_llm.py
from dataclasses import dataclass, field
from typing import Optional


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
    """
    Base configuration for all LLM providers. Intended to be extended by provider-specific configs.

    Attributes:
        model (Optional[str]): The model name or ID to use.
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum tokens to generate in one call.
        top_p (float): Top-p (nucleus) sampling.
        resources (ResourceConfig): Resource management details.
        api_base_url (Optional[str]): API base URL if the LLM provider requires it.
        auto_pull_models (bool): Whether to automatically pull models if not found locally.
        fallback_model (Optional[str]): Name of a fallback model if primary fails.
    """

    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    resources: ResourceConfig = field(default_factory=create_resource_config)

    # Additional optional fields often needed by remote or local providers
    api_base_url: Optional[str] = None
    auto_pull_models: bool = False
    fallback_model: Optional[str] = None
