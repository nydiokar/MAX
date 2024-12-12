from dataclasses import dataclass
from typing import Optional

@dataclass
class ResourceConfig:
    """Configuration for LLM resource management"""
    max_parallel_calls: int = 5
    cost_per_token: float = 0.0
    priority: int = 1
    local_only: bool = False

@dataclass
class BaseLlmConfig:
    """Base configuration for all LLM providers"""
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    resources: ResourceConfig = ResourceConfig()