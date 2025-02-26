# base_llm.py
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, List

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
    temperature: float = 0.7
    max_tokens: int = 1024
    streaming: bool = False
    resources: Optional[ResourceConfig] = None
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    context_window: int = 4096
    stop_sequences: Optional[List[str]] = None
    timeout: float = 30.0

    @property
    @abstractmethod
    def model(self) -> str:
        """Each LLM provider must implement how to get the model identifier"""
        pass

    @classmethod
    def create(cls, **kwargs) -> 'BaseLlmConfig':
        """Factory method to create config instances"""
        config = object.__new__(cls)
        
        # Get all fields from dataclass
        fields = cls.__dataclass_fields__.keys()
        
        # Set defaults from dataclass defaults
        values = {
            field: getattr(cls, field).default 
            for field in fields
            if hasattr(cls, field) and hasattr(getattr(cls, field), 'default')
        }
        
        # Update with provided kwargs
        values.update(kwargs)
        
        # Set all attributes
        for key, value in values.items():
            setattr(config, key, value)
            
        return config

    def to_dict(self) -> Dict:
        """Convert config to dictionary for API requests"""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": self.stop_sequences,
            "stream": self.streaming,
        }

    def __str__(self) -> str:
        """Safe string representation that hides API key"""
        return f"LLMConfig(model={self.model}, api_key=[REDACTED])"

    def __repr__(self) -> str:
        """Safe repr that hides API key"""
        return self.__str__()
