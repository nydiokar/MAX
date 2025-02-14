# ollama.py
from dataclasses import dataclass, field
from typing import List
from .base import BaseLlmConfig, ResourceConfig


@dataclass
class OllamaConfig(BaseLlmConfig):
    """
    Ollama-specific configuration extending the base LLM config.

    Attributes:
        model_type (str): 'general', 'code', or 'fast' to indicate usage pattern.
        context_size (int): The context window size for Ollama.
        repeat_penalty (float): Factor for penalizing repeated tokens.
        stop_sequences (List[str]): Sequences at which generation should stop.
    """

    model_type: str = "general"
    context_size: int = 4096
    repeat_penalty: float = 1.1
    stop_sequences: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Default to local Ollama if no API base is set
        if not self.api_base_url:
            self.api_base_url = "http://localhost:11434"
        # Default resource config if none is provided
        if not self.resources:
            self.resources = ResourceConfig(local_only=True)
