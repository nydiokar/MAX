from dataclasses import dataclass
from MAX.config.llms.base import BaseLlmConfig
from MAX.config.models import AnthropicModels

@dataclass
class AnthropicConfig(BaseLlmConfig):
    """Anthropic-specific configuration"""
    model: str = AnthropicModels.HAIKU
    max_tokens: int = 4096  # Anthropic-specific default
    api_base_url: str = "https://api.anthropic.com/v1"
