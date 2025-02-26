from typing import Optional
from MAX.config.base_llm import BaseLlmConfig
from MAX.llms.base import AsyncLLMBase


# Import providers conditionally to avoid hard dependencies
def create_llm_provider(
    provider_type: str, config: Optional[BaseLlmConfig] = None
) -> AsyncLLMBase:
    """Factory function to create LLM providers with lazy imports"""
    if provider_type == "ollama":
        try:
            from MAX.llms.ollama import OllamaLLM

            return OllamaLLM(config)
        except ImportError:
            raise ImportError(
                "Ollama is not installed. Install it with 'pip install ollama'"
            )

    elif provider_type == "anthropic":
        try:
            from MAX.llms.anthropic import AnthropicLLM

            return AnthropicLLM(config)
        except ImportError:
            raise ImportError(
                "Anthropic client is not installed. Install it with 'pip install anthropic'"
            )

    raise ValueError(f"Unknown provider type: {provider_type}")


# Optional: Expose provider classes for type hints while avoiding import errors
try:
    from MAX.llms.ollama import OllamaLLM
except ImportError:
    OllamaLLM = None

try:
    from MAX.llms.anthropic import AnthropicLLM
except ImportError:
    AnthropicLLM = None

__all__ = ["create_llm_provider", "AsyncLLMBase", "OllamaLLM", "AnthropicLLM"]
