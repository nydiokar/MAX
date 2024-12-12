from typing import Optional
from MAX.config.llms.base import BaseLlmConfig
from MAX.llms.base import AsyncLLMBase
from MAX.llms.ollama import OllamaLLM
from MAX.llms.antropic import AnthropicLLM

def create_llm_provider(provider_type: str, config: Optional[BaseLlmConfig] = None) -> AsyncLLMBase:
    """Factory function to create LLM providers"""
    providers = {
        "ollama": OllamaLLM,
        "anthropic": AnthropicLLM,
        # Add other providers here
    }
    
    if provider_type not in providers:
        raise ValueError(f"Unknown provider type: {provider_type}")
        
    return providers[provider_type](config)
