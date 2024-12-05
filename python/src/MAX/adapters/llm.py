from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import ollama
from anthropic import AsyncAnthropic

# Base Provider Class
class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        pass

# Specific Providers
class OllamaProvider(LLMProvider):
    def __init__(self, model_id: str = "hermes3:8b", **kwargs):
        self.model_id = model_id
        self.kwargs = kwargs
        
    async def generate(self, prompt: str, **kwargs) -> str:
        response = await ollama.chat(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            **{**self.kwargs, **kwargs}
        )
        return response['message']['content']

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model_id: str = "claude-3-sonnet", **kwargs):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model_id = model_id
        self.kwargs = kwargs

    async def generate(self, prompt: str, **kwargs) -> str:
        response = await self.client.messages.create(
            model=self.model_id,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
            **{**self.kwargs, **kwargs}
        )
        return response.content

# Factory Function
def create_llm_provider(
    provider_type: str,
    model_id: Optional[str] = None,
    **kwargs
) -> LLMProvider:
    """Create an LLM provider instance based on type"""
    if provider_type == "ollama":
        return OllamaProvider(model_id or "hermes3:8b", **kwargs)
    elif provider_type == "anthropic":
        if "api_key" not in kwargs:
            raise ValueError("api_key required for Anthropic provider")
        return AnthropicProvider(
            api_key=kwargs.pop("api_key"),
            model_id=model_id or "claude-3-sonnet",
            **kwargs
        )
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")