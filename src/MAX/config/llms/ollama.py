from dataclasses import dataclass, field
from typing import Optional, List
from MAX.config.llms.base import BaseLlmConfig, ResourceConfig

@dataclass
class OllamaConfig(BaseLlmConfig):
    """Ollama-specific configuration"""
    fallback_model: Optional[str] = None
    model_type: str = "general"
    context_size: int = 4096
    repeat_penalty: float = 1.1
    stop_sequences: List[str] = field(default_factory=list)

# In your llm_configs.py
LLM_CONFIGS = {
    "local": {
        "general": OllamaConfig(
            model="llama2",
            resources=ResourceConfig(
                local_only=True,
                priority=1,
                max_parallel_calls=3
            )
        ),
        "code": OllamaConfig(
            model="codellama",
            model_type="code",
            resources=ResourceConfig(
                local_only=True,
                priority=2
            )
        ),
        "fast": OllamaConfig(
            model="mistral",
            model_type="fast",
            resources=ResourceConfig(
                local_only=True,
                priority=1,
                max_parallel_calls=5  # Allow more parallel calls for fast model
            )
        )
    }
}