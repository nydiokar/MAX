from .base import ResourceConfig
from MAX.config.llms.ollama import OllamaConfig
from MAX.config.llms.anthropic import AnthropicConfig

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
                max_parallel_calls=5
            )
        )
    }, 
    "cloud": {
        "claude": AnthropicConfig(
            model="claude-3-sonnet",
            resources=ResourceConfig(
                local_only=False,
                priority=3,
                cost_per_token=0.01
            )
        ),
        "claude-fast": AnthropicConfig(
            model="claude-instant",
            resources=ResourceConfig(
                local_only=False,
                priority=2,
                cost_per_token=0.005
            )
        )
    }
}