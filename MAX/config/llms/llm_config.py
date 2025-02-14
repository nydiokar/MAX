# llm_config.py
from .base import ResourceConfig
from .ollama import OllamaConfig

# Import for cloud providers - uncomment when needed
# from MAX.config.llms.anthropic import AnthropicConfig

LLM_CONFIGS = {
    "local": {
        "general": OllamaConfig(
            model="llama3.1:8b-instruct-q8_0",
            resources=ResourceConfig(
                local_only=True, priority=1, max_parallel_calls=3
            ),
            model_type="general",
        ),
        "code": OllamaConfig(
            model="llama3.1:8b-instruct-q8_0",
            resources=ResourceConfig(local_only=True, priority=2),
            model_type="code",
        ),
        "fast": OllamaConfig(
            model="llama3.1:8b-instruct-q8_0",
            resources=ResourceConfig(
                local_only=True, priority=1, max_parallel_calls=5
            ),
            model_type="fast",
        ),
    },
    # "cloud": {
    #    "claude": AnthropicConfig(
    #        model="claude-3-sonnet",
    #        resources=ResourceConfig(local_only=False, priority=3, cost_per_token=0.01)
    #        ),
    #    "claude-fast": AnthropicConfig(
    #        model="claude-instant",
    #        resources=ResourceConfig(local_only=False, priority=2, cost_per_token=0.005)
    #    )
}
# Cloud provider configurations - uncomment and modify when needed
# "cloud": {
#     "claude": AnthropicConfig(
#         model="claude-3-sonnet",
#         resources=ResourceConfig(
#             local_only=False,
#             priority=3,
#             cost_per_token=0.01
#         )
#     ),
#     "claude-fast": AnthropicConfig(
#         model="claude-instant",
#         resources=ResourceConfig(
#             local_only=False,
#             priority=2,
#             cost_per_token=0.005
#         )
#     )
# }

# Additional provider-specific configurations can be added here
# Example structure for future cloud providers:
#
# CLOUD_PROVIDER_CONFIGS = {
#     "anthropic": {
#         "api_version": "2023-06-01",
#         "default_model": "claude-3-sonnet",
#         "timeout": 30.0
#     }
# }
