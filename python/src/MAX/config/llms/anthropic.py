from dataclasses import dataclass, field
from typing import Optional, List
from MAX.config.llms.base import BaseLlmConfig


@dataclass
class AnthropicConfig(BaseLlmConfig):
    """Anthropic-specific LLM configuration"""

    api_key: Optional[str] = None
    streaming: bool = False
    use_async: bool = True
    model_id: str = "claude-3-5-sonnet-20240620"
    max_retries: int = 3
    stop_sequences: List[str] = field(default_factory=list)
