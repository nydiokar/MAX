from typing import Dict, List, Optional
from anthropic import AsyncAnthropic
from anthropic.types import APIError
from MAX.llms.base import AsyncLLMBase
from MAX.config.llms.anthropic import AnthropicConfig
from MAX.llms.utils.exceptions import (
    LLMProviderError,
    LLMRateLimitError,
    LLMAuthenticationError,
)
from MAX.llms.utils import async_retry_with_backoff
import logging
from MAX.llms.utils.exceptions import LLMConfigError

logger = logging.getLogger(__name__)


class AnthropicLLM(AsyncLLMBase):
    """Anthropic Claude implementation with enhanced error handling and retries"""

    def __init__(self, config: Optional[AnthropicConfig] = None):
        super().__init__(config or AnthropicConfig())
        if not self.config.api_key:
            raise LLMConfigError("API key required for Anthropic provider")
        self.client = AsyncAnthropic(api_key=self.config.api_key)

    @async_retry_with_backoff(
        max_retries=3, initial_delay=1.0, exponential_base=2.0
    )
    async def _generate(self, messages: List[Dict[str, str]]) -> str:
        """Implement Anthropic-specific generation with error handling"""
        try:
            logger.debug(f"Generating response with {len(messages)} messages")
            response = await self.client.messages.create(
                model=self.config.model_id,
                max_tokens=self.config.max_tokens,
                messages=messages,
                temperature=self.config.temperature,
                stop_sequences=self.config.stop_sequences,
            )
            logger.debug("Successfully generated response")
            return response.content

        except APIError as e:
            if "rate_limit" in str(e).lower():
                raise LLMRateLimitError(str(e), provider="anthropic")
            elif "unauthorized" in str(e).lower():
                raise LLMAuthenticationError(str(e), provider="anthropic")
            else:
                raise LLMProviderError(str(e), provider="anthropic")
        except Exception as e:
            logger.error(f"Unexpected error in Anthropic provider: {str(e)}")
            raise LLMProviderError(str(e), provider="anthropic")
