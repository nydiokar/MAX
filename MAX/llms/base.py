from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from MAX.config.base_llm import BaseLlmConfig


class AsyncLLMBase(ABC):
    """Base class for all LLM implementations"""

    def __init__(self, config: Optional[BaseLlmConfig] = None):
        self.config = config or BaseLlmConfig()
        self._current_tasks = 0

    @property
    def available(self) -> bool:
        """Check if provider can accept more tasks"""
        return self._current_tasks < self.config.resources.max_parallel_calls

    async def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Template method pattern for response generation"""
        self._current_tasks += 1
        try:
            return await self._generate(messages)
        finally:
            self._current_tasks -= 1

    @abstractmethod
    async def _generate(self, messages: List[Dict[str, str]]) -> str:
        """Actual implementation by specific providers"""
        pass
