from typing import Dict, List, Union, AsyncIterable, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re

from MAX.types import ConversationMessage
from MAX.config.llms.base import ResourceConfig, BaseLlmConfig
from MAX.llms.base import AsyncLLMBase


@dataclass
class AgentProcessingResult:
    user_input: str
    agent_id: str
    agent_name: str
    user_id: str
    session_id: str
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    metadata: AgentProcessingResult
    output: Union[Any, str]
    streaming: bool


class AgentCallbacks:
    """Optional callbacks for token streaming or custom logging."""

    def on_llm_new_token(self, token: str) -> None:
        """Called whenever a new token is streamed from the LLM."""
        pass


@dataclass
class AgentOptions:
    """Base options/configuration for any agent."""

    name: str
    description: str
    model_id: Optional[str] = None
    save_chat: bool = True
    callbacks: Optional[AgentCallbacks] = None
    resources: ResourceConfig = field(default_factory=ResourceConfig)


class Agent(ABC):
    """
    Base abstract Agent class. Other specialized agents (like TaskExpertAgent)
    should inherit from this and implement their domain-specific logic.
    """

    def __init__(self, options: AgentOptions):
        self.name = options.name
        self.id = self.generate_key_from_name(options.name)
        self.description = options.description
        self.save_chat = options.save_chat
        self.callbacks = options.callbacks if options.callbacks else AgentCallbacks()
        self.options = options
        self._llm = self._create_llm_provider(BaseLlmConfig(
            model=options.model_id,
            resources=options.resources
        ))

    @abstractmethod
    def _create_llm_provider(self, llm_config: BaseLlmConfig) -> AsyncLLMBase:
        """Create an LLM provider. Must be implemented by subclasses."""
        raise NotImplementedError(
            "Please implement _create_llm_provider() in your agent class"
        )

    def is_streaming_enabled(self) -> bool:
        """
        Whether streaming is enabled for this agent.
        Override if your agent uses a streaming LLM.
        """
        return False

    @staticmethod
    def generate_key_from_name(name: str) -> str:
        """
        Generate a lowercase key from an agent name by removing special chars
        and replacing whitespace with hyphens.
        """
        key = re.sub(r"[^a-zA-Z\s-]", "", name)
        key = re.sub(r"\s+", "-", key).strip("-")
        return key.lower()

    @abstractmethod
    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: List[ConversationMessage],
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> Union[ConversationMessage, AsyncIterable[Any]]:
        """
        Subclasses must implement how they handle a request:
          - Possibly call an LLM
          - Possibly store/retrieve conversation context
          - Return a final response (or a streaming generator)
        """
        pass
