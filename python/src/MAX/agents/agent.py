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
    region: Optional[str] = None
    save_chat: bool = True
    callbacks: Optional[AgentCallbacks] = None

    # Additional fields for LLM usage
    cloud_enabled: bool = True
    prefer_local: bool = False
    cloud_model_id: Optional[str] = None

    # Resource config helps define how we schedule or cost usage
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
        self.callbacks = (
            options.callbacks if options.callbacks else AgentCallbacks()
        )
        self.options = options

        # Dictionary to store multiple LLM providers (local, cloud, etc.)
        self._llm_providers: Dict[str, Dict[str, Any]] = {}
        self._initialize_llm_providers()

    def _initialize_llm_providers(self) -> None:
        """
        Initialize default LLM providers based on the AgentOptions.
        This base method is intentionally generic and does NOT import
        specific providers like Ollama or Anthropic.

        In your specialized agent or via config, you can override this
        or add new providers. For example:
          - A local provider with the user-specified model_id
          - A cloud provider with cloud_model_id
        """
        if self.options.model_id:
            # Example: a local LLM using model_id
            # We create a config for it:
            local_llm_config = BaseLlmConfig(
                model=self.options.model_id,
                resources=ResourceConfig(local_only=True),
            )
            # Then create the LLM via an abstract factory method
            local_llm = self._create_llm_provider(local_llm_config)
            self._llm_providers["local"] = {
                "provider": local_llm,
            }

        if self.options.cloud_enabled and self.options.cloud_model_id:
            # Example: a cloud-based LLM using cloud_model_id
            cloud_llm_config = BaseLlmConfig(
                model=self.options.cloud_model_id,
                resources=ResourceConfig(local_only=False),
            )
            cloud_llm = self._create_llm_provider(cloud_llm_config)
            self._llm_providers["cloud"] = {
                "provider": cloud_llm,
            }

    def _create_llm_provider(self, llm_config: BaseLlmConfig) -> AsyncLLMBase:
        """
        Placeholder/factory method for creating an LLM provider.
        By default, it raises NotImplementedError.
        Override in your specialized agent or a separate factory
        to return an instance of OllamaLLM, AnthropicLLM, etc.
        """
        raise NotImplementedError(
            "Please override _create_llm_provider() in a subclass or your agent’s logic "
            "to return a real LLM instance."
        )

    async def _get_llm(
        self, requirements: Optional[Dict[str, Any]] = None
    ) -> AsyncLLMBase:
        """
        Returns an appropriate LLM based on 'requirements' (e.g., cost, local_only).
        If 'prefer_local' is True, local provider is tried first if available.
        Otherwise, picks among the providers that meet the requirements,
        returning the one with the highest priority if multiple exist.
        """
        requirements = requirements or {}

        # If user wants local first
        if self.options.prefer_local and "local" in self._llm_providers:
            return self._llm_providers["local"]["provider"]

        # Filter providers that meet the requirements
        available_providers = []
        for provider_info in self._llm_providers.values():
            provider = provider_info["provider"]
            # We'll assume a property 'available = True' on the provider, or default to True
            if getattr(
                provider, "available", True
            ) and self._meets_requirements(
                provider.config.resources, requirements
            ):
                available_providers.append(provider)

        # If none meet the requirements, fallback to the first
        if not available_providers:
            return next(iter(self._llm_providers.values()))["provider"]

        # Return the provider with the highest priority
        return max(
            available_providers, key=lambda p: p.config.resources.priority
        )

    def _meets_requirements(
        self, resources: ResourceConfig, requirements: Dict[str, Any]
    ) -> bool:
        """Check if a given LLM resource config meets the specified requirements."""
        if requirements.get("local_only") and not resources.local_only:
            return False
        if (
            requirements.get("max_cost")
            and resources.cost_per_token > requirements["max_cost"]
        ):
            return False
        return True

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
