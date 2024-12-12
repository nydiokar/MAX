from typing import Dict, List, Union, AsyncIterable, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
    additional_params: Dict[str, any] = field(default_factory=dict)

@dataclass
class AgentResponse:
    metadata: AgentProcessingResult
    output: Union[Any, str]
    streaming: bool


class AgentCallbacks:
    def on_llm_new_token(self, token: str) -> None:
        # Default implementation
        pass

@dataclass
class AgentOptions:
    name: str
    description: str
    model_id: Optional[str] = None
    region: Optional[str] = None
    save_chat: bool = True
    callbacks: Optional[AgentCallbacks] = None
    # Add new fields
    cloud_enabled: bool = True
    prefer_local: bool = False
    cloud_model_id: Optional[str] = None
    resources: ResourceConfig = field(default_factory=ResourceConfig)


class Agent(ABC):
    def __init__(self, options: AgentOptions):
        self.name = options.name
        self.id = self.generate_key_from_name(options.name)
        self.description = options.description
        self.save_chat = options.save_chat
        self.callbacks = options.callbacks if options.callbacks is not None else AgentCallbacks()
        self.options = options
        
        # Initialize LLM providers
        self._llm_providers: Dict[str, Dict[str, Any]] = {}
        self._initialize_llm_providers()

    def _initialize_llm_providers(self):
        """Initialize default LLM providers"""
        if self.options.model_id:
            from MAX.llms.providers.ollama import OllamaLLM
            self._llm_providers["local"] = {
                "provider": OllamaLLM(BaseLlmConfig(
                    model=self.options.model_id,
                    resources=ResourceConfig(local_only=True)
                )),
            }
            
        if self.options.cloud_enabled and self.options.cloud_model_id:
            from MAX.llms.providers.antropic import AntropicLLM
            self._llm_providers["cloud"] = {
                "provider": AntropicLLM(BaseLlmConfig(
                    model=self.options.cloud_model_id,
                    resources=ResourceConfig(local_only=False)
                )),
            }

    async def _get_llm(self, requirements: Optional[Dict[str, Any]] = None) -> AsyncLLMBase:
        """Get appropriate LLM based on requirements"""
        requirements = requirements or {}
        
        if self.options.prefer_local and "local" in self._llm_providers:
            return self._llm_providers["local"]["provider"]
            
        available_providers = [
            provider_info["provider"]
            for provider_info in self._llm_providers.values()
            if provider_info["provider"].available
            and self._meets_requirements(provider_info["provider"].config.resources, requirements)
        ]
        
        if not available_providers:
            # Return first provider if no suitable one found
            return next(iter(self._llm_providers.values()))["provider"]
            
        # Return provider with highest priority
        return max(available_providers, 
                  key=lambda p: p.config.resources.priority)

    def _meets_requirements(self, resources: ResourceConfig, requirements: Dict[str, Any]) -> bool:
        if requirements.get("local_only") and not resources.local_only:
            return False
        if requirements.get("max_cost") and resources.cost_per_token > requirements.get("max_cost"):
            return False
        return True

    def is_streaming_enabled(self) -> bool:
        return False

    @staticmethod
    def generate_key_from_name(name: str) -> str:
        import re
        # Remove special characters and replace spaces with hyphens
        key = re.sub(r'[^a-zA-Z\s-]', '', name)
        key = re.sub(r'\s+', '-', key)
        return key.lower()

    @abstractmethod
    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: List[ConversationMessage],
        additional_params: Optional[Dict[str, str]] = None
    ) -> Union[ConversationMessage, AsyncIterable[any]]:
        pass
