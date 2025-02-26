from typing import Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field
from MAX.tools.tool_config import ToolConfig
from MAX.config.llms.ollama import OllamaConfig

# from MAX.config.llms.anthropic import AnthropicConfig  # Uncomment when needed
from MAX.config.base_llm import BaseLlmConfig
from MAX.config.llms.llm_config import LLM_CONFIGS
from MAX.retrievers import Retriever


class TaskExpertOptions(BaseModel):
    """
    Options/configuration for the TaskExpertAgent.
    If `llm_config` is provided, it overrides the default from LLM_CONFIGS.
    Otherwise, it attempts to use LLM_CONFIGS based on `llm_type` + `model_type`.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    callbacks: Optional[Any] = None
    model_id: Optional[str] = None
    description: str = "An agent that manages and coordinates tasks"
    region: Optional[str] = None
    save_chat: bool = True
    cloud_enabled: bool = True
    prefer_local: bool = False
    cloud_model_id: Optional[str] = None

    # Core references
    storage_client: Any
    notification_service: Any

    # Agent identity
    name: str = "TaskExpert"
    description: str = "An agent that manages and coordinates tasks"

    # LLM Configuration
    llm_type: str = "local"  # "local" for now, "cloud" later
    model_type: str = "general"  # "general", "code", "fast"
    # Update type hint to only include OllamaConfig for now
    llm_config: Optional[OllamaConfig] = None

    # LLM selection
    # llm_type: str = "local"     # "local" or "cloud"
    # model_type: str = "general" # "general", "code", "fast", "claude", etc.
    # llm_config: Optional[Union[OllamaConfig, AnthropicConfig]] = None

    # include AnthropicConfig later on

    @property
    def get_llm_config(self) -> BaseLlmConfig:
        """
        Return the final LLM config, either from user-provided `llm_config`
        or by default from `LLM_CONFIGS` using `llm_type` + `model_type`.
        """
        if self.llm_config:
            return self.llm_config

        # Fallback lookup in the dictionary
        try:
            return LLM_CONFIGS[self.llm_type][self.model_type]
        except KeyError:
            # Fallback to default local general config
            return LLM_CONFIGS["local"]["general"]

    # Task-related settings
    default_task_ttl: int = 7
    max_retries: int = 3
    retry_delay: float = 1.0

    # Retrievers and tool configs
    retriever: Optional[Retriever] = None
    tool_configs: Dict[str, ToolConfig] = Field(default_factory=dict)

    # Add this line
    save_chat: bool = False

    def __init__(self, **data: Any):
        """
        Override __init__ if you need extra logic beyond normal Pydantic behavior.
        This is optional; simply showing where you'd do post-processing if needed.
        """
        super().__init__(**data)
        # e.g., you might do: self.name = self.name.upper() if you like
