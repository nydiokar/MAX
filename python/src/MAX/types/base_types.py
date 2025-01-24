from enum import Enum
from typing import List, Dict, Union, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pydantic import BaseModel


class AgentTypes(Enum):
    DEFAULT = "Common Knowledge"
    CLASSIFIER = "classifier"
    TASK_EXPERT = "task_expert"


class ToolInput(BaseModel):
    """
    Generic structure for tool input.
    Adjust or extend as needed based on your actual tool input format.
    """

    userinput: str
    selected_agent: str
    confidence: str


class RequestMetadata(BaseModel):
    """
    Metadata associated with a user request.
    """

    user_input: str
    agent_id: str
    agent_name: str
    user_id: str
    session_id: str
    additional_params: Optional[Dict[str, str]] = None
    error_type: Optional[str] = None


class ParticipantRole(Enum):
    ASSISTANT = "assistant"
    USER = "user"
    SYSTEM = "system"
    STATE = "state"


@dataclass
class AgentResponse:
    """
    Represents a response from an agent.
    """

    metadata: Dict[str, Any]
    output: Any
    streaming: bool = False


class ConversationMessage(BaseModel):
    """
    Represents a single message in a conversation.
    """

    role: ParticipantRole
    content: Union[str, List[Dict[str, str]]]
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = {}


class TimestampedMessage(ConversationMessage):
    timestamp: datetime

    class Config:
        arbitrary_types_allowed = True


TemplateVariables = Dict[str, Union[str, List[str]]]


@dataclass
class OrchestratorConfig:
    """
    Configuration for the orchestrator (routing, logging, etc.).
    Adjust fields as necessary.
    """

    LOG_AGENT_CHAT: bool = False
    LOG_CLASSIFIER_CHAT: bool = False
    LOG_CLASSIFIER_RAW_OUTPUT: bool = False
    LOG_CLASSIFIER_OUTPUT: bool = False
    LOG_EXECUTION_TIMES: bool = False
    MAX_RETRIES: int = 3
    USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED: bool = True
    CLASSIFICATION_ERROR_MESSAGE: Optional[str] = None
    NO_SELECTED_AGENT_MESSAGE: str = (
        "I'm sorry, I couldn't determine how to handle your request.\n"
        "Could you please rephrase it?"
    )
    GENERAL_ROUTING_ERROR_MSG_MESSAGE: Optional[str] = None
    MAX_MESSAGE_PAIRS_PER_AGENT: int = 100


@dataclass
class DiscordAdapterConfig:
    """
    Discord-specific adapter configuration.
    """

    command_prefix: str = "!"
    enable_dm: bool = True
    allowed_channels: Optional[List[int]] = None
    max_message_length: int = 2000
    cache_message_count: int = 20
    retry_attempts: int = 3
    retry_delay: int = 2  # seconds
    allowed_guilds: Optional[List[int]] = None
    log_messages: bool = False


@dataclass
class DiscordAttachment:
    """
    Representation of an attachment in Discord.
    """

    id: str
    url: str
    filename: str
    content_type: Optional[str] = None
    size: int = 0
    text_content: Optional[str] = None
    description: Optional[str] = None


@dataclass
class ProcessedContent:
    """
    Represents processed content (text plus attachments).
    """

    text: str
    attachments: List[Dict[str, Any]] = field(default_factory=list)


class PlatformAdapter:
    """
    Base protocol for platform adapters.
    """

    async def start(self):
        """Start the adapter."""
        raise NotImplementedError()

    def set_orchestrator(self, orchestrator: Any):
        """Set the orchestrator reference."""
        raise NotImplementedError()

    async def process_message(self, message: Any):
        """Process incoming messages."""
        raise NotImplementedError()


class DatabaseConfig(BaseModel):
    """
    Database configuration for storage.
    """

    database_type: str
    database_config: Dict[str, Any]


class AgentToolConfig(BaseModel):
    """
    Configuration for agent tools.
    """

    allowed_tools: List[str]
    tool_permissions: Dict[str, str] = {}
    rate_limits: Dict[str, int] = {}


class AgentConfig(BaseModel):
    """
    Base configuration for agents.
    """

    name: str
    description: str
    agent_type: AgentTypes
    tools: AgentToolConfig
    capabilities: List[str] = []
    max_concurrent_tasks: int = 1
