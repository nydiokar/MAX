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


class ParticipantRole(str, Enum):
    """Roles for conversation participants"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageType(str, Enum):
    """Types of messages in the system"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"


@dataclass
class AgentResponse:
    """Standard response format for agents"""
    content: Union[str, List[Dict[str, str]]]
    confidence: float
    metadata: Optional[Dict[str, Any]] = None
    agent_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    message_type: MessageType = MessageType.TEXT
    status: str = "success"
    error: Optional[str] = None


MessageContent = Union[str, List[Dict[str, str]]]


@dataclass
class BaseMessage:
    """Base message type for the system"""
    role: ParticipantRole
    content: MessageContent
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ConversationMessage(BaseMessage):
    """Standard message format for conversations"""
    message_type: MessageType = MessageType.TEXT


@dataclass
class AgentMessage:
    """Message specific to agent communications"""
    agent_id: str
    role: ParticipantRole
    content: MessageContent
    confidence: float = 0.0
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    message_type: MessageType = MessageType.TEXT


@dataclass
class TimestampedMessage:
    """Message with timestamp information"""
    message: ConversationMessage
    timestamp: datetime = datetime.now()
    metadata: Optional[Dict[str, Any]] = None

    @property
    def role(self) -> ParticipantRole:
        return self.message.role

    @property
    def content(self) -> MessageContent:
        return self.message.content


TemplateVariables = Dict[str, Union[str, List[str]]]


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


@dataclass
class ResourceConfig:
    """Configuration for resource management"""
    max_parallel_calls: int = 5
    cost_per_token: float = 0.0
    priority: int = 1
    local_only: bool = False


class AgentProviderType(str, Enum):
    """Types of agent providers"""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    CUSTOM = "custom"
    SYSTEM = "system"


class AgentTypes(str, Enum):
    """Types of agents"""
    RECURSIVE_THINKER = "recursive_thinker"
    TASK_EXECUTOR = "task_executor"
    CONVERSATION = "conversation"
    SYSTEM = "system"


@dataclass
class AgentMetadata:
    """Metadata for agents"""
    provider: AgentProviderType
    type: AgentTypes
    capabilities: List[str]
    model_id: str
    version: str
    created_at: str
    metadata: Optional[Dict[str, Any]] = None