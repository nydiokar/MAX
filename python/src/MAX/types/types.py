from enum import Enum
from typing import List, Dict, Union, TypedDict, Optional, Any, Protocol
from dataclasses import dataclass, field
from datetime import datetime
from pydantic import BaseModel


class AgentTypes(Enum):
    DEFAULT = "Common Knowledge"
    CLASSIFIER = "classifier"


class ToolInput(TypedDict):
    userinput: str
    selected_agent: str
    confidence: str

class RequestMetadata(TypedDict):
    user_input: str
    agent_id: str
    agent_name: str
    user_id: str
    session_id: str
    additional_params :Optional[Dict[str, str]]
    error_type: Optional[str]


class ParticipantRole(Enum):
    ASSISTANT = "assistant"
    USER = "user"
    SYSTEM = "system"
    STATE = "state" 

@dataclass
class AgentResponse:
    metadata: Dict[str, Any]
    output: Any
    streaming: bool = False

class ConversationMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, str]]]
    timestamp: Optional[datetime] = None

class TimestampedMessage(ConversationMessage):
    timestamp: datetime

    class Config:
        arbitrary_types_allowed = True

TemplateVariables = Dict[str, Union[str, List[str]]]

class Config:
    arbitrary_types_allowed = True


@dataclass
class OrchestratorConfig:
    LOG_AGENT_CHAT: bool = False    # pylint: disable=invalid-name
    LOG_CLASSIFIER_CHAT: bool = False   # pylint: disable=invalid-name
    LOG_CLASSIFIER_RAW_OUTPUT: bool = False # pylint: disable=invalid-name
    LOG_CLASSIFIER_OUTPUT: bool = False # pylint: disable=invalid-name
    LOG_EXECUTION_TIMES: bool = False   # pylint: disable=invalid-name
    MAX_RETRIES: int = 3    # pylint: disable=invalid-name
    USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED: bool = True   # pylint: disable=invalid-name
    CLASSIFICATION_ERROR_MESSAGE: str = None
    NO_SELECTED_AGENT_MESSAGE: str = "I'm sorry, I couldn't determine how to handle your request.\
    Could you please rephrase it?"  # pylint: disable=invalid-name
    GENERAL_ROUTING_ERROR_MSG_MESSAGE: str = None
    MAX_MESSAGE_PAIRS_PER_AGENT: int = 100  # pylint: disable=invalid-name


@dataclass
class DiscordAdapterConfig:
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
    id: str
    url: str
    filename: str
    content_type: Optional[str] = None
    size: int = 0
    text_content: Optional[str] = None
    description: Optional[str] = None

@dataclass
class ProcessedContent:
    text: str
    attachments: List[Dict[str, any]] = field(default_factory=list)

class PlatformAdapter(Protocol):
    """Base protocol for platform adapters"""
    async def start(self):
        """Start the adapter"""
        ...

    def set_orchestrator(self, orchestrator):
        """Set the orchestrator reference"""
        ...

    async def process_message(self, message: Any):
        """Process incoming messages"""
        ...

class DatabaseConfig(TypedDict):
    database_type: str
    database_config: Dict[str, Any]

