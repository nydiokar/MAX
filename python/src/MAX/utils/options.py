from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from MAX.retrievers import Retriever   
from pydantic import field_validator
from MAX.utils.interfaces import TaskStorage, NotificationService

@dataclass
class BaseAgentOptions:
    """Base class for all agent options with required fields"""
    name: str
    description: str

@dataclass
class AgentOptions(BaseAgentOptions):
    """Standard agent options with optional fields"""
    model_id: Optional[str] = None
    region: Optional[str] = None
    save_chat: bool = True
    callbacks: Optional[Any] = None

@dataclass
class AnthropicAgentOptions(AgentOptions):
    api_key: Optional[str] = None
    client: Optional[Any] = None
    model_id: str = "claude-3-5-sonnet-20240620"
    streaming: Optional[bool] = False
    inference_config: Optional[Dict[str, Any]] = None
    retriever: Optional[Retriever] = None
    tool_config: Optional[Dict[str, Any]] = None
    custom_system_prompt: Optional[Dict[str, Any]] = None

@dataclass
class TaskExpertOptions(BaseAgentOptions):
    # Required arguments must come first
    storage_client: TaskStorage
    notification_service: NotificationService
    name: str
    description: str
    
    # Optional arguments with defaults must come after required arguments
    model_id: str = "hermes3:8b"
    region: str = "us-east-1"
    save_chat: bool = True
    callbacks: Optional[List[Callable]] = None
    fallback_model_id: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 0.9
    default_task_ttl: int = 7
    retriever: Optional[Retriever] = None
    
    def __post_init__(self):
        if not self.name:
            self.name = "TaskExpert"
        if not self.description:
            self.description = "An agent that manages and coordinates tasks"
    
    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Temperature must be between 0 and 1')
        return v
    
    @field_validator('model_id')
    @classmethod
    def validate_model_id(cls, v):
        valid_models = ["hermes3:8b", "llama3.1:8b-instruct-q8_0", "llama2", "mistral"]
        if v not in valid_models:
            raise ValueError(f'Model must be one of: {valid_models}')
        return v

    def to_agent_options(self) -> AgentOptions:
        """Convert to base AgentOptions"""
        return AgentOptions(
            name=self.name,
            description=self.description,
            model_id=self.model_id,
            region=self.region,
            save_chat=self.save_chat,
            callbacks=self.callbacks
        )