from typing import Dict, Any, Optional
from dataclasses import dataclass
from MAX.agents.task_expert import TaskStorage, NotificationService
from MAX.retrievers import Retriever

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
class TaskExpertOptions:
    # Required arguments first
    storage_client: TaskStorage
    notification_service: NotificationService
    agent_registry: Dict[str, Any]
    # Optional arguments
    name: str = "Task Expert"
    description: str = "Expert in managing and coordinating tasks"
    default_task_ttl: int = 7
    retriever: Optional[Retriever] = None
    inference_config: Optional[Dict[str, Any]] = None
    model_id: Optional[str] = None
    region: Optional[str] = None
    save_chat: bool = True
    callbacks: Optional[Any] = None

    def to_agent_options(self) -> AgentOptions:
        """Convert to AgentOptions for base Agent class"""
        return AgentOptions(
            name=self.name,
            description=self.description,
            model_id=self.model_id,
            region=self.region,
            save_chat=self.save_chat,
            callbacks=self.callbacks
        )