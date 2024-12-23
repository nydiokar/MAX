from dataclasses import dataclass, field
from typing import Optional, List, Callable, Dict
from MAX.storage.protocols import TaskStorage, NotificationService
from MAX.retrievers import Retriever
from MAX.types import AgentConfig  # Assuming this is the correct path.
from MAX.tools.tool_config import ToolConfig

@dataclass
class TaskExpertOptions(AgentConfig):
    """
    Options/configuration for the TaskExpertAgent.
    Inherits base fields from AgentConfig if desired, or just define needed fields here.
    """
    storage_client: TaskStorage
    notification_service: NotificationService
    name: str
    description: str

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
    tool_configs: Dict[str, ToolConfig] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            self.name = "TaskExpert"
        if not self.description:
            self.description = "An agent that manages and coordinates tasks"
