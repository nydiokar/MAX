from typing import Dict, Any, Optional, List, Protocol
from MAX.storage.utils.types import Task

class TaskStorage(Protocol):
    """
    Protocol for task storage implementations.
    """
    async def save_task(self, task: Task) -> None:
        ...

    async def get_task(self, task_id: str) -> Optional[Task]:
        ...

    async def get_tasks(self, filters: Dict[str, Any]) -> List[Task]:
        ...

    async def delete_task(self, task_id: str) -> None:
        ...

    async def update_task(self, task_id: str, updates: Dict[str, Any]) -> None:
        ...

    async def save_interaction(self, interaction: Dict[str, Any]) -> None:
        ...


class NotificationService(Protocol):
    """
    Protocol for notification service implementations.
    """
    async def send(self, agent_id: str, notification: Dict[str, Any]) -> None:
        ...

    async def send_bulk(self, notifications: List[Dict[str, Any]]) -> None:
        ...