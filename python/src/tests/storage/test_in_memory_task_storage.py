# in_memory_fakes.py
from typing import Dict, Optional, List
from datetime import datetime
from MAX.storage.utils.types import Task
from MAX.storage.utils.protocols import TaskStorage, NotificationService

class InMemoryTaskStorage(TaskStorage):
    """
    A simple in-memory TaskStorage that stores Tasks in a dictionary.
    Suitable for testing without an actual database.
    """

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.interactions: List[Dict] = []

    async def save_task(self, task: Task) -> None:
        """
        Save or update a task in the in-memory store.
        """
        self.tasks[task.id] = task

    async def get_task(self, task_id: str) -> Optional[Task]:
        """
        Retrieve a task by its ID.
        """
        return self.tasks.get(task_id)

    async def save_interaction(self, interaction_data: dict) -> None:
        """
        Save an interaction record (for conversation logs or usage).
        """
        self.interactions.append(interaction_data)

    async def fetch_task(self, task_id: str) -> Optional[Task]:
        """
        Fetch a task by ID, used by the dependency-verification logic.
        """
        return self.tasks.get(task_id)


class InMemoryNotificationService(NotificationService):
    """
    In-memory notification service that just logs sent notifications.
    """

    def __init__(self):
        self.sent_notifications: List[Dict] = []

    async def send(self, recipient: str, data: dict) -> None:
        """
        Store the notification data in memory, simulating a send.
        """
        self.sent_notifications.append({"recipient": recipient, "data": data})
