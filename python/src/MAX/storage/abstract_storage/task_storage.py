from abc import ABC, abstractmethod
from typing import List, Optional
from MAX.storage.utils.types import Task, ExecutionHistoryEntry


class TaskStorage(ABC):
    """
    Abstract base class for task storage backends.
    Provides shared logic and defines the interface for task storage implementations.
    """

    @abstractmethod
    async def create_task(self, task_data: Task) -> str:
        """
        Create a new task in the storage backend.

        Args:
            task_data (Task): A Task object containing task details, such as
                title, description, priority, status, assigned agent, due date, and more.

        Returns:
            str: The unique identifier of the newly created task.
        """
        pass

    @abstractmethod
    async def update_task(self, task_id: str, updates: Task) -> bool:
        """
        Update an existing task with new information.

        Args:
            task_id (str): The unique identifier of the task to be updated.
            updates (Task): A Task object containing the fields to be updated,
                such as status, priority, due date, or description.

        Returns:
            bool: True if the update was successful, False otherwise.
        """
        pass

    @abstractmethod
    async def fetch_task(self, task_id: str) -> Optional[Task]:
        """
        Fetch the details of a specific task by its ID.

        Args:
            task_id (str): The unique identifier of the task to fetch.

        Returns:
            Optional[Task]: A Task object containing task details, or None if the task does not exist.
        """
        pass

    @abstractmethod
    async def delete_task(self, task_id: str) -> bool:
        """
        Delete a task and its associated data (e.g., execution history) from the storage backend.

        Args:
            task_id (str): The unique identifier of the task to delete.

        Returns:
            bool: True if the task was successfully deleted, False otherwise.
        """
        pass

    @abstractmethod
    async def log_execution_history(
        self,
        task_id: str,
        status: str,
        changed_by: str,
        comment: Optional[str] = None,
    ) -> None:
        """
        Log a change in the execution history of a task.

        Args:
            task_id (str): The unique identifier of the task for which the change is logged.
            status (str): The new status of the task (e.g., "In Progress", "Completed").
            changed_by (str): The identifier of the agent or user making the change.
            comment (Optional[str]): An optional comment describing the reason for the change.

        Returns:
            None
        """
        pass

    @abstractmethod
    async def fetch_execution_history(
        self, task_id: str
    ) -> List[ExecutionHistoryEntry]:
        """
        Fetch the execution history of a task.

        Args:
            task_id (str): The unique identifier of the task whose history is to be fetched.

        Returns:
            List[ExecutionHistoryEntry]: A list of ExecutionHistoryEntry objects, each representing a logged change
                in the task's execution history, sorted by timestamp.
        """
        pass

    @abstractmethod
    async def search_tasks(
        self, filters: Optional[Task] = None, limit: Optional[int] = 10
    ) -> List[Task]:
        """
        Search for tasks matching specified filters.

        Args:
            filters (Optional[Task]): A Task object specifying query criteria,
                such as status, priority, assigned agent, or tags.
            limit (Optional[int]): The maximum number of tasks to return. Defaults to 10.

        Returns:
            List[Task]: A list of Task objects representing the matching tasks.
        """
        pass

    @abstractmethod
    async def add_task_dependency(
        self, task_id: str, dependency_id: str
    ) -> bool:
        """
        Add a dependency to a task.

        Args:
            task_id (str): The unique identifier of the task.
            dependency_id (str): The unique identifier of the task to be added as a dependency.

        Returns:
            bool: True if the dependency was successfully added, False otherwise.
        """
        pass

    @abstractmethod
    async def remove_task_dependency(
        self, task_id: str, dependency_id: str
    ) -> bool:
        """
        Remove a dependency from a task.

        Args:
            task_id (str): The unique identifier of the task.
            dependency_id (str): The unique identifier of the dependency to be removed.

        Returns:
            bool: True if the dependency was successfully removed, False otherwise.
        """
        pass

    @abstractmethod
    async def check_health(self) -> bool:
        """
        Check the health of the task storage backend.

        Returns:
            bool: True if the storage backend is healthy and operational, False otherwise.
        """
        pass

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the storage connection and perform any necessary setup.

        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """
        Clean up resources and close connections.

        Returns:
            None
        """
        pass
