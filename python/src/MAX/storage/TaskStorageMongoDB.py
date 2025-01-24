from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import random
from enum import Enum

from motor.motor_asyncio import AsyncIOMotorClient
from MAX.utils import Logger

from MAX.storage.abstract_storage.task_storage import TaskStorage
from MAX.storage.utils.types import (
    Task,
    ExecutionHistoryEntry,
    TaskStatus,
    TaskPriority,
)

from MAX.storage.data_hub_cache import DataHubCache


class TaskStorageError(Exception):
    """Base exception for task storage errors."""

    pass


class DependencyCycleError(TaskStorageError):
    """Raised when a dependency cycle is detected."""

    pass


class ValidationError(TaskStorageError):
    """Raised when task data validation fails."""

    pass


class MongoDBTaskStorage(TaskStorage):
    """
    MongoDB implementation for managing tasks and execution history.
    Provides methods for creating, updating, fetching, deleting, and
    logging transitions (execution history).
    """

    def __init__(
        self,
        client: AsyncIOMotorClient,
        db_name: str,
        cache: Optional[DataHubCache] = None,
    ):
        """
        :param client: An existing AsyncIOMotorClient instance.
        :param db_name: Name of the database to store tasks and execution history.
        :param cache: Optional DataHubCache instance for invalidating task cache.
        """
        self.client = client
        self.db = self.client[db_name]
        self.tasks = self.db["tasks"]
        self.execution_history = self.db["execution_history"]

        # Optionally attach the cache for invalidation
        self.cache = cache

    # ----------------------------------------------------------------------
    # Initialization & Health Checks
    # ----------------------------------------------------------------------

    async def initialize(self) -> bool:
        """
        Check MongoDB health and set up indices.
        Returns True if successful, False otherwise.
        """
        try:
            await self.check_health()
            await self._setup_indices()
            return True
        except Exception as e:
            Logger.error(f"Failed to initialize MongoDB: {str(e)}")
            return False

    async def check_health(self) -> bool:
        """Check MongoDB connection health by running a 'ping' command."""
        try:
            await self.db.command("ping")
            return True
        except Exception as e:
            Logger.error(f"MongoDB health check failed: {str(e)}")
            return False

    async def _setup_indices(self):
        """
        Set up indices for the tasks and execution history collections.
        """
        try:
            # Existing indices
            await self.tasks.create_index([("task_id", 1)], unique=True)
            await self.tasks.create_index(
                [("status", 1), ("priority", 1), ("due_date", 1)]
            )
            await self.tasks.create_index(
                [("status", 1), ("assigned_agent", 1)]
            )
            await self.tasks.create_index([("priority", 1)])
            await self.tasks.create_index([("due_date", 1)])
            await self.tasks.create_index([("dependencies", 1)])

            # New indices for priority-based querying
            await self.tasks.create_index(
                [("status", 1), ("priority_score", -1), ("due_date", 1)]
            )

            # Text indices for content search
            await self.tasks.create_index(
                [("title", "text"), ("description", "text")]
            )

            # Execution history indices
            await self.execution_history.create_index(
                [("task_id", 1), ("timestamp", -1)]
            )
            await self.execution_history.create_index(
                [("entry_id", 1)], unique=True
            )

        except Exception as e:
            Logger.error(f"Failed to setup MongoDB indices: {str(e)}")
            raise

    # ----------------------------------------------------------------------
    # CRUD Operations for Tasks
    # ----------------------------------------------------------------------

    async def create_task(self, task_data: Task) -> str:
        """
        Create a new task (with validation). Automatically assigns a task ID if missing.
        Returns the final task_id.
        """
        task_dict = (
            task_data.model_dump()
            if hasattr(task_data, "model_dump")
            else dict(task_data)
        )

        # Convert enum values to strings
        if isinstance(task_dict.get("status"), Enum):
            task_dict["status"] = task_dict["status"].value
        if isinstance(task_dict.get("priority"), Enum):
            task_dict["priority"] = task_dict["priority"].value

        # Validate for new tasks
        if not self._validate_task_update(None, task_dict):
            raise ValidationError("Invalid task data")

        # Auto-generate a unique task_id if none provided
        if "task_id" not in task_dict or not task_dict["task_id"]:
            timestamp = datetime.now(timezone.utc).timestamp()
            random_suffix = "".join(random.choices("0123456789abcdef", k=6))
            task_id = f"task-{timestamp}-{random_suffix}"
        else:
            task_id = task_dict["task_id"]

        # Set timestamps
        now_utc = datetime.now(timezone.utc)
        task_dict["task_id"] = task_id
        task_dict.setdefault("created_at", now_utc)
        task_dict["updated_at"] = now_utc

        # Insert into Mongo
        try:
            await self.tasks.insert_one(task_dict)
            return task_id
        except Exception as e:
            raise TaskStorageError(f"Failed to create task: {str(e)}")

    async def fetch_task(self, task_id: str) -> Optional[Task]:
        """
        Fetch a task by its ID. Returns the Task object or None if not found.
        """
        try:
            task_data = await self.tasks.find_one({"task_id": task_id})
            if not task_data:
                return None

            # Convert internal _id to string
            if "_id" in task_data:
                task_data["_id"] = str(task_data["_id"])

            # Convert strings back to Enums
            if isinstance(task_data.get("status"), str):
                task_data["status"] = TaskStatus(task_data["status"])
            if isinstance(task_data.get("priority"), str):
                task_data["priority"] = TaskPriority(task_data["priority"])

            return Task(**task_data)

        except Exception as e:
            raise TaskStorageError(f"Failed to fetch task: {str(e)}")

    async def update_task(self, task_id: str, updates: Task) -> bool:
        """
        Update an existing task with validation and log status transitions to execution_history.
        Returns True if an update occurred, False otherwise.
        """
        # Convert the Task model into a dict
        update_dict = (
            updates.model_dump()
            if hasattr(updates, "model_dump")
            else dict(updates)
        )
        update_dict = self._convert_enums_to_str(update_dict)

        # Validate data for an update
        if not self._validate_task_update(task_id, update_dict):
            raise ValidationError("Invalid update data")

        # Set new updated_at
        now_utc = datetime.now(timezone.utc)
        update_dict["updated_at"] = now_utc

        try:
            # Fetch current task to compare old and new status
            current_task = await self.fetch_task(task_id)
            old_status = current_task.status if current_task else None

            # Perform the update in Mongo
            result = await self.tasks.update_one(
                {"task_id": task_id}, {"$set": update_dict}
            )

            # If Mongo updated something
            if result.modified_count > 0:
                new_status = update_dict.get("status", None)
                if (
                    new_status
                    and old_status
                    and new_status != old_status.value
                ):
                    # Log status transition
                    entry_id = f"entry-{now_utc.timestamp()}"
                    log_entry = {
                        "entry_id": entry_id,
                        "task_id": task_id,
                        "timestamp": now_utc,
                        "status": new_status,
                        "changed_by": update_dict.get(
                            "assigned_agent", "Agent1"
                        ),
                        "comment": f"Transition from {old_status.value} to {new_status}",
                    }
                    await self.execution_history.insert_one(log_entry)

                # Invalidate the cache if we have one
                if self.cache:
                    self.cache.invalidate_task_cache(task_id)

                return True

            return False

        except Exception as e:
            raise TaskStorageError(f"Failed to update task: {str(e)}")

    async def delete_task(self, task_id: str) -> bool:
        """
        Delete a task by ID, along with its execution history.
        Returns True if deletion succeeded, False otherwise.
        """
        try:
            result = await self.tasks.delete_one({"task_id": task_id})
            await self.execution_history.delete_many({"task_id": task_id})
            return result.deleted_count > 0
        except Exception as e:
            raise TaskStorageError(f"Failed to delete task: {str(e)}")

    # ----------------------------------------------------------------------
    # Execution History
    # ----------------------------------------------------------------------

    async def log_execution_history(
        self, entry: ExecutionHistoryEntry
    ) -> None:
        """
        Log an entry in execution_history for a given task.
        Useful if you want to log arbitrary transitions outside update_task.
        """
        entry_id = f"entry-{datetime.now().timestamp()}"
        log_entry = {
            "entry_id": entry_id,
            "task_id": entry.task_id,
            "timestamp": entry.timestamp,
            "status": entry.status,
            "changed_by": entry.changed_by,
            "comment": entry.comment,
        }
        try:
            await self.execution_history.insert_one(log_entry)
        except Exception as e:
            raise TaskStorageError(
                f"Failed to log execution history: {str(e)}"
            )

    async def fetch_execution_history(
        self, task_id: str
    ) -> List[ExecutionHistoryEntry]:
        """
        Retrieve a task's execution history entries, sorted descending by timestamp.
        """
        try:
            cursor = self.execution_history.find({"task_id": task_id}).sort(
                "timestamp", -1
            )
            return [ExecutionHistoryEntry(**entry) async for entry in cursor]
        except Exception as e:
            raise TaskStorageError(
                f"Failed to fetch execution history: {str(e)}"
            )

    # ----------------------------------------------------------------------
    # Dependency Management
    # ----------------------------------------------------------------------

    async def add_task_dependency(
        self, task_id: str, dependency_id: str
    ) -> bool:
        """Add a dependency (dependency_id) to a task (task_id)."""
        try:
            result = await self.tasks.update_one(
                {"task_id": task_id},
                {"$addToSet": {"dependencies": dependency_id}},
            )
            return result.modified_count > 0
        except Exception as e:
            raise TaskStorageError(f"Failed to add task dependency: {str(e)}")

    async def remove_task_dependency(
        self, task_id: str, dependency_id: str
    ) -> bool:
        """Remove a dependency (dependency_id) from a task (task_id)."""
        try:
            result = await self.tasks.update_one(
                {"task_id": task_id},
                {"$pull": {"dependencies": dependency_id}},
            )
            return result.modified_count > 0
        except Exception as e:
            raise TaskStorageError(
                f"Failed to remove task dependency: {str(e)}"
            )

    async def check_dependency_cycle(
        self, task_id: str, dependency_id: str
    ) -> bool:
        """
        Check for circular dependencies (returns True if a cycle is found).
        This does a depth-first search of dependencies, tracking visited tasks.
        """
        visited = set()

        async def check_deps(current_id: str) -> bool:
            if current_id in visited:
                return True
            visited.add(current_id)
            task = await self.fetch_task(current_id)
            if task and task.dependencies:
                for dep_id in task.dependencies:
                    if await check_deps(dep_id):
                        return True
            return False

        return await check_deps(dependency_id)

    # ----------------------------------------------------------------------
    # Searching
    # ----------------------------------------------------------------------

    async def search_tasks(
        self, filters: Optional[Task] = None, limit: Optional[int] = 10
    ) -> List[Task]:
        """
        Search for tasks matching the specified filters.
        When searching by agent, it should return all tasks assigned to that agent regardless of other default values.
        When searching by priority, it should return tasks matching that priority exactly.

        :param filters: Task object with search criteria
        :param limit: Maximum number of tasks to return
        :return: List of matching Task objects
        """
        query = {}
        if filters:
            # Build query with only non-empty/non-None values
            if filters.status and filters.status != TaskStatus.PENDING:
                query["status"] = filters.status.value
            if filters.priority and filters.priority != TaskPriority.MEDIUM:
                query["priority"] = filters.priority.value
            if filters.assigned_agent and filters.assigned_agent.strip():
                query["assigned_agent"] = filters.assigned_agent
            if filters.created_by and filters.created_by.strip():
                query["created_by"] = filters.created_by

            # Debug logging
            Logger.debug(f"Search query: {query}")

        try:
            cursor = self.tasks.find(query).limit(limit)
            results = []
            async for task in cursor:
                # Convert ObjectId to string
                if "_id" in task:
                    task["_id"] = str(task["_id"])
                # Convert strings back to Enums
                if "status" in task:
                    task["status"] = TaskStatus(task["status"])
                if "priority" in task:
                    task["priority"] = TaskPriority(task["priority"])
                results.append(Task(**task))

            # Debug logging
            Logger.debug(f"Found {len(results)} tasks matching query {query}")
            return results
        except Exception as e:
            Logger.error(f"Search failed with query {query}: {str(e)}")
            raise TaskStorageError(f"Failed to search tasks: {str(e)}")

    # ----------------------------------------------------------------------
    # Cleanup & Helpers
    # ----------------------------------------------------------------------

    async def cleanup(self) -> None:
        """
        Clean up all tasks and execution history (useful for testing).
        Closes the MongoDB client after clearing data.
        """
        try:
            await self.tasks.delete_many({})
            await self.execution_history.delete_many({})
            if self.client:
                self.client.close()
        except Exception as e:
            Logger.error(f"Error during MongoDB cleanup: {str(e)}")

    def _convert_enums_to_str(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Enum values in a dictionary to strings, for MongoDB storage."""
        result = {}
        for key, value in data.items():
            if isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result

    def _validate_task_update(
        self, task_id: Optional[str], updates: Dict[str, Any]
    ) -> bool:
        """
        Validate task update data. For new tasks (task_id=None), require
        certain fields (title, description, etc.). For updates, require
        at least one valid field out of the allowed set.
        """
        try:
            # For new tasks (task_id is None)
            if task_id is None:
                required_fields = {
                    "title",
                    "description",
                    "status",
                    "priority",
                    "assigned_agent",
                }
                if not all(field in updates for field in required_fields):
                    Logger.error(
                        f"Missing required fields. Required: {required_fields}"
                    )
                    return False

                # Field length validations
                if len(updates.get("title", "")) > 200:
                    Logger.error("Title too long (max 200 chars)")
                    return False
                if len(updates.get("description", "")) > 5000:
                    Logger.error("Description too long (max 5000 chars)")
                    return False

                # Validate due date for new tasks
                if due_date := updates.get("due_date"):
                    if isinstance(due_date, str):
                        try:
                            due_date = datetime.fromisoformat(due_date)
                        except ValueError:
                            Logger.error("Invalid due_date format")
                            return False
                    if due_date < datetime.now(timezone.utc):
                        Logger.error("Due date cannot be in the past")
                        return False

            # Common validations for both new and updates
            valid_fields = {
                "title",
                "description",
                "status",
                "priority",
                "assigned_agent",
                "due_date",
                "dependencies",
                "progress",
                "metadata",
                "tags",
            }
            update_fields = set(updates.keys())
            invalid_fields = update_fields - valid_fields
            if invalid_fields:
                Logger.error(f"Invalid fields detected: {invalid_fields}")
                return False

            # Type validations
            if "progress" in updates and not isinstance(
                updates["progress"], (int, float)
            ):
                Logger.error("Progress must be a number")
                return False
            if "progress" in updates and not (0 <= updates["progress"] <= 100):
                Logger.error("Progress must be between 0 and 100")
                return False

            # Enum validations
            if (
                "status" in updates
                and updates["status"] not in TaskStatus.__members__.values()
            ):
                Logger.error(f"Invalid status value: {updates['status']}")
                return False
            if (
                "priority" in updates
                and updates["priority"]
                not in TaskPriority.__members__.values()
            ):
                Logger.error(f"Invalid priority value: {updates['priority']}")
                return False

            # Validate due date for updates
            if due_date := updates.get("due_date"):
                if isinstance(due_date, str):
                    try:
                        due_date = datetime.fromisoformat(due_date)
                    except ValueError:
                        Logger.error("Invalid due_date format")
                        return False
                if due_date < datetime.now(timezone.utc):
                    Logger.error("Due date cannot be in the past")
                    return False

            return True

        except Exception as e:
            Logger.error(f"Validation error: {str(e)}")
            return False

    async def get_prioritized_tasks(
        self, filters: Optional[Dict[str, Any]] = None, limit: int = 10
    ) -> List[Task]:
        """
        Get tasks ordered by priority score.

        Args:
            filters: Optional dictionary of filter criteria
            limit: Maximum number of tasks to return

        Returns:
            List of Task objects ordered by priority score
        """
        try:
            query = filters or {}

            # Convert any enum values in filters to strings
            query = self._convert_enums_to_str(query)

            # Use find() to get a cursor and convert to list
            cursor = self.tasks.find(query)

            # Apply sort and limit
            cursor = cursor.sort(
                [("priority_score", -1), ("due_date", 1)]
            ).limit(limit)

            # Handle both real MongoDB cursor and mock list
            if hasattr(cursor, "to_list"):
                tasks_list = await cursor.to_list(length=limit)
            else:
                # For mocks that return a list directly
                tasks_list = cursor

            results = []
            for task in tasks_list:
                # Convert ObjectId to string
                if "_id" in task:
                    task["_id"] = str(task["_id"])

                # Convert string values back to Enums
                if "status" in task:
                    task["status"] = TaskStatus(task["status"])
                if "priority" in task:
                    task["priority"] = TaskPriority(task["priority"])

                results.append(Task(**task))

            Logger.debug(
                f"Found {len(results)} prioritized tasks matching query {query}"
            )
            return results

        except Exception as e:
            Logger.error(f"Failed to fetch prioritized tasks: {str(e)}")
            raise TaskStorageError(
                f"Failed to fetch prioritized tasks: {str(e)}"
            )
