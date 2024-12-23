from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorClient
import random
from enum import Enum
from MAX.utils import Logger
from MAX.storage.abstract_storage.task_storage import TaskStorage
from MAX.storage.utils.types import Task, ExecutionHistoryEntry, TaskStatus, TaskPriority

class TaskStorageError(Exception):
    """Base exception for task storage errors"""
    pass

class DependencyCycleError(TaskStorageError):
    """Raised when a dependency cycle is detected"""
    pass

class ValidationError(TaskStorageError):
    """Raised when task data validation fails"""
    pass


class MongoDBTaskStorage(TaskStorage):
    """MongoDB implementation for managing tasks and execution history."""

    def __init__(self, client: AsyncIOMotorClient, db_name: str):
        """Initialize with an existing client instead of creating one"""
        self.client = client
        self.db = self.client[db_name]
        self.tasks = self.db["tasks"]
        self.execution_history = self.db["execution_history"]

    async def _setup_indices(self):
        """Setup indices for task and execution history collections."""
        try:
            await self.tasks.create_index([("task_id", 1)], unique=True)
            await self.tasks.create_index([("status", 1), ("priority", 1), ("due_date", 1)])
            await self.execution_history.create_index([("task_id", 1), ("timestamp", -1)])
            await self.execution_history.create_index([("entry_id", 1)], unique=True)

            # Additional indices
            await self.tasks.create_index([("status", 1), ("assigned_agent", 1)])
            await self.tasks.create_index([("priority", 1)])
            await self.tasks.create_index([("due_date", 1)])
            await self.tasks.create_index([("dependencies", 1)])
        except Exception as e:
            Logger.error(f"Failed to setup MongoDB indices: {str(e)}")

    def validate_task_update(self, task_id: Optional[str], updates: Dict[str, Any]) -> bool:
        """Validate task update data"""
        if task_id is None:
            # For new tasks, require these fields:
            required_fields = {"title", "description", "status", "priority", "assigned_agent"}
            update_fields = set(updates.keys())
            Logger.debug(f"Required fields: {required_fields}")
            Logger.debug(f"Provided fields: {update_fields}")
            return all(field in update_fields for field in required_fields)
        else:
            # For updates, allow any valid field
            valid_fields = {
                "title", "description", "status", "priority", "assigned_agent",
                "due_date", "dependencies", "progress", "metadata", "tags"
            }
            update_fields = set(updates.keys())
            # Must contain at least one valid field
            return bool(update_fields & valid_fields)

    async def check_dependency_cycle(self, task_id: str, dependency_id: str) -> bool:
        """Check for circular dependencies (returns True if cycle found)"""
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

    def _convert_enums_to_str(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Enum values to strings for MongoDB storage"""
        result = {}
        for key, value in data.items():
            if isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result

    async def create_task(self, task_data: Task) -> str:
        """Create a new task with validation."""
        # Convert Task (Pydantic) to dict
        task_dict = task_data.model_dump() if hasattr(task_data, 'model_dump') else dict(task_data)

        # Convert Enum values to strings
        if isinstance(task_dict.get("status"), Enum):
            task_dict["status"] = task_dict["status"].value
        if isinstance(task_dict.get("priority"), Enum):
            task_dict["priority"] = task_dict["priority"].value

        # Validate data
        if not self.validate_task_update(None, task_dict):
            raise ValidationError("Invalid task data")
        
        # Generate a unique task ID if not provided
        if "task_id" not in task_dict or not task_dict["task_id"]:
            timestamp = datetime.now(timezone.utc).timestamp()
            random_suffix = ''.join(random.choices('0123456789abcdef', k=6))
            task_id = f"task-{timestamp}-{random_suffix}"
        else:
            task_id = task_dict["task_id"]
        
        task_dict["task_id"] = task_id
        now_utc = datetime.now(timezone.utc)
        task_dict.setdefault("created_at", now_utc)
        task_dict["updated_at"] = now_utc
        
        try:
            await self.tasks.insert_one(task_dict)
            return task_id
        except Exception as e:
            raise TaskStorageError(f"Failed to create task: {str(e)}")

    async def update_task(self, task_id: str, updates: Task) -> bool:
        """Update an existing task with validation."""
        update_dict = updates.model_dump() if hasattr(updates, 'model_dump') else dict(updates)
        update_dict = self._convert_enums_to_str(update_dict)
        
        if not self.validate_task_update(task_id, update_dict):
            raise ValidationError("Invalid update data")
        
        update_dict["updated_at"] = datetime.now(timezone.utc)
        try:
            result = await self.tasks.update_one(
                {"task_id": task_id},
                {"$set": update_dict}
            )
            return result.modified_count > 0
        except Exception as e:
            raise TaskStorageError(f"Failed to update task: {str(e)}")

    async def fetch_task(self, task_id: str) -> Optional[Task]:
        """Fetch a task by its ID."""
        try:
            task_data = await self.tasks.find_one({"task_id": task_id})
            if task_data:
                if "_id" in task_data:
                    task_data["_id"] = str(task_data["_id"])
                # Convert string enums to proper types
                if isinstance(task_data["status"], str):
                    task_data["status"] = TaskStatus(task_data["status"])
                if isinstance(task_data["priority"], str):
                    task_data["priority"] = TaskPriority(task_data["priority"])
                return Task(**task_data)
            return None
        except Exception as e:
            raise TaskStorageError(f"Failed to fetch task: {str(e)}")

    async def delete_task(self, task_id: str) -> bool:
        """Delete a task and its execution history."""
        result = await self.tasks.delete_one({"task_id": task_id})
        await self.execution_history.delete_many({"task_id": task_id})
        return result.deleted_count > 0

    async def log_execution_history(self, entry: ExecutionHistoryEntry) -> None:
        """Log execution history for a task."""
        entry_id = f"entry-{datetime.now().timestamp()}"
        log_entry = {
            "entry_id": entry_id,
            "task_id": entry.task_id,
            "timestamp": entry.timestamp,
            "status": entry.status,
            "changed_by": entry.changed_by,
            "comment": entry.comment
        }
        await self.execution_history.insert_one(log_entry)

    async def fetch_execution_history(self, task_id: str) -> List[ExecutionHistoryEntry]:
        """Fetch execution history for a task."""
        cursor = self.execution_history.find({"task_id": task_id}).sort("timestamp", -1)
        return [ExecutionHistoryEntry(**entry) async for entry in cursor]

    async def search_tasks(
        self,
        filters: Optional[Task] = None,
        limit: Optional[int] = 10
    ) -> List[Task]:
        """Search for tasks with filters."""
        query = {}
        if filters:
            if filters.status:
                query["status"] = filters.status.value
            if filters.priority:
                query["priority"] = filters.priority.value
            if filters.assigned_agent:
                query["assigned_agent"] = filters.assigned_agent

        cursor = self.tasks.find(query).limit(limit)
        results = []
        async for task in cursor:
            if "_id" in task:
                task["_id"] = str(task["_id"])
            # Convert string values back to Enums
            if "status" in task:
                task["status"] = TaskStatus(task["status"])
            if "priority" in task:
                task["priority"] = TaskPriority(task["priority"])
            results.append(Task(**task))
        return results

    async def add_task_dependency(self, task_id: str, dependency_id: str) -> bool:
        """Add a dependency to a task."""
        result = await self.tasks.update_one(
            {"task_id": task_id},
            {"$addToSet": {"dependencies": dependency_id}}
        )
        return result.modified_count > 0

    async def remove_task_dependency(self, task_id: str, dependency_id: str) -> bool:
        """Remove a dependency from a task."""
        result = await self.tasks.update_one(
            {"task_id": task_id},
            {"$pull": {"dependencies": dependency_id}}
        )
        return result.modified_count > 0

    async def check_health(self) -> bool:
        """Check MongoDB connection health."""
        try:
            await self.db.command("ping")
            return True
        except Exception as e:
            Logger.error(f"MongoDB health check failed: {str(e)}")
            return False

    async def initialize(self) -> bool:
        """Initialize (check health, set up indices)."""
        try:
            await self.check_health()
            await self._setup_indices()
            return True
        except Exception as e:
            Logger.error(f"Failed to initialize MongoDB: {str(e)}")
            return False

    async def cleanup(self) -> None:
        """Clean up all tasks and execution history (for test usage)."""
        try:
            await self.tasks.delete_many({})
            await self.execution_history.delete_many({})
            if self.client:
                self.client.close()
        except Exception as e:
            Logger.error(f"Error during MongoDB cleanup: {str(e)}")
