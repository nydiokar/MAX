from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta
from pydantic import Field
from MAX.tools.tool_registry import (
    BaseTool,
    ToolCategory,
    BaseToolInput,
    BaseToolOutput,
)
from MAX.storage.utils.types import Task, TaskStatus, TaskPriority
from MAX.agents.task_expert.errors import ValidationError, TaskExpertError
from MAX.utils.logger import Logger
import time


# Base TaskTool Class
class TaskTool(BaseTool):
    """Base class for all task tools."""

    class Config:
        arbitrary_types_allowed = True

    storage: Any = Field(default=None, exclude=True)
    logger: Any = Field(
        default_factory=lambda: Logger.get_logger(), exclude=True
    )

    async def _validate_task_exists(self, task_id: str) -> Optional[Task]:
        """Helper method to validate task existence."""
        existing_task = await self.storage.fetch_task(task_id)
        if not existing_task:
            return None
        return existing_task


# Concrete Task Tools
class CreateTaskTool(TaskTool):
    async def _execute(self, **kwargs) -> Dict[str, Any]:
        try:
            # Validate input data
            current_time = datetime.now(timezone.utc)
            validated_data = {
                "title": kwargs["title"],
                "description": kwargs.get("description", ""),
                "status": TaskStatus.PENDING,
                "priority": kwargs.get("priority", TaskPriority.MEDIUM),
                "created_at": current_time,
                "updated_at": current_time,
                "last_updated": current_time,
                "assigned_agent": kwargs.get("assigned_agent", "system"),
                "due_date": kwargs.get(
                    "due_date", current_time + timedelta(days=7)
                ),
                "created_by": kwargs.get("created_by", "system"),
                "dependencies": kwargs.get("dependencies", []),
                "metadata": kwargs.get("metadata", {}),
                "tags": kwargs.get("tags", []),
                "progress": 0,
            }

            # Additional validations
            if validated_data["due_date"] < current_time:
                raise ValidationError("Due date cannot be in the past")

            # Validate dependencies exist
            if validated_data["dependencies"]:
                for dep_id in validated_data["dependencies"]:
                    if not await self._validate_task_exists(dep_id):
                        raise ValidationError(f"Dependency {dep_id} not found")

            # Create task
            task = Task(**validated_data)
            task_id = await self.storage.create_task(task)
            if not task_id:
                raise ValueError("Failed to create task")

            # Fetch and return created task
            saved_task = await self.storage.fetch_task(task_id)
            return {"task": saved_task}

        except ValidationError as e:
            self.logger.error(f"Validation error in CreateTaskTool: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error in CreateTaskTool: {str(e)}")
            raise


class UpdateTaskTool(TaskTool):
    async def _execute(self, **kwargs) -> Dict[str, Any]:
        task_id = kwargs["task_id"]
        existing_task = await self._validate_task_exists(task_id)

        if not existing_task:
            raise ValueError(f"Task {task_id} not found")

        updates = {
            "updated_at": datetime.now(timezone.utc),
            "last_updated": datetime.now(timezone.utc),
        }

        updateable_fields = [
            "status",
            "description",
            "priority",
            "due_date",
            "assigned_agent",
            "progress",
        ]

        for field in updateable_fields:
            if field in kwargs:
                updates[field] = kwargs[field]

        success = await self.storage.update_task(task_id, updates)
        if not success:
            raise ValueError("Failed to update task")

        updated_task = await self.storage.fetch_task(task_id)
        return {"task": updated_task}


class DeleteTaskTool(TaskTool):
    async def _execute(self, **kwargs) -> Dict[str, Any]:
        task_id = kwargs["task_id"]
        existing_task = await self._validate_task_exists(task_id)

        if not existing_task:
            raise ValueError(f"Task {task_id} not found")

        dependent_tasks = await self.storage.search_tasks(
            filters={"dependencies": task_id}
        )

        if dependent_tasks:
            raise ValueError(f"Task {task_id} has dependencies")

        success = await self.storage.delete_task(task_id)
        return {"success": success}


# Input Schemas
class CreateTaskInput(BaseToolInput):
    title: str
    description: str
    priority: TaskPriority
    due_date: Optional[datetime] = None
    assigned_agent: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)
    estimated_hours: Optional[float] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_by: Optional[str] = None


class UpdateTaskInput(BaseToolInput):
    task_id: str
    status: Optional[TaskStatus] = None
    progress: Optional[float] = None
    description: Optional[str] = None
    priority: Optional[TaskPriority] = None
    due_date: Optional[datetime] = None
    assigned_agent: Optional[str] = None


class DeleteTaskInput(BaseToolInput):
    task_id: str
    reason: Optional[str] = None


class QueryTasksInput(BaseToolInput):
    status: Optional[TaskStatus] = None
    assigned_agent: Optional[str] = None
    priority: Optional[TaskPriority] = None
    due_before: Optional[datetime] = None
    tags: Optional[List[str]] = None
    limit: int = 10


# Output Schemas
class TaskOutput(BaseToolOutput):
    task: Task


class TaskListOutput(BaseToolOutput):
    tasks: List[Task]
    total_count: int
    page: int = 1
    has_more: bool = False


# Tool Registry
class TaskToolRegistry:
    """Registry for task-related tools."""

    def __init__(self, storage=None):
        self.tools = {}
        self.storage = storage
        if storage:
            self._register_task_tools()

    def register(self, tool: TaskTool):
        """Register a tool in the registry."""
        self.tools[tool.name] = tool

    async def execute(
        self, tool_name: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a tool by name with given parameters.

        Args:
            tool_name: Name of the tool to execute
            params: Parameters to pass to the tool

        Returns:
            Dict containing the tool's execution results

        Raises:
            ValueError: If tool_name is not found
            TaskExpertError: If tool execution fails
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")

        start_time = time.time()
        try:
            result = await self.tools[tool_name]._execute(**params)
            execution_time = time.time() - start_time
            Logger.info(
                f"Tool {tool_name} executed in {execution_time:.2f}s with params: {params}"
            )
            return result

        except ValidationError as e:
            execution_time = time.time() - start_time
            Logger.error(
                f"Tool {tool_name} validation failed after {execution_time:.2f}s: {str(e)}"
            )
            raise

        except Exception as e:
            execution_time = time.time() - start_time
            Logger.error(
                f"Tool {tool_name} failed after {execution_time:.2f}s: {str(e)}"
            )
            raise TaskExpertError(f"Tool execution failed: {str(e)}") from e

    def _register_task_tools(self):
        """Register all available task tools."""
        self.register(
            CreateTaskTool(
                name="create_task",
                description="Create a new task",
                category=ToolCategory.SYSTEM,
                input_schema=CreateTaskInput,
                output_schema=TaskOutput,
                storage=self.storage,
            )
        )

        self.register(
            UpdateTaskTool(
                name="update_task",
                description="Update an existing task",
                category=ToolCategory.SYSTEM,
                input_schema=UpdateTaskInput,
                output_schema=TaskOutput,
                storage=self.storage,
            )
        )

        self.register(
            DeleteTaskTool(
                name="delete_task",
                description="Delete a task",
                category=ToolCategory.SYSTEM,
                input_schema=DeleteTaskInput,
                output_schema=TaskOutput,
                storage=self.storage,
            )
        )
