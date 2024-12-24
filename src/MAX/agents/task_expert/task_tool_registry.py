from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from pydantic import Field
from MAX.tools.tool_registry import ToolRegistry, BaseTool, ToolCategory, BaseToolInput, BaseToolOutput
from MAX.storage.utils.types import Task, TaskStatus, TaskPriority
import logging

# Base TaskTool Class
class TaskTool(BaseTool):
    """Base class for all task tools."""
    
    class Config:
        arbitrary_types_allowed = True

    storage: Any = Field(default=None, exclude=True)
    logger: Any = Field(default_factory=lambda: logging.getLogger(__name__), exclude=True)

    async def _validate_task_exists(self, task_id: str) -> Optional[Task]:
        """Helper method to validate task existence."""
        existing_task = await self.storage.fetch_task(task_id)
        if not existing_task:
            return None
        return existing_task

# Concrete Task Tools
class CreateTaskTool(TaskTool):
    async def _execute(self, **kwargs) -> Dict[str, Any]:
        current_time = datetime.now(timezone.utc)
        task = Task(
            task_id=str(uuid4()),
            title=kwargs["title"],
            description=kwargs.get("description", ""),
            status=TaskStatus.PENDING,
            priority=kwargs.get("priority", TaskPriority.MEDIUM),
            created_at=current_time,
            last_updated=current_time,
            updated_at=current_time,
            assigned_agent=kwargs.get("assigned_agent", "system"),
            due_date=kwargs.get("due_date", current_time + timedelta(days=7)),
            created_by=kwargs.get("created_by", "system"),
            dependencies=kwargs.get("dependencies", []),
            metadata=kwargs.get("metadata", {})
        )
        
        task_id = await self.storage.create_task(task)
        saved_task = await self.storage.fetch_task(task_id)
        
        return {"task": saved_task}

class UpdateTaskTool(TaskTool):
    async def _execute(self, **kwargs) -> Dict[str, Any]:
        task_id = kwargs["task_id"]
        existing_task = await self._validate_task_exists(task_id)
        
        if not existing_task:
            raise ValueError(f"Task {task_id} not found")
        
        updates = {
            "updated_at": datetime.now(timezone.utc),
            "last_updated": datetime.now(timezone.utc)
        }
        
        updateable_fields = [
            "status", "description", "priority",
            "due_date", "assigned_agent", "progress"
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

    async def execute(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name with given parameters."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")
        
        tool = self.tools[tool_name]
        try:
            result = await tool._execute(**params)
            return result
        except Exception as e:
            raise ValueError(f"Tool execution failed: {str(e)}")

    def _register_task_tools(self):
        """Register all available task tools."""
        self.register(CreateTaskTool(
            name="create_task",
            description="Create a new task",
            category=ToolCategory.SYSTEM,
            input_schema=CreateTaskInput,
            output_schema=TaskOutput,
            storage=self.storage
        ))

        self.register(UpdateTaskTool(
            name="update_task",
            description="Update an existing task",
            category=ToolCategory.SYSTEM,
            input_schema=UpdateTaskInput,
            output_schema=TaskOutput,
            storage=self.storage
        ))

        self.register(DeleteTaskTool(
            name="delete_task",
            description="Delete a task",
            category=ToolCategory.SYSTEM,
            input_schema=DeleteTaskInput,
            output_schema=TaskOutput,
            storage=self.storage
        ))

