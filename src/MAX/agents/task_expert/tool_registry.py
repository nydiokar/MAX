from typing import Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field
from MAX.tools.tool_registry import ToolRegistry, BaseTool, ToolCategory, BaseToolInput, BaseToolOutput
from storage.interfaces import TaskModel, TaskStatus, TaskPriority

# Input Schemas
class CreateTaskInput(BaseToolInput):
    title: str
    description: str
    priority: TaskPriority
    due_date: datetime = None
    assigned_agent: str = None
    dependencies: List[str] = Field(default_factory=list)
    estimated_hours: float = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UpdateTaskInput(BaseToolInput):
    task_id: str
    status: TaskStatus = None
    progress: float = None
    description: str = None
    priority: TaskPriority = None
    due_date: datetime = None
    assigned_agent: str = None


class DeleteTaskInput(BaseToolInput):
    task_id: str
    reason: str = None


class QueryTasksInput(BaseToolInput):
    status: TaskStatus = None
    assigned_agent: str = None
    priority: TaskPriority = None
    due_before: datetime = None
    tags: List[str] = None
    limit: int = 10


class AddDependencyInput(BaseToolInput):
    task_id: str
    dependency_id: str


# Output Schemas
class TaskOutput(BaseToolOutput):
    task: TaskModel


class TaskListOutput(BaseToolOutput):
    tasks: List[TaskModel]
    total_count: int
    page: int = 1
    has_more: bool = False


class TaskToolRegistry(ToolRegistry):
    """
    Registry for task-specific tools.
    Tools handle CRUD operations on tasks and related functionalities.
    """
    def __init__(self):
        super().__init__()
        self._register_task_tools()
    
    def _register_task_tools(self):
        tools_definitions = [
            {
                "name": "create_task",
                "description": "Create a new task",
                "category": ToolCategory.SYSTEM,
                "input_schema": CreateTaskInput,
                "output_schema": TaskOutput
            },
            {
                "name": "update_task",
                "description": "Update an existing task",
                "category": ToolCategory.SYSTEM,
                "input_schema": UpdateTaskInput,
                "output_schema": TaskOutput
            },
            {
                "name": "delete_task",
                "description": "Delete a task",
                "category": ToolCategory.SYSTEM,
                "input_schema": DeleteTaskInput,
                "output_schema": BaseToolOutput
            },
            {
                "name": "query_tasks",
                "description": "Query tasks by various criteria",
                "category": ToolCategory.SYSTEM,
                "input_schema": QueryTasksInput,
                "output_schema": TaskListOutput
            },
            {
                "name": "get_task_dependencies",
                "description": "Get all dependencies for a task",
                "category": ToolCategory.SYSTEM,
                "input_schema": BaseToolInput.with_fields({"task_id": (str, ...)}),
                "output_schema": TaskListOutput
            },
            {
                "name": "add_dependency",
                "description": "Add a dependency to a task",
                "category": ToolCategory.SYSTEM,
                "input_schema": AddDependencyInput,
                "output_schema": TaskOutput
            },
            {
                "name": "remove_dependency",
                "description": "Remove a dependency from a task",
                "category": ToolCategory.SYSTEM,
                "input_schema": AddDependencyInput,
                "output_schema": TaskOutput
            },
            {
                "name": "analyze_task_progress",
                "description": "Analyze task progress",
                "category": ToolCategory.ANALYSIS,
                "input_schema": BaseToolInput.with_fields({"task_id": (str, ...), "include_dependencies": (bool, False)}),
                "output_schema": BaseToolOutput.with_fields({
                    "analysis": (Dict[str, Any], ...),
                    "recommendations": (List[str], ...)
                })
            },
            {
                "name": "estimate_completion",
                "description": "Estimate task completion date",
                "category": ToolCategory.ANALYSIS,
                "input_schema": BaseToolInput.with_fields({"task_id": (str, ...)}),
                "output_schema": BaseToolOutput.with_fields({
                    "estimated_completion": (datetime, ...),
                    "confidence_score": (float, ...)
                })
            }
        ]

        # Register each tool defined above
        for tool_def in tools_definitions:
            self.register(
                BaseTool(
                    name=tool_def["name"],
                    description=tool_def["description"],
                    category=tool_def["category"],
                    input_schema=tool_def["input_schema"],
                    output_schema=tool_def["output_schema"],
                    requires_auth=True
                )
            )
