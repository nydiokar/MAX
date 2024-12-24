from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from MAX.tools.tool_registry import ToolRegistry, BaseTool, ToolCategory, BaseToolInput, BaseToolOutput
from MAX.storage.utils.types import Task, TaskStatus, TaskPriority

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

class AddDependencyInput(BaseToolInput):
   task_id: str
   dependency_id: str

class GetTaskDependenciesInput(BaseToolInput):
   task_id: str

class AnalyzeTaskProgressInput(BaseToolInput):
   task_id: str
   include_dependencies: bool = False

class EstimateCompletionInput(BaseToolInput):
   task_id: str

# Output Schemas
class TaskOutput(BaseToolOutput):
   task: Task

class TaskListOutput(BaseToolOutput):
   tasks: List[Task]
   total_count: int
   page: int = 1
   has_more: bool = False

class AnalysisOutput(BaseToolOutput):
   analysis: Dict[str, Any]
   recommendations: List[str]

class EstimateOutput(BaseToolOutput): 
   estimated_completion: datetime
   confidence_score: float

class TaskToolRegistry(ToolRegistry):
   """
   Registry for task-specific tools.
   Tools handle CRUD operations on tasks and related functionalities.
   """
   def __init__(self):
       super().__init__()
       self._register_task_tools()
   
   async def execute(self, tool_name: str, input_data: Dict[str, Any]) -> Any:
       """Execute a tool by name with given input data"""
       tool = self.get_tool(tool_name)
       if not tool:
           raise ValueError(f"Tool {tool_name} not found")
       return await tool.execute(input_data)
   
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
               "input_schema": GetTaskDependenciesInput,
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
               "input_schema": AnalyzeTaskProgressInput,
               "output_schema": AnalysisOutput
           },
           {
               "name": "estimate_completion",
               "description": "Estimate task completion date",
               "category": ToolCategory.ANALYSIS,
               "input_schema": EstimateCompletionInput,
               "output_schema": EstimateOutput
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