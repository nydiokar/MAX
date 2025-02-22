import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from uuid import uuid4

from MAX.storage.utils.types import TaskStatus, TaskPriority, Task
from MAX.storage.utils.protocols import TaskStorage, NotificationService
from MAX.agents.task_expert.errors import (
    TaskExpertError,
    ValidationError,
    ErrorHandler,
    StorageError,
    LLMError,
    TaskInputModel,
)
from MAX.utils import Logger
from MAX.agents.agent import Agent
from MAX.agents.task_expert.options import TaskExpertOptions
from MAX.agents.task_expert.task_tool_registry import TaskToolRegistry
from MAX.llms.base import AsyncLLMBase
from MAX.config.llms.ollama import OllamaConfig
from MAX.config.llms.base import BaseLlmConfig
from MAX.llms.ollama import OllamaLLM


class TaskExpertAgent(Agent):
    """
    Task Expert Agent for interpreting user requests and managing tasks.
    Simplified version without KPU and retriever dependencies.
    """

    def __init__(self, options: TaskExpertOptions):
        super().__init__(options)

        # Core components
        self.storage: TaskStorage = options.storage_client
        self.notifications: NotificationService = options.notification_service
        self.error_handler = ErrorHandler(
            max_retries=options.max_retries,
            retry_delay=options.retry_delay
        )

        # LLM setup
        self.llm = self._create_llm_provider(options.get_llm_config)

        # System prompt
        self.system_prompt = (
            "You are a Task Expert agent. Analyze the user request and create appropriate tasks. "
            "Return a JSON object with: 'tasks' (list of tasks to create) and 'priority' (HIGH/MEDIUM/LOW). "
            "Each task should have: 'title', 'description', 'type' (agent_task/human_task), "
            "and 'estimated_duration' (in hours). No explanations, just JSON."
        )

        # Task management
        self.tool_registry = TaskToolRegistry(storage=self.storage)

    def _create_llm_provider(self, llm_config: BaseLlmConfig) -> AsyncLLMBase:
        """Create LLM instance with simplified config."""
        if not isinstance(llm_config, OllamaConfig):
            Logger.warn("Converting to OllamaConfig with defaults")
            llm_config = OllamaConfig(
                model=llm_config.model or "llama3.1:8b-instruct-q8_0",
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                top_p=llm_config.top_p,
                resources=llm_config.resources,
            )
        return OllamaLLM(llm_config)

    async def process_request(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a user request and create appropriate tasks."""
        try:
            # Validate input
            if not input_text.strip():
                raise ValidationError("Input text cannot be empty")

            # Analyze request with LLM
            analysis = await self.error_handler.handle_llm_operation(
                self._analyze_request,
                input_text,
                context or {}
            )

            # Create tasks
            tasks_created = await self._create_and_validate_tasks(
                analysis.get("tasks", [])
            )

            # Set priorities
            priority = TaskPriority[analysis.get("priority", "MEDIUM")]
            for task in tasks_created:
                await self.storage.update_task(
                    task.id,
                    {"priority": priority}
                )

            return {
                "tasks_created": tasks_created,
                "priority": priority.value
            }

        except (ValidationError, LLMError, StorageError) as e:
            Logger.error(f"Error processing request: {str(e)}")
            raise TaskExpertError(f"Failed to process request: {str(e)}")

    async def _analyze_request(
        self,
        input_text: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze request with LLM to determine tasks."""
        prompt = f"""
        {self.system_prompt}

        User Request: {input_text}
        Context: {json.dumps(context) if context else 'No additional context'}
        """

        try:
            response_text = await self.llm.generate(prompt)
            return json.loads(response_text)
        except (json.JSONDecodeError, ValueError) as e:
            Logger.error(f"Failed to parse LLM response: {str(e)}")
            return {"tasks": [], "priority": "MEDIUM"}

    async def _create_and_validate_tasks(
        self,
        tasks_list: List[Dict[str, Any]]
    ) -> List[Task]:
        """Create and validate tasks from LLM analysis."""
        tasks_created = []

        for task_data in tasks_list:
            try:
                # Basic validation
                if not task_data.get("title") or not task_data.get("description"):
                    continue

                # Create task
                task = Task(
                    id=str(uuid4()),
                    title=task_data["title"],
                    description=task_data["description"],
                    type=task_data["type"],
                    status=TaskStatus.PENDING,
                    created_at=datetime.now(timezone.utc),
                    estimated_duration=float(task_data.get("estimated_duration", 1.0))
                )

                # Save task
                saved_task = await self.error_handler.handle_storage_operation(
                    self.storage.save_task,
                    task
                )
                tasks_created.append(saved_task)

                # Send notification
                await self.notifications.send(
                    "system",
                    {
                        "type": "task_created",
                        "task_id": task.id,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )

            except Exception as e:
                Logger.error(f"Failed to create task: {str(e)}")
                continue

        return tasks_created

    async def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        progress: Optional[float] = None
    ) -> Task:
        """Update task status and progress."""
        try:
            task = await self.storage.fetch_task(task_id)
            if not task:
                raise ValueError(f"Task {task_id} not found")

            updates = {
                "status": status,
                "last_updated": datetime.now(timezone.utc)
            }
            if progress is not None:
                updates["progress"] = progress

            updated_task = await self.storage.update_task(task_id, updates)
            if not updated_task:
                raise ValueError(f"Failed to update task {task_id}")

            return updated_task

        except Exception as e:
            Logger.error(f"Failed to update task status: {str(e)}")
            raise TaskExpertError(f"Failed to update task: {str(e)}")

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get current task status and details."""
        try:
            task = await self.storage.fetch_task(task_id)
            if not task:
                raise ValueError(f"Task {task_id} not found")

            return {
                "id": task.id,
                "title": task.title,
                "status": task.status.value,
                "progress": getattr(task, "progress", 0),
                "last_updated": task.last_updated.isoformat(),
                "estimated_duration": getattr(task, "estimated_duration", None)
            }

        except Exception as e:
            Logger.error(f"Failed to get task status: {str(e)}")
            raise TaskExpertError(f"Failed to get task status: {str(e)}")
