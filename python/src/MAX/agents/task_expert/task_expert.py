# try to make this guy more simpler (remove KPU and retriever) 

import asyncio
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
    DependencyError,
    MonitoringError,
    TaskInputModel,
)
from MAX.utils import Logger
from MAX.retrievers import Retriever
from MAX.retrievers.kb_retriever import (
    KnowledgeBasesRetrieverOptions,
    KnowledgeBasesRetriever,
)
from MAX.agents.agent import Agent
from MAX.agents.task_expert.options import TaskExpertOptions
from MAX.agents.task_expert.task_tool_registry import TaskToolRegistry
from MAX.config.llms.llm_config import LLM_CONFIGS
from MAX.llms.base import AsyncLLMBase
from MAX.config.llms.ollama import OllamaConfig
from MAX.config.llms.base import BaseLlmConfig
from MAX.llms.ollama import OllamaLLM

# from MAX.config.llms.anthropic import AnthropicConfig  # Uncomment when needed


class TaskExpertAgent(Agent):
    """
    Task Expert Agent is responsible for interpreting user requests related to tasks,
    creating or updating tasks in the database, and optionally sending notifications.
    """

    def __init__(self, options: TaskExpertOptions):
        super().__init__(options)

        # Set references
        self.storage: TaskStorage = options.storage_client
        self.notifications: NotificationService = options.notification_service
        self.default_ttl = options.default_task_ttl
        self.retriever: Optional[Retriever] = options.retriever
        self.error_handler = ErrorHandler(
            max_retries=options.max_retries, retry_delay=options.retry_delay
        )

        # Initialize kpu_context that's used in priority calculations
        self.kpu_context: Dict[str, Any] = {}

        # 1) Grab the final LLM config from the user or the dictionary
        self.llm_config = options.get_llm_config

        # 2) Create the main LLM
        self.llm = self._create_llm_provider(self.llm_config)

        # 3) Optional fallback (example usage)
        self.fallback_llm = None
        if self.llm_config.fallback_model:
            # If the config has a fallback_model field, try to find it in the dictionary
            fallback_cfg = LLM_CONFIGS["local"].get(
                "fast",  # or however you want to pick a fallback
                LLM_CONFIGS["local"]["general"],
            )
            self.fallback_llm = self._create_llm_provider(fallback_cfg)

        # System prompt
        self.system_prompt = (
            "You are a Task Expert agent. You receive a user request and additional context. "
            "Your job is to determine what tasks need to be created or updated. "
            "Return a JSON object with keys: 'tasks' (a list of tasks to create/update) "
            "and 'monitoring_requirements' (object detailing any monitoring or notifications needed). "
            "For each task in 'tasks', include 'type' (e.g., 'agent_task' or 'human_task') and 'details' as a dict. "
            "No explanations, just JSON."
        )

        # Initialize a retriever if none is provided
        if self.retriever is None and self.storage is not None:
            retriever_options = KnowledgeBasesRetrieverOptions(
                storage_client=self.storage,
                collection_name="knowledge_base",
                max_results=5,
                similarity_threshold=0.7,
            )
            self.retriever = KnowledgeBasesRetriever(retriever_options)

        # Initialize tool registry for task operations
        self.tool_registry = TaskToolRegistry(storage=self.storage)

    def _create_llm_provider(self, llm_config: BaseLlmConfig) -> AsyncLLMBase:
        """Create an LLM instance with the specified config."""
        if isinstance(llm_config, OllamaConfig):
            return OllamaLLM(llm_config)
        # elif isinstance(llm_config, AnthropicConfig):  # Uncomment when needed
        #     return AnthropicLLM(llm_config)
        else:
            # Here we now use our custom Logger instead of the built-in logging:
            Logger.warn("Unknown config type; converting to OllamaConfig")

            # Convert base config to OllamaConfig with defaults
            llm_config = OllamaConfig(
                model=llm_config.model or "llama3.1:8b-instruct-q8_0",
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                top_p=llm_config.top_p,
                resources=llm_config.resources,
            )
            return OllamaLLM(llm_config)

    async def process_request(
        self, input_text: str, kpu_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Entry point for processing a user request."""
        try:
            # 1. Input validation
            if not input_text.strip():
                raise ValidationError(
                    message="Input text cannot be empty",
                    validation_errors=[
                        {
                            "loc": ("input",),
                            "msg": "Input text cannot be empty",
                            "type": "value_error",
                        }
                    ],
                )

            try:
                # 2. LLM Analysis
                try:
                    analysis = await self.error_handler.handle_llm_operation(
                        self._analyze_request, input_text, kpu_context
                    )
                except LLMError:
                    # Directly raise new TaskExpertError
                    raise TaskExpertError("Failed to process request")

                # 3. Task Creation
                tasks_created = []
                try:
                    tasks_created = await self._create_and_validate_tasks(
                        analysis.get("tasks", [])
                    )
                except StorageError:
                    # Directly raise new TaskExpertError
                    raise TaskExpertError("Failed to process request")
                except ValidationError as e:
                    Logger.warn(f"Task validation failed: {str(e)}")
                    return {"tasks_created": [], "monitoring_setup": "none"}

                # Add priority calculation
                for task in tasks_created:
                    priority_score = await self.calculate_task_priority(
                        task.model_dump(), kpu_context
                    )
                    await self.storage.update_task(
                        task.id, {"priority_score": priority_score}
                    )

                # Reorder queue
                tasks_created = await self.reorder_task_queue(tasks_created)

                # 4. Monitoring Setup
                monitoring_error = None
                try:
                    await self._setup_monitoring(
                        tasks_created,
                        analysis.get("monitoring_requirements", {}),
                    )
                except MonitoringError as me:
                    monitoring_error = me
                    Logger.error(f"Monitoring setup failed: {str(me)}")

                return {
                    "tasks_created": tasks_created,
                    "monitoring_setup": "complete",
                    "priority_scores": [
                        await self.calculate_task_priority(
                            t.model_dump(), kpu_context
                        )
                        for t in tasks_created
                    ],
                }

            except TaskExpertError:
                raise
            except Exception as e:
                Logger.error(f"Unexpected error: {str(e)}")
                raise TaskExpertError("Failed to process request")

        except ValidationError:
            raise
        except Exception as e:
            Logger.error(f"Unexpected exception: {str(e)}")
            raise TaskExpertError("Failed to process request")

    async def _analyze_request(
        self, input_text: str, kpu_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use the LLM to analyze the request and produce a structured plan in JSON format.
        """
        enriched_context = kpu_context.get("enriched_context", "")
        implications = kpu_context.get("implications", "")
        decomposition = kpu_context.get("suggested_decomposition", "")
        collaborators = kpu_context.get("collaborators", "")

        prompt = f"""
        {self.system_prompt}

        User Request: {input_text}

        Additional Context: {enriched_context}
        Potential Implications: {implications}
        Suggested Breakdown: {decomposition}
        Potential Collaborators: {collaborators}
        """

        response_text = await self.llm.generate(prompt)

        try:
            analysis = json.loads(response_text)
            if not isinstance(analysis, dict):
                raise ValueError("Analysis did not return a JSON object.")
        except (ValueError, json.JSONDecodeError) as e:
            Logger.warn(f"JSON parsing error in analysis: {str(e)}")
            analysis = {"tasks": [], "monitoring_requirements": {}}

        return analysis

    async def _create_and_validate_tasks(
        self, tasks_list: List[Dict[str, Any]]
    ) -> List[Any]:
        """
        Creates tasks based on the provided list from the LLM analysis.
        Validates data, handles dependencies, and logs any failures.
        """
        tasks_created = []

        for task in tasks_list:
            task_type = task.get("type")
            task_details = task.get("details", {})

            try:
                # Validate task constraints first
                await self._validate_task_constraints(task_details)

                # Validate task data using TaskInputModel
                validated_data = TaskInputModel(**task_details).model_dump(
                    exclude_unset=True
                )

                # Create the task with the appropriate tool function
                if task_type == "agent_task":
                    result = await self.error_handler.handle_storage_operation(
                        self.tool_registry.execute,
                        "create_task",
                        validated_data,
                    )
                elif task_type == "human_task":
                    result = await self.error_handler.handle_storage_operation(
                        self.tool_registry.execute,
                        "create_human_task",
                        validated_data,
                    )
                else:
                    Logger.warn(
                        f"Unknown task type encountered: {task_type}"
                    )
                    continue

                tasks_created.append(result)
                Logger.info(f"Task created successfully: {result}")

            except StorageError:
                # Important: Re-raise StorageError to be caught by process_request
                raise
            except ValidationError as e:
                Logger.warn(f"Task validation failed: {str(e)}")
                continue
            except Exception as e:
                Logger.error(
                    f"Unexpected error during task creation: {str(e)}"
                )
                continue

        return tasks_created

    async def _setup_monitoring(
        self, tasks_created: List[Any], monitoring_requirements: Dict[str, Any]
    ) -> None:
        """
        Setup monitoring for created tasks and send notifications.
        """
        for task_id in tasks_created:
            try:
                # Send task creation notification
                await self.notifications.send(
                    "system",
                    {
                        "type": "task_created",
                        "task_id": task_id,
                        "timestamp": datetime.now().isoformat(),
                        "requirements": monitoring_requirements.get(
                            task_id, {}
                        ),
                    },
                )

                # Set up any specific monitoring requirements
                if task_specific_reqs := monitoring_requirements.get(task_id):
                    await self._setup_task_specific_monitoring(
                        task_id, task_specific_reqs
                    )

                Logger.info(f"Monitoring setup complete for task {task_id}")

            except Exception as e:
                raise MonitoringError(
                    message=f"Monitoring setup failed for task {task_id}: {str(e)}",
                    monitoring_config={"task_id": task_id},
                )

    async def _validate_task_constraints(
        self, task_data: Dict[str, Any]
    ) -> bool:
        """
        Validate task constraints before creation.
        Raises ValidationError if constraints are not met.
        """
        try:
            # Validate due date
            if due_date := task_data.get("due_date"):
                if isinstance(due_date, str):
                    due_date = datetime.fromisoformat(due_date)
                if due_date < datetime.now():
                    raise ValidationError(
                        validation_errors=[
                            {
                                "loc": ("due_date",),
                                "msg": "Due date cannot be in the past",
                                "type": "value_error",
                            }
                        ]
                    )

            # Validate priority if present
            if priority := task_data.get("priority"):
                if priority not in TaskPriority.__members__:
                    raise ValidationError(
                        validation_errors=[
                            {
                                "loc": ("priority",),
                                "msg": f"Invalid priority value: {priority}",
                                "type": "value_error",
                            }
                        ]
                    )

            # Validate task title and description
            if not task_data.get("title", "").strip():
                raise ValidationError(
                    validation_errors=[
                        {
                            "loc": ("title",),
                            "msg": "Task title cannot be empty",
                            "type": "value_error",
                        }
                    ]
                )

            return True
        except Exception as e:
            Logger.error(f"Task validation failed: {str(e)}")
            raise

    async def _setup_task_specific_monitoring(
        self, task_id: str, requirements: Dict[str, Any]
    ) -> None:
        """
        Helper method to setup task-specific monitoring requirements.
        """
        # Implementation can be added based on specific monitoring needs
        pass

    async def _get_model_response(
        self, messages: List[Dict[str, str]], retry_count: int = 2
    ) -> Dict[str, Any]:
        """
        Get response from the model with fallback and retry logic.
        Not currently used in the main flow, but kept here for potential use cases.
        """
        last_exception = None
        for attempt in range(retry_count):
            try:
                return await self.llm.generate(messages)
            except Exception as e:
                last_exception = e
                Logger.warn(f"Attempt {attempt + 1} failed: {str(e)}")
                await asyncio.sleep(1)  # Wait before retrying

        # If retries are exhausted, attempt fallback model
        if self.fallback_llm:
            Logger.warn("Attempting fallback model")
            try:
                return await self.fallback_llm.generate(messages)
            except Exception as fallback_exception:
                Logger.error(
                    f"Fallback model response error: {str(fallback_exception)}"
                )
                raise TaskExpertError(
                    "Fallback model failed"
                ) from fallback_exception

        # If no fallback or fallback failed
        raise TaskExpertError("Model failed after retries") from last_exception

    async def add_task(self, task_data: Dict[str, Any]) -> str:
        """
        Directly add a new task to the storage (example usage).
        """
        task_id = str(uuid4())
        task = Task(
            id=task_id,
            **task_data,
            created_at=datetime.now(),
            last_updated=datetime.now(),
            status=TaskStatus.PENDING,
        )
        await self.storage.save_task(task)
        return task_id

    async def update_task_status(
        self, task_id: str, status: TaskStatus, progress: float
    ) -> Task:
        task = await self.storage.fetch_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        task.status = status
        task.progress = progress
        task.last_updated = datetime.now(timezone.utc)

        # Use storage's update_task method
        updated = await self.storage.update_task(task_id, task)
        if not updated:
            raise ValueError(f"Failed to update task {task_id}")

        return task

    async def _store_interaction(
        self,
        user_id: str,
        session_id: str,
        reasoning: Dict[str, Any],
        action: Dict[str, Any],
        observation: Dict[str, Any],
        next_steps: Dict[str, Any],
    ) -> None:
        """
        Store interaction history in the storage.
        """
        await self.storage.save_interaction(
            {
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "reasoning": reasoning,
                "action": action,
                "observation": observation,
                "next_steps": next_steps,
            }
        )

    async def calculate_task_priority(
        self, task_data: Dict[str, Any] | Task, kpu_context: Dict[str, Any]
    ) -> float:
        """
        Calculate priority score (0-100) based on multiple factors
        """
        # Convert Task object to dict if needed
        if isinstance(task_data, Task):
            task_data = task_data.model_dump()

        priority_str = task_data.get("priority", "LOW")
        if isinstance(priority_str, TaskPriority):
            priority_str = priority_str.value

        base_score = {
            TaskPriority.LOW.value: 25,
            TaskPriority.MEDIUM.value: 50,
            TaskPriority.HIGH.value: 75,
            TaskPriority.URGENT.value: 100,
        }.get(
            priority_str, 25
        )  # Default to LOW if missing

        # Time factor: Higher score for closer due dates
        time_factor = 0
        if due_date := task_data.get("due_date"):
            try:
                due_date_obj = (
                    datetime.fromisoformat(due_date)
                    if isinstance(due_date, str)
                    else due_date
                )
                time_until_due = (
                    due_date_obj - datetime.now(timezone.utc)
                ).total_seconds()
                if time_until_due > 0:
                    time_factor = max(
                        0,
                        min(
                            25,
                            (7 * 24 * 3600 - time_until_due)
                            / (7 * 24 * 3600)
                            * 25,
                        ),
                    )
            except ValueError:
                Logger.warn(f"Invalid due_date format: {due_date}")

        # Dependency factor: More dependencies = higher priority
        dependency_count = len(task_data.get("dependencies", []))
        dependency_factor = min(
            25, dependency_count * 5
        )  # 5 points per dependency, max 25

        # Context factor: Use KPU enrichment
        context_factor = 0
        if importance_signals := kpu_context.get("importance_signals", []):
            context_factor = min(25, len(importance_signals) * 5)

        # Calculate final score
        final_score = (
            base_score + time_factor + dependency_factor + context_factor
        )
        return min(100, final_score)

    async def reorder_task_queue(
        self, tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Reorder tasks based on priority scores
        """
        if not tasks:
            return []

        # Calculate priorities
        task_scores = []
        for task in tasks:
            score = await self.calculate_task_priority(task, self.kpu_context)
            task_scores.append((task, score))

        # Sort by score descending
        return [
            task
            for task, _ in sorted(
                task_scores, key=lambda x: x[1], reverse=True
            )
        ]
