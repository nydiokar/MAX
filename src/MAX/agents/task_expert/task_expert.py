import asyncio
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import uuid4

from MAX.storage.models import TaskStatus, TaskPriority, TaskModel
from MAX.storage.protocols import TaskStorage, NotificationService
from MAX.utils import Logger
from MAX.retrievers import Retriever
from MAX.retrievers.kb_retriever import KnowledgeBasesRetrieverOptions, KnowledgeBasesRetriever
from MAX.llms import create_llm_provider
from MAX.agents.agent import Agent
from MAX.agents.task_expert.config import TaskExpertOptions
from MAX.agents.task_expert.tool_registry import TaskToolRegistry


class TaskExpertError(Exception):
    """Custom exception class for TaskExpert-specific errors."""
    pass


class TaskExpertAgent(Agent):
    """
    Task Expert Agent is responsible for interpreting user requests related to tasks,
    creating or updating tasks in the database, and optionally sending notifications.
    
    Workflow:
    1. Analyze the user request with the LLM to determine the tasks to create or update.
    2. Use the registered tools to interact with the task storage (create tasks, update them, etc.).
    3. Return a structured result indicating what was done.
    """
    def __init__(self, options: TaskExpertOptions):
        super().__init__(options)
        
        # Set references to storage and notification services
        self.storage: TaskStorage = options.storage_client
        self.notifications: NotificationService = options.notification_service
        self.default_ttl = options.default_task_ttl
        self.retriever: Optional[Retriever] = options.retriever
        
        # Set up the main LLM
        self.llm = create_llm_provider(
            engine="ollama",
            model_id=options.model_id,
            temperature=options.temperature,
            max_tokens=options.max_tokens,
            top_p=options.top_p
        )

        # Optional fallback LLM
        if options.fallback_model_id:
            self.fallback_llm = create_llm_provider(
                engine="ollama",
                model_id=options.fallback_model_id
            )

        # Set a clear, stable system prompt
        # This prompt instructs the LLM to always return a structured JSON.
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
                similarity_threshold=0.7
            )
            self.retriever = KnowledgeBasesRetriever(retriever_options)

        # Initialize tool registry for task operations
        self.tools = TaskToolRegistry()

    async def process_request(
        self,
        input_text: str,
        kpu_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a task-related request with KPU-enriched context.
        
        Steps:
        1. Analyze the request via the LLM to get a JSON plan.
        2. Execute the plan using tools (create or update tasks).
        3. Setup monitoring if required.
        4. Return a summary of what was done.
        """
        analysis = await self._analyze_request(input_text, kpu_context)

        # Execute the plan using task management tools
        tasks_created = []
        for task in analysis.get("tasks", []):
            task_type = task.get("type")
            task_details = task.get("details", {})
            
            if task_type == "agent_task":
                result = await self.tools.execute("create_task", task_details)
                tasks_created.append(result)
            elif task_type == "human_task":
                # Ensure that such a tool exists or handle accordingly
                result = await self.tools.execute("create_human_task", task_details)
                tasks_created.append(result)
            else:
                Logger.warning(f"Unknown task type encountered: {task_type}")

        # Set up monitoring and notifications if specified
        monitoring_reqs = analysis.get("monitoring_requirements", {})
        await self._setup_monitoring(tasks_created, monitoring_reqs)
        
        return {
            "tasks_created": tasks_created,
            "monitoring_setup": monitoring_reqs,
            "feedback_points": kpu_context.get('feedback_required', [])
        }

    async def _analyze_request(
        self,
        input_text: str,
        kpu_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use the LLM to analyze the request and produce a structured plan in JSON format.
        """
        enriched_context = kpu_context.get('enriched_context', '')
        implications = kpu_context.get('implications', '')
        decomposition = kpu_context.get('suggested_decomposition', '')
        collaborators = kpu_context.get('collaborators', '')

        prompt = f"""
        {self.system_prompt}

        User Request: {input_text}

        Additional Context: {enriched_context}
        Potential Implications: {implications}
        Suggested Breakdown: {decomposition}
        Potential Collaborators: {collaborators}
        """

        response_text = await self.llm.generate(prompt)
        
        # Attempt to parse the LLM output as JSON
        try:
            analysis = json.loads(response_text)
            if not isinstance(analysis, dict):
                raise ValueError("Analysis did not return a JSON object.")
        except (ValueError, json.JSONDecodeError) as e:
            Logger.warning(f"JSON parsing error in analysis: {str(e)}")
            analysis = {"tasks": [], "monitoring_requirements": {}}

        return analysis

    async def _get_model_response(
        self,
        messages: List[Dict[str, str]],
        retry_count: int = 2
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
                Logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                await asyncio.sleep(1)  # Wait before retrying

        # If retries are exhausted, attempt fallback model
        if hasattr(self, 'fallback_llm'):
            Logger.warning("Attempting fallback model")
            try:
                return await self.fallback_llm.generate(messages)
            except Exception as fallback_exception:
                Logger.error(f"Fallback model response error: {str(fallback_exception)}")
                raise TaskExpertError("Fallback model failed") from fallback_exception

        # If no fallback or fallback failed
        raise TaskExpertError("Model failed after retries") from last_exception

    async def add_task(self, task_data: Dict[str, Any]) -> str:
        """Add a new task to the storage."""
        task_id = str(uuid4())
        task = TaskModel(
            id=task_id,
            **task_data,
            created_at=datetime.now(),
            last_updated=datetime.now(),
            status=TaskStatus.PENDING
        )
        await self.storage.save_task(task)
        return task_id

    async def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        progress: float
    ) -> TaskModel:
        """Update the status and progress of an existing task."""
        task = await self.storage.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
            
        task.status = status
        task.progress = progress
        task.last_updated = datetime.now()
        
        await self.storage.save_task(task)
        return task

    async def _setup_monitoring(
        self,
        tasks_created: List[Any],
        monitoring_requirements: Dict[str, Any]
    ) -> None:
        """
        Setup monitoring for created tasks if required.
        Here you can implement notification logic or periodic checks.
        For now, this is a placeholder.
        """
        # Implement your monitoring logic here if needed
        pass

    async def _store_interaction(
        self,
        user_id: str,
        session_id: str,
        reasoning: Dict[str, Any],
        action: Dict[str, Any],
        observation: Dict[str, Any],
        next_steps: Dict[str, Any]
    ) -> None:
        """Store interaction history in the storage."""
        await self.storage.save_interaction({
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "reasoning": reasoning,
            "action": action,
            "observation": observation,
            "next_steps": next_steps
        })
