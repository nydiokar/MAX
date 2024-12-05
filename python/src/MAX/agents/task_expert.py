import asyncio
from typing import List, Dict, Any, Optional, Union, AsyncIterable
from datetime import datetime, timedelta
from enum import Enum
from uuid import uuid4
from ..utils.interfaces import TaskStorage, NotificationService, TaskStatus, TaskPriority, TaskModel
from ..types import ConversationMessage, ParticipantRole
from ..utils import Logger
from MAX.retrievers import Retriever
from MAX.retrievers.kb_retriever import KnowledgeBasesRetrieverOptions, KnowledgeBasesRetriever
from MAX.adapters.llm import create_llm_provider
from .agent import Agent
from ..utils.options import TaskExpertOptions

class TaskExpertError(Exception):
    """Custom exception class for TaskExpert-specific errors"""
    pass

class TaskExpertAgent(Agent):
    """
    Task Expert Agent responsible for managing and coordinating tasks.
    
    This agent implements the REACT framework (Reason, Act, Observe, Think)
    for processing tasks and maintains state through a storage system.
    
    Attributes:
        storage: TaskStorage instance for persistent storage
        notifications: NotificationService for alert handling
        llm: Language Model provider for task processing
        retriever: Optional knowledge base retriever
        
    The agent processes requests through the following steps:
    1. Retrieves relevant context
    2. Reasons about the request
    3. Creates and executes action plans
    4. Observes results and plans next steps
    """
    def __init__(self, options: TaskExpertOptions):
        super().__init__(options)
        
        # Initialize task-specific attributes
        self.storage = options.storage_client
        self.notifications = options.notification_service
        self.default_ttl = options.default_task_ttl
        self.retriever = options.retriever
        
        # Set up LLM
        self.llm = create_llm_provider(
            "ollama",
            model_id=options.model_id,
            temperature=options.temperature,
            max_tokens=options.max_tokens,
            top_p=options.top_p
        )
    
        if options.fallback_model_id:
            self.fallback_llm = create_llm_provider(
                "ollama",
                model_id=options.fallback_model_id
            )
        
        # REACT state management
        self.current_reasoning: Dict[str, Any] = {}
        self.last_observation: Dict[str, Any] = {}
        self.action_history: List[Dict[str, Any]] = []

        # Configure system prompt
        self.system_prompt = """You are a Task Expert Agent responsible for managing and coordinating tasks.
        Your primary responsibilities include:
        1. Creating and managing tasks based on user requests
        2. Monitoring task progress and dependencies
        3. Generating status reports
        4. Coordinating task execution
        
        Follow the REACT framework for all operations:
        1. Reason: Analyze the request and context
        2. Act: Execute planned actions
        3. Observe: Monitor results
        4. Think: Plan next steps
        
        Maintain clear documentation and provide detailed responses."""

        # Initialize retriever if needed
        if self.retriever is None and self.storage is not None:
            retriever_options = KnowledgeBasesRetrieverOptions(
                storage_client=self.storage,
                collection_name="knowledge_base",
                max_results=5,
                similarity_threshold=0.7
            )
            self.retriever = KnowledgeBasesRetriever(retriever_options)

    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: List[ConversationMessage],
        additional_params: Optional[Dict[str, str]] = None
    ) -> Union[ConversationMessage, AsyncIterable[Any]]:
        Logger.info(f"Processing request for user {user_id} in session {session_id}")
        try:
            # Get context from knowledge base
            context = await self.retriever.retrieve_and_combine_results(input_text) if self.retriever else ""
            Logger.debug(f"Retrieved context: {context[:100]}...")

            # REACT Framework Implementation
            # 1. Reasoning Phase
            reasoning = await self._reason_about_request(input_text, chat_history, context)
            Logger.info(f"Reasoning complete with type: {reasoning.get('type', 'unknown')}")

            # 2. Action Phase
            action_plan = await self._create_action_plan(reasoning)
            result = await self._execute_actions(action_plan, user_id, session_id)

            # 3. Observe Phase
            observation = self._observe_results(result)
            self.last_observation = observation

            # 4. Think Phase
            next_steps = self._plan_next_steps(observation)

            # Store interaction history
            await self._store_interaction(
                user_id=user_id,
                session_id=session_id,
                reasoning=reasoning,
                action=action_plan,
                observation=observation,
                next_steps=next_steps
            )

            return self._format_response(next_steps)

        except Exception as e:
            Logger.error(f"Error in task processing: {str(e)}")
            return self._handle_error(e)

    async def _get_model_response(
        self,
        messages: List[Dict[str, str]],
        retry_count: int = 2
    ) -> Dict[str, Any]:
        """Get response from the model with fallback and retry logic."""
        last_exception = None
        for attempt in range(retry_count):
            try:
                return await self.llm.generate(messages)
            except Exception as e:
                last_exception = e
                Logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                await asyncio.sleep(1)  # Wait for 1 second before retrying

        # If retries are exhausted, attempt fallback model
        if hasattr(self, 'fallback_llm'):
            Logger.warning("Attempting fallback model")
            try:
                return await self.fallback_llm.generate(messages)
            except Exception as fallback_exception:
                Logger.error(f"Fallback model response error: {str(fallback_exception)}")
                raise TaskExpertError("Fallback model failed") from fallback_exception

        # If no fallback model or fallback failed
        raise TaskExpertError("Model failed after retries") from last_exception

    async def add_task(self, task_data: Dict[str, Any]) -> str:
        """Add a new task."""
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
        """Update task status and progress."""
        task = await self.storage.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
            
        task.status = status
        task.progress = progress
        task.last_updated = datetime.now()
        
        await self.storage.save_task(task)
        return task

    async def _store_interaction(
        self,
        user_id: str,
        session_id: str,
        reasoning: Dict[str, Any],
        action: Dict[str, Any],
        observation: Dict[str, Any],
        next_steps: Dict[str, Any]
    ) -> None:
        """Store interaction history."""
        await self.storage.save_interaction({
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "reasoning": reasoning,
            "action": action,
            "observation": observation,
            "next_steps": next_steps
        })