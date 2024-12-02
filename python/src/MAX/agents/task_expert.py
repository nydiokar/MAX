# MAO + REACT
from typing import List, Dict, Any, Optional, Union, AsyncIterable
from datetime import datetime, timedelta
from enum import Enum
import json
import asyncio
from uuid import uuid4
from pydantic import BaseModel, field_validator

from MAX.types import (
    ConversationMessage,
    ParticipantRole,
    TemplateVariables
)
from MAX.utils import Logger, conversation_to_dict
from MAX.retrievers import Retriever

# Data Models
class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    OVERDUE = "overdue"
    BLOCKED = "blocked"

class TaskPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class TaskModel(BaseModel):
    id: str
    title: str
    description: str
    assigned_agent: str
    status: TaskStatus
    priority: TaskPriority
    created_at: datetime
    due_date: Optional[datetime]
    dependencies: List[str]
    progress: float
    last_updated: datetime
    metadata: Dict[str, Any]

    @field_validator('progress')
    @classmethod
    def validate_progress(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Progress must be between 0 and 100')
        return v

# Storage Interface
class TaskStorage:
    """Abstract base class for task storage"""
    async def save_task(self, task: TaskModel) -> None:
        raise NotImplementedError
        
    async def get_task(self, task_id: str) -> Optional[TaskModel]:
        raise NotImplementedError
        
    async def get_tasks(self, filters: Dict[str, Any]) -> List[TaskModel]:
        raise NotImplementedError
        
    async def delete_task(self, task_id: str) -> None:
        raise NotImplementedError

# Notification Interface
class NotificationService:
    """Abstract base class for notification service"""
    async def send(self, agent_id: str, notification: Dict[str, Any]) -> None:
        raise NotImplementedError


class TaskExpertAgent:
    def __init__(self, options):
        # Import Agent locally to avoid circular import
        from .options import TaskExpertOptions
        from .agent import Agent

        # Convert TaskExpertOptions to AgentOptions for base class
        agent_options = options.to_agent_options()
        self.agent = Agent(agent_options)
        
        # Initialize task-specific attributes
        self.storage = options.storage_client
        self.notifications = options.notification_service
        self.agent_registry = options.agent_registry
        self.default_ttl = options.default_task_ttl
        self.retriever = options.retriever
        
        # REACT state management
        self.current_reasoning: Dict[str, Any] = {}
        self.last_observation: Dict[str, Any] = {}
        self.action_history: List[Dict[str, Any]] = []

        # Configure system prompt for task management
        self.system_prompt = """You are a Task Expert Agent responsible for managing and coordinating tasks across multiple AI agents.
        Your primary responsibilities include:
        1. Creating and managing tasks based on user requests
        2. Monitoring task progress and dependencies
        3. Generating status reports
        4. Coordinating between agents
        
        Follow the REACT framework for all operations:
        1. Reason: Analyze the request and context
        2. Act: Execute planned actions
        3. Observe: Monitor results
        4. Think: Plan next steps
        
        Maintain clear documentation and provide detailed responses."""

    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: List[ConversationMessage],
        additional_params: Optional[Dict[str, str]] = None
    ) -> Union[ConversationMessage, AsyncIterable[Any]]:
        """Main request processing following MAO framework with REACT methodology"""
        try:
            # REACT: Reasoning Phase
            reasoning = await self._reason_about_request(input_text, chat_history)
            self.current_reasoning = reasoning
            
            # Get additional context if retriever is configured
            context = None
            if self.retriever:
                context = await self.retriever.retrieve_and_combine_results(input_text)
            
            # REACT: Action Phase
            action_plan = await self._create_action_plan(reasoning, context)
            
            # Execute actions
            result = await self._execute_actions(action_plan, user_id, session_id)
            
            # REACT: Observe Phase
            observation = self._observe_results(result)
            self.last_observation = observation
            
            # REACT: Think Phase
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

    async def _reason_about_request(
        self,
        input_text: str,
        chat_history: List[ConversationMessage]
    ) -> Dict[str, Any]:
        """Enhanced reasoning with structured output"""
        return {
            "type": self._classify_request(input_text),
            "context": self._extract_context(chat_history),
            "requirements": self._extract_requirements(input_text),
            "constraints": self._identify_constraints(input_text),
            "priorities": self._determine_priorities(input_text),
            "timestamp": datetime.now().isoformat(),
            "reasoning_steps": [
                self._analyze_intent(input_text),
                self._check_prerequisites(input_text),
                self._evaluate_complexity(input_text)
            ]
        }

    def _classify_request(self, input_text: str) -> str:
        """Classify the type of task request"""
        # Implement classification logic using regex or ML model
        if "create" in input_text.lower() or "new task" in input_text.lower():
            return "create_task"
        elif "update" in input_text.lower() or "modify" in input_text.lower():
            return "update_task"
        elif "delete" in input_text.lower() or "remove" in input_text.lower():
            return "delete_task"
        elif "status" in input_text.lower() or "report" in input_text.lower():
            return "report_status"
        else:
            return "query_tasks"

    async def _execute_actions(
        self,
        action_plan: List[Dict[str, Any]],
        user_id: str,
        session_id: str
    ) -> Dict[str, Any]:
        """Execute planned actions with error handling and logging"""
        results = []
        for action in action_plan:
            try:
                if action["type"] == "create_task":
                    task = await self.create_task(action["data"])
                    results.append({"type": "task_created", "task_id": task.id})
                elif action["type"] == "update_task":
                    task = await self.update_task_status(
                        action["data"]["task_id"],
                        action["data"]["status"],
                        action["data"]["progress"]
                    )
                    results.append({"type": "task_updated", "task_id": task.id})
                elif action["type"] == "generate_report":
                    report = await self.generate_status_report(action["data"].get("filters"))
                    results.append({"type": "report_generated", "report": report})
                
                # Log successful action
                Logger.info(f"Action executed successfully: {action['type']}")
                
            except Exception as e:
                Logger.error(f"Action execution failed: {str(e)}")
                results.append({"type": "error", "action": action["type"], "error": str(e)})
        
        return {"results": results}

    async def _store_interaction(
        self,
        user_id: str,
        session_id: str,
        reasoning: Dict[str, Any],
        action: Dict[str, Any],
        observation: Dict[str, Any],
        next_steps: Dict[str, Any]
    ) -> None:
        """Store interaction history for analysis and debugging"""
        interaction = {
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "reasoning": reasoning,
            "action": action,
            "observation": observation,
            "next_steps": next_steps
        }
        
        # Store in your preferred storage solution
        await self.storage.save_interaction(interaction)

    def _format_response(self, next_steps: Dict[str, Any]) -> ConversationMessage:
        """Format response following MAO's conversation structure with enhanced metadata"""
        return ConversationMessage(
            role=ParticipantRole.ASSISTANT.value,
            content=[{
                "text": next_steps["response_text"],
                "metadata": {
                    "reasoning": self.current_reasoning,
                    "observation": self.last_observation,
                    "next_steps": next_steps,
                    "agent_info": {
                        "name": self.agent.name,
                        "version": "1.0",
                        "capabilities": ["task_management", "coordination", "reporting"]
                    }
                }
            }]
        )

    async def add_recurring_task(self, agent_id, start_time, interval_hours, end_time, task_data):
        """Adds a recurring task to the database."""
        task_id = str(uuid4())
        await self.storage.save_task({
            "id": task_id,
            "agent_id": agent_id,
            "next_run_time": start_time,
            "repeat_interval": interval_hours,
            "end_time": end_time,
            "data": task_data,
            "status": TaskStatus.PENDING.value  # Mark as pending
        })
        Logger.info(f"Recurring task {task_id} added for agent {agent_id}.")
        return task_id

    async def check_and_execute_tasks(self):
        """Periodically checks and executes due tasks."""
        while True:
            try:
                # Fetch tasks that are due for execution
                tasks_due = await self.storage.get_tasks({"next_run_time <= ": datetime.now()})
                
                for task in tasks_due:
                    try:
                        # Execute the task
                        agent_id = task["agent_id"]
                        task_data = task["data"]
                        await self.execute_task(agent_id, task_data)
                        
                        # Update the next run time or mark as completed
                        next_run = task["next_run_time"] + timedelta(hours=task["repeat_interval"])
                        if next_run <= task["end_time"]:
                            await self.storage.update_task(task["id"], {"next_run_time": next_run})
                        else:
                            await self.storage.update_task(task["id"], {"status": TaskStatus.COMPLETED.value})
                            Logger.info(f"Task {task['id']} marked as completed.")
                    
                    except Exception as e:
                        Logger.error(f"Failed to execute task {task['id']}: {str(e)}")
                        await self.storage.update_task(task["id"], {"status": TaskStatus.BLOCKED.value})
                
                # Wait before checking again
                await asyncio.sleep(60)  # Adjust based on desired precision
            except Exception as e:
                Logger.error(f"Error during task execution loop: {str(e)}")

    async def execute_task(self, agent_id, task_data):
        """Passes the task to the specified agent for execution."""
        if agent_id not in self.agent_registry:
            raise ValueError(f"Agent {agent_id} is not registered.")
        
        agent = self.agent_registry[agent_id]
        Logger.info(f"Passing task to agent {agent_id}: {task_data}")
        await agent.execute_task(task_data)

