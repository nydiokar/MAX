from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from MAX.types.base_types import Message
from MAX.types.collaboration_types import SubTask, TaskState
from MAX.utils.logger import Logger

class StateType(Enum):
    TASK = "task"
    MESSAGE = "message"
    AGENT = "agent"
    WORKFLOW = "workflow"

@dataclass
class StateEntry:
    """Represents a single state entry with metadata."""
    data: Any
    timestamp: datetime
    version: int
    state_type: StateType
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = None

class UnifiedStateManager:
    """Centralized state management for the Multi-Agent system."""
    
    def __init__(self):
        self._state: Dict[str, StateEntry] = {}
        self._message_queues: Dict[str, List[Message]] = {}
        self._task_dependencies: Dict[str, Set[str]] = {}
        self._version_history: Dict[str, List[StateEntry]] = {}
        self._subscribers: Dict[str, List[callable]] = {}

    async def update_state(
        self,
        state_id: str,
        data: Any,
        state_type: StateType,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update state with versioning and notification."""
        current_version = 0
        if state_id in self._state:
            current_version = self._state[state_id].version

        new_entry = StateEntry(
            data=data,
            timestamp=datetime.utcnow(),
            version=current_version + 1,
            state_type=state_type,
            parent_id=parent_id,
            metadata=metadata or {}
        )

        # Store the previous version in history
        if state_id not in self._version_history:
            self._version_history[state_id] = []
        if state_id in self._state:
            self._version_history[state_id].append(self._state[state_id])

        # Update current state
        self._state[state_id] = new_entry

        # Notify subscribers
        await self._notify_subscribers(state_id, new_entry)

    async def get_state(
        self,
        state_id: str,
        version: Optional[int] = None
    ) -> Optional[StateEntry]:
        """Retrieve state, optionally from a specific version."""
        if version is None:
            return self._state.get(state_id)

        if state_id not in self._version_history:
            return None

        # Find specific version in history
        for entry in reversed(self._version_history[state_id]):
            if entry.version == version:
                return entry

        return None

    async def update_task_state(
        self,
        task_id: str,
        task_data: SubTask,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update task-specific state."""
        await self.update_state(
            state_id=task_id,
            data=task_data,
            state_type=StateType.TASK,
            metadata=metadata
        )

    async def enqueue_message(
        self,
        queue_id: str,
        message: Message
    ) -> None:
        """Add message to a queue."""
        if queue_id not in self._message_queues:
            self._message_queues[queue_id] = []
        self._message_queues[queue_id].append(message)

        # Update state to track queue changes
        await self.update_state(
            state_id=f"queue_{queue_id}",
            data={"queue_length": len(self._message_queues[queue_id])},
            state_type=StateType.MESSAGE,
            metadata={"last_message_timestamp": datetime.utcnow()}
        )

    async def dequeue_messages(
        self,
        queue_id: str,
        max_messages: int = 10
    ) -> List[Message]:
        """Retrieve and remove messages from a queue."""
        if queue_id not in self._message_queues:
            return []

        messages = []
        while len(messages) < max_messages and self._message_queues[queue_id]:
            messages.append(self._message_queues[queue_id].pop(0))

        # Update state to track queue changes
        if messages:
            await self.update_state(
                state_id=f"queue_{queue_id}",
                data={"queue_length": len(self._message_queues[queue_id])},
                state_type=StateType.MESSAGE,
                metadata={"last_dequeue_timestamp": datetime.utcnow()}
            )

        return messages

    async def add_task_dependency(
        self,
        task_id: str,
        dependency_id: str
    ) -> None:
        """Add a dependency between tasks."""
        if task_id not in self._task_dependencies:
            self._task_dependencies[task_id] = set()
        self._task_dependencies[task_id].add(dependency_id)

        await self.update_state(
            state_id=f"dependencies_{task_id}",
            data=list(self._task_dependencies[task_id]),
            state_type=StateType.TASK,
            parent_id=task_id,
            metadata={"updated_at": datetime.utcnow()}
        )

    async def get_task_dependencies(
        self,
        task_id: str
    ) -> Set[str]:
        """Get all dependencies for a task."""
        return self._task_dependencies.get(task_id, set())

    async def subscribe_to_state(
        self,
        state_id: str,
        callback: callable
    ) -> None:
        """Subscribe to state changes."""
        if state_id not in self._subscribers:
            self._subscribers[state_id] = []
        self._subscribers[state_id].append(callback)

    async def unsubscribe_from_state(
        self,
        state_id: str,
        callback: callable
    ) -> None:
        """Unsubscribe from state changes."""
        if state_id in self._subscribers:
            self._subscribers[state_id] = [
                cb for cb in self._subscribers[state_id]
                if cb != callback
            ]

    async def _notify_subscribers(
        self,
        state_id: str,
        state_entry: StateEntry
    ) -> None:
        """Notify all subscribers of state changes."""
        if state_id in self._subscribers:
            for callback in self._subscribers[state_id]:
                try:
                    await callback(state_entry)
                except Exception as e:
                    Logger.error(f"Error notifying subscriber for {state_id}: {str(e)}")

    def cleanup_state(
        self,
        max_history_entries: int = 100
    ) -> None:
        """Clean up old state history entries."""
        for state_id in self._version_history:
            if len(self._version_history[state_id]) > max_history_entries:
                # Keep only the most recent entries
                self._version_history[state_id] = \
                    self._version_history[state_id][-max_history_entries:]
