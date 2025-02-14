from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from dataclasses import dataclass
import asyncio
from MAX.storage import MongoDBChatStorage, ChromaDBChatStorage
from MAX.types import ConversationMessage, ParticipantRole
from MAX.utils import Logger
from MAX.config.database_config import DatabaseConfig


@dataclass
class SystemState:
    agent_states: Dict[str, Dict[str, Any]]
    conversation_states: Dict[str, Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: datetime

    @property
    def is_healthy(self) -> bool:
        return all(
            agent.get("status") == "active"
            for agent in self.agent_states.values()
        )


class StateManager:
    def __init__(self, config: DatabaseConfig):
        """Initialize state manager with configuration"""
        self._lock = asyncio.Lock()  # Add lock for state updates
        self.config = config
        self.mongo_storage: Optional[MongoDBChatStorage] = None
        self.vector_storage: Optional[ChromaDBChatStorage] = None
        self.system_state = SystemState(
            agent_states={},
            conversation_states={},
            metadata={},
            timestamp=datetime.now(timezone.utc),
        )
        
        # Add health check task
        self._health_check_task: Optional[asyncio.Task] = None
        self._persistence_task: Optional[asyncio.Task] = None

    async def start(self) -> bool:
        """Initialize and start background tasks."""
        try:
            if not await self._initialize_storages():
                return False

            # Start background tasks
            self._health_check_task = asyncio.create_task(
                self._run_periodic_health_check()
            )
            self._persistence_task = asyncio.create_task(
                self._run_periodic_state_persistence()
            )
            return True
        except Exception as e:
            Logger.error(f"Failed to start StateManager: {str(e)}")
            return False

    async def stop(self) -> None:
        """Cleanup and stop background tasks."""
        if self._health_check_task:
            self._health_check_task.cancel()
        if self._persistence_task:
            self._persistence_task.cancel()
            
        try:
            if self.mongo_storage:
                await self.mongo_storage.cleanup()
            if self.vector_storage:
                await self.vector_storage.cleanup()
        except Exception as e:
            Logger.error(f"Error during StateManager cleanup: {str(e)}")

    async def _initialize_storages(self) -> bool:
        """Initialize storage systems and restore state"""
        try:
            if not self.mongo_storage:
                self.mongo_storage = MongoDBChatStorage(
                    mongo_uri=self.config.mongodb.uri,
                    db_name=self.config.mongodb.database,
                    collection_name=self.config.mongodb.state_collection,
                )

            mongo_init = await self.mongo_storage.initialize()
            if not mongo_init:
                raise RuntimeError("Failed to initialize MongoDB storage")

            # Restore system state
            stored_state = await self.mongo_storage.get_system_state()
            if stored_state:
                self.system_state = SystemState(**stored_state)

            return True
        except Exception as e:
            Logger.error(f"Failed to initialize storages: {str(e)}")
            return False

    async def _run_periodic_health_check(self):
        """Run periodic health checks"""
        while True:
            try:
                health_status = await self.check_storage_health()
                if not health_status["healthy"]:
                    Logger.warn(
                        "Storage health check failed", extra=health_status
                    )
                await asyncio.sleep(
                    self.config.state_manager.health_check_interval_seconds
                )
            except Exception as e:
                Logger.error(f"Error in periodic health check: {str(e)}")
                await asyncio.sleep(60)

    async def check_storage_health(self) -> Dict[str, Any]:
        """Check health of all storage systems"""
        results = {
            "timestamp": datetime.now(timezone.utc),
            "storages": {},
            "healthy": True,
        }

        # Check MongoDB
        mongo_healthy, mongo_details = await self.mongo_storage.check_health()
        results["storages"]["mongodb"] = mongo_details
        results["healthy"] &= mongo_healthy

        # Check ChromaDB if enabled
        if self.vector_storage:
            vector_healthy, vector_details = (
                await self.vector_storage.check_health()
            )
            results["storages"]["chromadb"] = vector_details
            results["healthy"] &= vector_healthy

        return results

    async def _run_periodic_state_persistence(self):
        """Periodically persist system state to storage"""
        while True:
            try:
                await self._persist_system_state()
                await asyncio.sleep(
                    self.config.state_manager.state_persistence_interval_seconds
                )
            except Exception as e:
                Logger.error(f"Error in periodic state persistence: {str(e)}")
                await asyncio.sleep(60)  # Retry after a minute

    async def _persist_system_state(self) -> bool:
        """Persist current system state to storage"""
        try:
            current_state = self.system_state.__dict__
            await self.mongo_storage.save_system_state(
                state_type="SYSTEM_STATE",
                state_data=current_state,
                ttl=(
                    self.config.mongodb.ttl_hours * 3600
                    if self.config.mongodb.ttl_hours
                    else None
                ),
            )
            return True
        except Exception as e:
            Logger.error(f"Failed to persist system state: {str(e)}")
            return False

    async def restore_state_from_storage(self) -> bool:
        """Restore system state from persistent storage"""
        try:
            # Get latest system state
            state_data = await self.mongo_storage.get_system_state(
                "SYSTEM_STATE"
            )
            if state_data:
                self.system_state = SystemState(**state_data)

            # Restore agent states
            agent_states = await self.mongo_storage.fetch_chat(
                user_id="system", session_id="agent_states", agent_id="*"
            )

            for message in agent_states:
                try:
                    state_dict = eval(
                        message.content
                    )  # Safe since we control the content
                    agent_id = state_dict.get("agent_id")
                    if agent_id:
                        self.system_state.agent_states[agent_id] = state_dict
                except Exception as e:
                    Logger.error(f"Failed to restore system state: {str(e)}")
                    continue

            return True
        except Exception as e:
            Logger.error(f"Failed to restore system state: {str(e)}")
            return False

    async def track_conversation_state(
        self,
        user_id: str,
        session_id: str,
        message: ConversationMessage,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        async with self._lock:  # Ensure atomic updates
            try:
                # Validate message
                if not message.content or (
                    isinstance(message.content, list)
                    and len(message.content) == 0
                ):
                    return False

                conversation_key = f"{user_id}:{session_id}"
                current_time = datetime.now(timezone.utc)

                # Update conversation state atomically
                state = {
                    "last_message": {
                        "role": message.role,
                        "content": message.content,
                        "timestamp": current_time,
                    },
                    "metadata": metadata or {},
                    "timestamp": current_time,
                }

                # Save state and message atomically
                self.system_state.conversation_states[conversation_key] = state
                await self.mongo_storage.save_chat_message(
                    user_id=user_id,
                    session_id=session_id,
                    agent_id="conversation_state",
                    new_message=message,
                    metadata=metadata,
                )

                return True

            except Exception as e:
                Logger.error(f"Failed to track conversation state: {str(e)}")
                return False

    async def _save_message_to_storages(
        self,
        user_id: str,
        session_id: str,
        message: ConversationMessage,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save message to both MongoDB and vector storage if enabled"""
        await self.mongo_storage.save_chat_message(
            user_id=user_id,
            session_id=session_id,
            agent_id="conversation_state",
            new_message=message,
            metadata=metadata,
        )

        if self.vector_storage:
            await self.vector_storage.save_chat_message(
                user_id=user_id,
                session_id=session_id,
                agent_id="conversation_state",
                new_message=message,
            )

    async def update_agent_state(
        self, agent_id: str, state: Dict[str, Any]
    ) -> bool:
        """Update agent state and persist to storage"""
        try:
            current_time = datetime.now(timezone.utc)
            state_dict = (
                state.dict() if hasattr(state, "dict") else dict(state)
            )
            state_dict["last_updated"] = current_time
            state_dict["status"] = state_dict.get("status", "active")

            # Update in-memory state
            self.system_state.agent_states[agent_id] = state_dict

            # Create state message
            message = ConversationMessage(
                role=ParticipantRole.STATE.value,
                content=str(state_dict),
                timestamp=current_time,
            )

            # Persist to storage
            await self.mongo_storage.save_chat_message(
                user_id="system",
                session_id="agent_states",
                agent_id=agent_id,
                new_message=message,
            )

            return True

        except Exception as e:
            Logger.error(f"Failed to update agent state: {str(e)}")
            return False

    async def get_system_snapshot(self) -> SystemState:
        """Get current system state snapshot"""
        self.system_state.timestamp = datetime.now(timezone.utc)
        return self.system_state

    async def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get state for specific agent"""
        return self.system_state.agent_states.get(agent_id)

    async def _run_periodic_cleanup(self):
        """Run periodic cleanup of old states"""
        while True:
            try:
                await self.cleanup_old_states(
                    max_age_hours=self.config.state_manager.max_state_age_hours
                )
                await asyncio.sleep(
                    self.config.state_manager.state_cleanup_interval_hours
                    * 3600
                )
            except Exception as e:
                Logger.error(f"Error in periodic cleanup: {str(e)}")
                await asyncio.sleep(3600)  # Wait an hour before retrying

    async def cleanup_old_states(self, max_age_hours: int = 24) -> bool:
        """Clean up old state entries"""
        try:
            cutoff_time = datetime.now(timezone.utc).timestamp() - (
                max_age_hours * 3600
            )

            # Clean up in-memory state
            for conversation_key, state in list(
                self.system_state.conversation_states.items()
            ):
                if state["timestamp"].timestamp() < cutoff_time:
                    del self.system_state.conversation_states[conversation_key]

            return True
        except Exception as e:
            Logger.error(f"Failed to cleanup old states: {str(e)}")
            return False
