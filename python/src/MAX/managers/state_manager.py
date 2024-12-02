from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from dataclasses import dataclass
import asyncio
from MAX.storage.mongoDB import MongoDBChatStorage
from MAX.storage.chromaDB import ChromaDBChatStorage
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
        self.config = config
        self.mongo_storage = MongoDBChatStorage(
        mongo_uri=config.mongodb.uri,
        db_name=config.mongodb.database,
        collection_name=config.mongodb.state_collection,
        ttl_index=config.mongodb.ttl_hours * 3600 if config.mongodb.ttl_hours else None
    )
    
        if config.state_manager.enable_vector_storage:
            self.vector_storage = ChromaDBChatStorage(
                collection_name=config.chromadb.collection_name
            )
        else:
            self.vector_storage = None

        self.system_state = SystemState(
            agent_states={},
            conversation_states={},
            metadata={},
            timestamp=datetime.now(timezone.utc)
        )

        # Start background cleanup task
        asyncio.create_task(self._run_periodic_cleanup())

    async def update_agent_state(self, agent_id: str, state: Dict[str, Any]) -> bool:
        """Update agent state and persist to storage"""
        try:
            current_time = datetime.now(timezone.utc)
            state_dict = state.dict() if hasattr(state, 'dict') else dict(state)
            state_dict['last_updated'] = current_time
            state_dict['status'] = state_dict.get('status', 'active')
            
            # Update in-memory state
            self.system_state.agent_states[agent_id] = state_dict
            
            # Create state message
            message = ConversationMessage(
                role=ParticipantRole.STATE.value,
                content=str(state_dict),
                timestamp=current_time
            )
            
            # Persist to storage
            await self.mongo_storage.save_chat_message(
                user_id="system",
                session_id="agent_states",
                agent_id=agent_id,
                new_message=message
            )
            
            return True
            
        except Exception as e:
            Logger.error(f"Failed to update agent state: {str(e)}")
            return False

    async def track_conversation_state(
        self,
        user_id: str,
        session_id: str,
        message: ConversationMessage,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Track conversation state with metadata"""
        try:
            conversation_key = f"{user_id}:{session_id}"
            current_time = datetime.now(timezone.utc)
            
            # Update conversation state
            self.system_state.conversation_states[conversation_key] = {
                "last_message": {
                    "role": message.role,
                    "content": message.content,
                    "timestamp": current_time
                },
                "metadata": metadata or {},
                "timestamp": current_time
            }
            
            # Persist to MongoDB
            await self.mongo_storage.save_chat_message(
                user_id=user_id,
                session_id=session_id,
                agent_id="conversation_state",
                new_message=message
            )
            
            # Optionally persist to vector storage
            if self.vector_storage:
                await self.vector_storage.save_chat_message(
                    user_id=user_id,
                    session_id=session_id,
                    agent_id="conversation_state",
                    new_message=message
                )
            
            return True
            
        except Exception as e:
            Logger.error(f"Failed to track conversation state: {str(e)}")
            return False

    async def get_system_snapshot(self) -> SystemState:
        """Get current system state snapshot"""
        self.system_state.timestamp = datetime.now(timezone.utc)
        return self.system_state

    async def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get state for specific agent"""
        return self.system_state.agent_states.get(agent_id)

    async def restore_state_from_storage(self) -> bool:
        """Restore system state from persistent storage"""
        try:
            # Restore agent states
            agent_states = await self.mongo_storage.fetch_chat(
                user_id="system",
                session_id="agent_states",
                agent_id="*"
            )
            
            for message in agent_states:
                try:
                    state_dict = eval(message.content)  # Safe since we control the content
                    agent_id = state_dict.get('agent_id')
                    if agent_id:
                        self.system_state.agent_states[agent_id] = state_dict
                except:
                    continue
            
            return True
        except Exception as e:
            Logger.error(f"Failed to restore system state: {str(e)}")
            return False

    async def _run_periodic_cleanup(self):
        """Run periodic cleanup of old states"""
        while True:
            try:
                await self.cleanup_old_states(
                    max_age_hours=self.config.state_manager.max_state_age_hours
                )
                await asyncio.sleep(
                    self.config.state_manager.state_cleanup_interval_hours * 3600
                )
            except Exception as e:
                Logger.error(f"Error in periodic cleanup: {str(e)}")
                await asyncio.sleep(3600)  # Wait an hour before retrying

    async def cleanup_old_states(self, max_age_hours: int = 24) -> bool:
        """Clean up old state entries"""
        try:
            cutoff_time = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)
            
            # Clean up in-memory state
            for conversation_key, state in list(self.system_state.conversation_states.items()):
                if state['timestamp'].timestamp() < cutoff_time:
                    del self.system_state.conversation_states[conversation_key]
            
            return True
        except Exception as e:
            Logger.error(f"Failed to cleanup old states: {str(e)}")
            return False