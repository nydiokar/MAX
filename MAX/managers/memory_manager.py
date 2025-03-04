from typing import List, Optional, Dict, Any
from datetime import datetime
from mem0 import Memory  # Using the simpler Memory class
from MAX.types import ConversationMessage
from MAX.utils.logger import Logger

logger = Logger.get_logger()

class MemoryManager:
    """Manages memory operations for agents using mem0"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {
                "vector_store": {
                    "provider": "chroma",
                    "config": {
                        "collection_name": "agent_memory",
                        "path": "./data/chroma"
                    }
                }
            }
        self.memory = Memory(config=config)
        self.logger = logger

    async def store_message(
        self,
        message: ConversationMessage,
        agent_id: str,
        session_id: str,
        metadata: Optional[dict] = None
    ) -> bool:
        """Store a message in memory"""
        try:
            self.memory.add(
                message.content[0].get('text', ''),
                user_id=session_id,
                metadata={
                    "agent_id": agent_id,
                    "role": message.role,
                    "timestamp": datetime.now().isoformat(),
                    **(metadata or {})
                }
            )
            self.logger.info(f"Stored message for agent {agent_id} in session {session_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to store message: {str(e)}")
            return False

    async def get_relevant_context(
        self,
        query: str,
        agent_id: str,
        session_id: str,
        limit: int = 5
    ) -> List[ConversationMessage]:
        """Get relevant context for a query"""
        try:
            memories = self.memory.search(
                query=query,
                user_id=session_id,
                metadata={"agent_id": agent_id}
            )
            
            return [
                ConversationMessage(
                    role=memory["metadata"]["role"],
                    content=[{"text": memory["content"]}],
                    timestamp=memory["metadata"].get("timestamp")
                )
                for memory in memories[:limit]
            ]
        except Exception as e:
            self.logger.error(f"Failed to retrieve context: {str(e)}")
            return []

    async def get_conversation_history(
        self,
        agent_id: str,
        session_id: str,
        limit: int = 10
    ) -> List[ConversationMessage]:
        """Get conversation history for an agent"""
        try:
            memories = self.memory.get_all(
                user_id=session_id,
                metadata={"agent_id": agent_id}
            )
            
            return [
                ConversationMessage(
                    role=memory["metadata"]["role"],
                    content=[{"text": memory["content"]}],
                    timestamp=memory["metadata"].get("timestamp")
                )
                for memory in memories[:limit]
            ]
        except Exception as e:
            self.logger.error(f"Failed to get conversation history: {str(e)}")
            return []

    async def delete_session_memory(
        self,
        agent_id: str,
        session_id: str
    ) -> bool:
        """Delete all memories for a specific session"""
        try:
            self.memory.delete_all(user_id=session_id)
            self.logger.info(f"Deleted memory for agent {agent_id} in session {session_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete session memory: {str(e)}")
            return False