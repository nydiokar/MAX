from typing import List, Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient
from MAX.types import ConversationMessage, TimestampedMessage
from MAX.storage.chat_storage import ChatStorage 
from MAX.utils import Logger
from datetime import datetime, timezone
import asyncio


class MongoDBChatStorage(ChatStorage):
    """
    MongoDB implementation of ChatStorage for scalable, persistent storage.
    """

    def __init__(self, mongo_uri: str, db_name: str, collection_name: str, ttl_index: Optional[int] = None):
        self.client = AsyncIOMotorClient(mongo_uri)
        self.db = self.client[db_name]
        
        # Initialize collections
        self.conversations = self.db[collection_name]
        self.system_states = self.db['system_states']
        self.task_states = self.db['task_states']
        
        # Setup indices asynchronously
        asyncio.create_task(self._setup_indices(ttl_index))

    async def _setup_indices(self, ttl_index: Optional[int] = None):
        """Setup necessary indices for all collections"""
        try:
            await self.conversations.create_index([("user_id", 1), ("session_id", 1)])
            await self.system_states.create_index([("type", 1), ("timestamp", 1)])
            await self.task_states.create_index([("agent_id", 1), ("status", 1)])
            
            if ttl_index:
                await self.conversations.create_index(
                    "timestamp", 
                    expireAfterSeconds=ttl_index
                )
        except Exception as e:
            Logger.error(f"Failed to setup MongoDB indices: {str(e)}")

    async def save_chat_message(
        self,
        user_id: str,
        session_id: str,
        agent_id: str,
        new_message: ConversationMessage,
        max_history_size: Optional[int] = None,
    ) -> bool:
        """Save a new chat message to MongoDB."""
        try:
            key = self._generate_key(user_id, session_id, agent_id)
            
            # Convert message to dict for storage
            message_dict = {
                "role": str(new_message.role),
                "content": new_message.content,
                "timestamp": new_message.timestamp
            }

            # Fetch existing conversation
            existing_conversation = await self.fetch_chat_with_timestamps(user_id, session_id, agent_id)

            # Check for consecutive message
            if self.is_consecutive_message(existing_conversation, new_message):
                Logger.debug(f"Consecutive {new_message.role} message detected for agent {agent_id}. Not saving.")
                return False

            # Add the new message and trim the conversation
            existing_conversation.append(TimestampedMessage(**message_dict))
            trimmed_conversation = self.trim_conversation(existing_conversation, max_history_size)

            # Save to MongoDB
            await self.collection.replace_one(
                {"_id": key},
                {
                    "_id": key,
                    "messages": [msg.model_dump() for msg in trimmed_conversation]
                },
                upsert=True
            )
            return True
        except Exception as e:
            Logger.error(f"Failed to save chat message to MongoDB: {str(e)}")
            return False

    async def fetch_chat(
        self,
        user_id: str,
        session_id: str,
        agent_id: str,
        max_history_size: Optional[int] = None,
    ) -> List[ConversationMessage]:
        """
        Fetch chat messages without timestamps for a specific user, session, and agent.
        """
        key = self._generate_key(user_id, session_id, agent_id)
        try:
            document = await self.collection.find_one({"_id": key})
            if not document:
                return []

            messages = [
                TimestampedMessage(**msg) for msg in document.get("messages", [])
            ]

            # Trim conversation and remove timestamps
            if max_history_size is not None:
                messages = self.trim_conversation(messages, max_history_size)
            return self._remove_timestamps(messages)
        except Exception as e:
            Logger.error(f"Failed to fetch chat from MongoDB: {str(e)}")
            raise e

    async def fetch_chat_with_timestamps(
        self,
        user_id: str,
        session_id: str,
        agent_id: str,
    ) -> List[TimestampedMessage]:
        """
        Fetch chat messages with timestamps for debugging or persistence.
        """
        key = self._generate_key(user_id, session_id, agent_id)
        try:
            document = await self.collection.find_one({"_id": key})
            if not document:
                return []
            return [TimestampedMessage(**msg) for msg in document.get("messages", [])]
        except Exception as e:
            Logger.error(f"Failed to fetch chat with timestamps from MongoDB: {str(e)}")
            raise e

    async def fetch_all_chats(
        self, user_id: str, session_id: str
    ) -> List[ConversationMessage]:
        """
        Fetch all chat messages for a user's session across agents.
        """
        try:
            cursor = self.collection.find(
                {"_id": {"$regex": f"^{user_id}#{session_id}#"}}
            )
            all_messages = []
            async for document in cursor:
                all_messages.extend(
                    [TimestampedMessage(**msg) for msg in document.get("messages", [])]
                )

            # Sort by timestamp and remove timestamps for return
            all_messages = self._sort_conversation(all_messages)
            return self._remove_timestamps(all_messages)
        except Exception as e:
            Logger.error(f"Failed to fetch all chats from MongoDB: {str(e)}")
            raise e

    async def save_system_state(self, state: Dict[str, Any]) -> bool:
        """Save system state snapshot"""
        try:
            await self.collection.update_one(
                {"_id": "system_state"},
                {"$set": {
                    "state": state,
                    "timestamp": datetime.now(timezone.utc)
                }},
                upsert=True
            )
            return True
        except Exception as e:
            Logger.error(f"Failed to save system state: {str(e)}")
            return False

    async def get_system_state(self) -> Optional[Dict[str, Any]]:
        """Retrieve latest system state"""
        try:
            doc = await self.collection.find_one({"_id": "system_state"})
            return doc.get("state") if doc else None
        except Exception as e:
            Logger.error(f"Failed to get system state: {str(e)}")
            return None

    async def search_similar_messages(
        self,
        query: str,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        try:
            filter_query = {
                "$text": {"$search": query}
            }
            if metadata_filter:
                filter_query.update(metadata_filter)
            
            cursor = self.conversations.find(
                filter_query,
                sort=[("score", {"$meta": "textScore"})]
            )
            
            results = []
            async for doc in cursor:
                results.append({
                    "content": doc["messages"][-1]["content"],
                    "metadata": doc.get("metadata", {}),
                
                })
            return results
        except Exception as e:
            Logger.error(f"Failed to search MongoDB: {str(e)}")
            return []
