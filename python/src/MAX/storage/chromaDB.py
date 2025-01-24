import chromadb
from typing import List, Optional, Dict, Any, Tuple
from MAX.types import ConversationMessage, TimestampedMessage
from MAX.storage.abstract_storage.chat_storage import ChatStorage
from MAX.utils import Logger
import time
import asyncio
from datetime import datetime, timezone
from contextlib import asynccontextmanager


class ChromaDBChatStorage(ChatStorage):
    """
    ChromaDB implementation of ChatStorage for semantic search and vector storage.
    """

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialized = False
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def connection(self):
        """Context manager for ensuring connection"""
        if not self._initialized:
            await self.initialize()
        try:
            yield self
        except Exception as e:
            Logger.error(f"ChromaDB operation error: {str(e)}")
            raise

    async def initialize(self) -> bool:
        """Initialize ChromaDB connection"""
        if self._initialized:
            return True

        async with self._lock:
            if self._initialized:
                return True

            try:
                self.client = chromadb.Client()
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name
                )
                self._initialized = True
                return True
            except Exception as e:
                Logger.error(f"Failed to initialize ChromaDB: {str(e)}")
                return False

    async def cleanup(self) -> None:
        """Cleanup ChromaDB resources"""
        if self.client:
            try:
                # ChromaDB might not have explicit cleanup
                self.client = None
                self.collection = None
                self._initialized = False
            except Exception as e:
                Logger.error(f"Error during ChromaDB cleanup: {str(e)}")

    async def check_health(self) -> Tuple[bool, Dict[str, Any]]:
        """Check ChromaDB health"""
        try:
            async with self.connection():
                # Try to perform a basic query
                results = self.collection.query(
                    query_texts=["health_check"], n_results=1
                )

                return True, {
                    "status": "healthy",
                    "collection_name": self.collection_name,
                    "count": len(results["ids"][0]) if results["ids"] else 0,
                    "timestamp": datetime.now(timezone.utc),
                }
        except Exception as e:
            return False, {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc),
            }

    async def save_chat_message(
        self,
        user_id: str,
        session_id: str,
        agent_id: str,
        new_message: ConversationMessage,
        max_history_size: Optional[int] = None,
    ) -> bool:
        try:
            key = self._generate_key(user_id, session_id, agent_id)
            self.collection.add(
                documents=[new_message.content],
                metadatas=[
                    {
                        "user_id": user_id,
                        "session_id": session_id,
                        "agent_id": agent_id,
                        "role": new_message.role,
                    }
                ],
                ids=[f"{key}_{time.time()}"],
            )
            return True
        except Exception as e:
            Logger.error(f"Failed to save to ChromaDB: {str(e)}")
            return False

    async def fetch_chat(
        self,
        user_id: str,
        session_id: str,
        agent_id: str,
        max_history_size: Optional[int] = None,
    ) -> List[ConversationMessage]:
        try:
            key = self._generate_key(user_id, session_id, agent_id)
            results = self.collection.query(
                query_texts=[""],
                where={
                    "user_id": user_id,
                    "session_id": session_id,
                    "agent_id": agent_id,
                },
                n_results=max_history_size or 100,
            )

            messages = []
            for doc, metadata in zip(
                results["documents"][0], results["metadatas"][0]
            ):
                messages.append(
                    ConversationMessage(role=metadata["role"], content=doc)
                )
            return messages
        except Exception as e:
            Logger.error(f"Failed to fetch from ChromaDB: {str(e)}")
            return []

    async def fetch_chat_with_timestamps(
        self,
        user_id: str,
        session_id: str,
        agent_id: str,
    ) -> List[TimestampedMessage]:
        """Implement the required abstract method"""
        try:
            messages = await self.fetch_chat(user_id, session_id, agent_id)
            # Convert to timestamped messages
            return [
                TimestampedMessage(
                    role=msg.role, content=msg.content, timestamp=time.time()
                )
                for msg in messages
            ]
        except Exception as e:
            Logger.error(
                f"Failed to fetch timestamped messages from ChromaDB: {str(e)}"
            )
            return []

    async def fetch_all_chats(
        self, user_id: str, session_id: str
    ) -> List[ConversationMessage]:
        """Implement the required abstract method"""
        try:
            results = self.collection.query(
                query_texts=[""],
                where={"user_id": user_id, "session_id": session_id},
                n_results=1000,  # Arbitrary large number
            )

            messages = []
            for doc, metadata in zip(
                results["documents"][0], results["metadatas"][0]
            ):
                messages.append(
                    ConversationMessage(role=metadata["role"], content=doc)
                )
            return messages
        except Exception as e:
            Logger.error(f"Failed to fetch all chats from ChromaDB: {str(e)}")
            return []

    async def save_task_state(self, task_state: Dict[str, Any]) -> bool:
        """
        Task states should be handled by MongoDB or another document store.
        ChromaDB is optimized for vector similarity search, not state management.
        """
        return True

    async def save_system_state(
        self,
        state_type: str,
        state_data: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """ChromaDB doesn't handle system states"""
        Logger.warn(
            f"Attempted to save system state '{state_type}' in ChromaDB which doesn't support state management"
        )
        return True

    async def search_similar_messages(
        self,
        query: str,
        metadata_filter: Optional[Dict[str, Any]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search for similar messages using semantic similarity"""
        try:
            results = self.collection.query(
                query_texts=[query], where=metadata_filter, n_results=limit
            )

            return [
                {
                    "content": doc,
                    "metadata": meta,
                    "distance": dist,  # ChromaDB provides actual semantic distance
                }
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ]
        except Exception as e:
            Logger.error(f"Failed to search ChromaDB: {str(e)}")
            return []
