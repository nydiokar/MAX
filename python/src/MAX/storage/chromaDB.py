import chromadb
from typing import List, Optional
from MAX.types import ConversationMessage, TimestampedMessage
from MAX.storage.chat_storage import ChatStorage
from MAX.utils import Logger
import time

class ChromaDBChatStorage(ChatStorage):
    """
    ChromaDB implementation of ChatStorage for semantic search and vector storage.
    """

    def __init__(self, collection_name: str):
        self.client = chromadb.Client()
        try:
            # Try to get existing collection or create new one
            self.collection = self.client.get_or_create_collection(name=collection_name)
        except Exception as e:
            Logger.error(f"Failed to initialize ChromaDB collection: {str(e)}")
            raise e

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
                metadatas=[{
                    "user_id": user_id,
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "role": new_message.role,
                }],
                ids=[f"{key}_{time.time()}"]
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
                    "agent_id": agent_id
                },
                n_results=max_history_size or 100
            )
            
            messages = []
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                messages.append(ConversationMessage(
                    role=metadata["role"],
                    content=doc
                ))
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
                    role=msg.role,
                    content=msg.content,
                    timestamp=time.time()
                ) for msg in messages
            ]
        except Exception as e:
            Logger.error(f"Failed to fetch timestamped messages from ChromaDB: {str(e)}")
            return []

    async def fetch_all_chats(
        self,
        user_id: str,
        session_id: str
    ) -> List[ConversationMessage]:
        """Implement the required abstract method"""
        try:
            results = self.collection.query(
                query_texts=[""],
                where={
                    "user_id": user_id,
                    "session_id": session_id
                },
                n_results=1000  # Arbitrary large number
            )
            
            messages = []
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                messages.append(ConversationMessage(
                    role=metadata["role"],
                    content=doc
                ))
            return messages
        except Exception as e:
            Logger.error(f"Failed to fetch all chats from ChromaDB: {str(e)}")
            return []