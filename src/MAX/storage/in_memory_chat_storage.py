from typing import List, Optional, Dict
from collections import defaultdict
from MAX.types import ConversationMessage, TimestampedMessage
from MAX.utils import Logger
from .chat_storage import ChatStorage


class InMemoryChatStorage(ChatStorage):
    """
    In-memory implementation of ChatStorage for development and testing.
    Stores chat data temporarily during runtime.
    """

    def __init__(self):
        super().__init__()
        self.conversations: Dict[str, List[TimestampedMessage]] = defaultdict(list)

    async def save_chat_message(
        self,
        user_id: str,
        session_id: str,
        agent_id: str,
        new_message: ConversationMessage,
        max_history_size: Optional[int] = None,
    ) -> bool:
        """
        Save a new chat message to the in-memory storage.
        """
        key = self._generate_key(user_id, session_id, agent_id)
        conversation = self.conversations[key]

        # Check for consecutive message
        if self.is_consecutive_message(conversation, new_message):
            Logger.debug(
                f"Consecutive {new_message.role} message detected for agent {agent_id}. Not saving."
            )
            return False

        # Add a timestamped message
        timestamped_message = TimestampedMessage(
            role=new_message.role,
            content=new_message.content,
            timestamp=new_message.timestamp or 0  # Use 0 if no timestamp is provided
        )
        conversation.append(timestamped_message)

        # Trim the conversation to the max history size
        self.conversations[key] = self.trim_conversation(conversation, max_history_size)
        return True

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
        conversation = self.conversations[key]

        if max_history_size is not None:
            conversation = self.trim_conversation(conversation, max_history_size)

        return self._remove_timestamps(conversation)

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
        return self.conversations[key]

    async def fetch_all_chats(
        self, user_id: str, session_id: str
    ) -> List[ConversationMessage]:
        """
        Fetch all chat messages across all agents for a user and session.
        """
        all_messages = []
        for key, conversation in self.conversations.items():
            stored_user_id, stored_session_id, _ = key.split("#")
            if stored_user_id == user_id and stored_session_id == session_id:
                all_messages.extend(conversation)

        # Sort messages by timestamp
        all_messages.sort(key=lambda msg: msg.timestamp)
        return self._remove_timestamps(all_messages)
