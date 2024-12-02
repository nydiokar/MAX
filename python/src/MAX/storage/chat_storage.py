from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from MAX.types import ConversationMessage, TimestampedMessage


class ChatStorage(ABC):
    """
    Abstract base class for chat storage backends.
    Provides shared logic and defines the interface for specific implementations.
    """

    def is_consecutive_message(
        self, conversation: List[ConversationMessage], new_message: ConversationMessage
    ) -> bool:
        """
        Check if the new message is consecutive with the last message in the conversation.

        Args:
            conversation (List[ConversationMessage]): The existing conversation.
            new_message (ConversationMessage): The new message to check.

        Returns:
            bool: True if the new message is consecutive, False otherwise.
        """
        if not conversation:
            return False
        return conversation[-1].role == new_message.role

    def trim_conversation(
        self, conversation: List[TimestampedMessage], max_history_size: Optional[int] = None
    ) -> List[TimestampedMessage]:
        """
        Trim the conversation to the specified maximum history size.

        Args:
            conversation (List[TimestampedMessage]): The conversation to trim.
            max_history_size (Optional[int]): The maximum number of messages to keep.

        Returns:
            List[TimestampedMessage]: The trimmed conversation.
        """
        if max_history_size is None:
            return conversation

        # Ensure max_history_size is even to maintain paired messages
        max_size = max_history_size if max_history_size % 2 == 0 else max_history_size - 1
        return conversation[-max_size:]

    def _generate_key(self, user_id: str, session_id: str, agent_id: str) -> str:
        """
        Generate a unique key for a user, session, and agent combination.

        Args:
            user_id (str): The user ID.
            session_id (str): The session ID.
            agent_id (str): The agent ID.

        Returns:
            str: A unique key combining user ID, session ID, and agent ID.
        """
        return f"{user_id}#{session_id}#{agent_id}"

    def _remove_timestamps(
        self, messages: List[TimestampedMessage]
    ) -> List[ConversationMessage]:
        """
        Remove timestamps from messages to return plain conversation data.

        Args:
            messages (List[TimestampedMessage]): Messages with timestamps.

        Returns:
            List[ConversationMessage]: Messages without timestamps.
        """
        return [ConversationMessage(role=msg.role, content=msg.content) for msg in messages]

    def _convert_to_timestamps(
        self, conversation: List[ConversationMessage]
    ) -> List[TimestampedMessage]:
        """
        Convert a plain conversation to timestamped messages.

        Args:
            conversation (List[ConversationMessage]): Plain conversation messages.

        Returns:
            List[TimestampedMessage]: Messages with added timestamps.
        """
        return [
            TimestampedMessage(role=msg.role, content=msg.content, timestamp=0)
            for msg in conversation
        ]

    @abstractmethod
    async def save_chat_message(
        self,
        user_id: str,
        session_id: str,
        agent_id: str,
        new_message: ConversationMessage,
        max_history_size: Optional[int] = None,
    ) -> bool:
        """
        Save a new chat message.

        Args:
            user_id (str): The user ID.
            session_id (str): The session ID.
            agent_id (str): The agent ID.
            new_message (ConversationMessage): The new message to save.
            max_history_size (Optional[int]): The maximum history size.

        Returns:
            bool: True if the message was saved successfully, False otherwise.
        """
        pass

    @abstractmethod
    async def fetch_chat(
        self,
        user_id: str,
        session_id: str,
        agent_id: str,
        max_history_size: Optional[int] = None,
    ) -> List[ConversationMessage]:
        """
        Fetch chat messages for a user, session, and agent.

        Args:
            user_id (str): The user ID.
            session_id (str): The session ID.
            agent_id (str): The agent ID.
            max_history_size (Optional[int]): The maximum number of messages to fetch.

        Returns:
            List[ConversationMessage]: The fetched chat messages.
        """
        pass

    @abstractmethod
    async def fetch_all_chats(
        self, user_id: str, session_id: str
    ) -> List[ConversationMessage]:
        """
        Fetch all chat messages for a user and session.

        Args:
            user_id (str): The user ID.
            session_id (str): The session ID.

        Returns:
            List[ConversationMessage]: All chat messages for the user and session.
        """
        pass

    @abstractmethod
    async def fetch_chat_with_timestamps(
        self, user_id: str, session_id: str, agent_id: str
    ) -> List[TimestampedMessage]:
        """
        Fetch chat messages with timestamps.

        Args:
            user_id (str): The user ID.
            session_id (str): The session ID.
            agent_id (str): The agent ID.

        Returns:
            List[TimestampedMessage]: The fetched chat messages with timestamps.
        """
        pass
