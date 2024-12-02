import pytest
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from MAX.types import (
    ConversationMessage,
    ParticipantRole,
    OrchestratorConfig,
    TimestampedMessage
)
from MAX.storage.chat_storage import ChatStorage
from MAX.storage.in_memory_chat_storage import InMemoryChatStorage
from MAX.utils import Logger
from MAX.config.database_config import DatabaseConfig
from MAX.managers.state_manager import StateManager

# Mock Classes
class MockChatStorage(ChatStorage):
    async def save_chat_message(
        self, 
        user_id: str, 
        session_id: str, 
        agent_id: str, 
        new_message: ConversationMessage, 
        max_history_size: int = None
    ) -> bool:
        return True

    async def fetch_chat(
        self, 
        user_id: str, 
        session_id: str, 
        agent_id: str, 
        max_history_size: int = None
    ) -> List[ConversationMessage]:
        return []

    async def fetch_all_chats(
        self, 
        user_id: str, 
        session_id: str
    ) -> List[ConversationMessage]:
        return []

    async def fetch_chat_with_timestamps(
        self,
        user_id: str,
        session_id: str,
        agent_id: str
    ) -> List[TimestampedMessage]:
        return []

class MockMongoDBStorage:
    def __init__(self):
        self.storage = {}
        self.save_chat_message = AsyncMock()
        self.fetch_chat = AsyncMock(return_value=[])
        self.fetch_all_chats = AsyncMock(return_value=[])
        self.fetch_chat_with_timestamps = AsyncMock(return_value=[])

class MockChromaDBStorage:
    def __init__(self):
        self.storage = {}
        self.save_chat_message = AsyncMock()
        self.fetch_chat = AsyncMock(return_value=[])

# Fixtures
@pytest.fixture
def mock_logger():
    with patch('multi_agent_orchestrator.utils.logger') as mock:
        yield mock

@pytest.fixture
def chat_storage():
    return MockChatStorage()

@pytest.fixture
def in_memory_storage(mock_logger):
    return InMemoryChatStorage()

@pytest.fixture
def mock_db_config():
    config = DatabaseConfig()
    config.mongodb.uri = "mongodb://localhost:27017"
    config.mongodb.database = "test_db"
    config.mongodb.state_collection = "test_states"
    config.state_manager.enable_vector_storage = True
    config.state_manager.max_state_age_hours = 24
    return config

@pytest.fixture
async def state_manager(mock_db_config):
    mongo_storage = MockMongoDBStorage()
    chroma_storage = MockChromaDBStorage()
    
    manager = StateManager(mock_db_config)
    manager.mongo_storage = mongo_storage
    manager.vector_storage = chroma_storage
    
    return manager

# Base Storage Tests
def test_is_consecutive_message(chat_storage):
    conversation = [
        ConversationMessage(role="user", content="Hello"),
        ConversationMessage(role="assistant", content="Hi there"),
    ]

    # Test consecutive message
    new_message = ConversationMessage(role="assistant", content="How can I help you?")
    assert chat_storage.is_consecutive_message(conversation, new_message) == True

    # Test non-consecutive message
    new_message = ConversationMessage(role="user", content="I have a question")
    assert chat_storage.is_consecutive_message(conversation, new_message) == False

    # Test empty conversation
    assert chat_storage.is_consecutive_message([], new_message) == False

def test_trim_conversation(chat_storage):
    conversation = [
        TimestampedMessage(role="user", content="Message 1", timestamp=1),
        TimestampedMessage(role="assistant", content="Response 1", timestamp=2),
        TimestampedMessage(role="user", content="Message 2", timestamp=3),
        TimestampedMessage(role="assistant", content="Response 2", timestamp=4),
    ]

    # Test with even max_history_size
    trimmed = chat_storage.trim_conversation(conversation, max_history_size=2)
    assert len(trimmed) == 2
    assert trimmed[-1].content == "Response 2"

    # Test with odd max_history_size
    trimmed = chat_storage.trim_conversation(conversation, max_history_size=3)
    assert len(trimmed) == 2

    # Test with None max_history_size
    trimmed = chat_storage.trim_conversation(conversation, max_history_size=None)
    assert trimmed == conversation

# In-Memory Storage Tests
@pytest.mark.asyncio
async def test_save_chat_message(in_memory_storage):
    user_id = "user1"
    session_id = "session1"
    agent_id = "agent1"
    message = ConversationMessage(role="user", content="Hello")

    result = await in_memory_storage.save_chat_message(user_id, session_id, agent_id, message)
    assert len(result) == 1
    assert result[0].role == "user"
    assert result[0].content == "Hello"

@pytest.mark.asyncio
async def test_fetch_chat(in_memory_storage):
    user_id = "user1"
    session_id = "session1"
    agent_id = "agent1"
    message = ConversationMessage(role="user", content="Hello")

    await in_memory_storage.save_chat_message(user_id, session_id, agent_id, message)
    result = await in_memory_storage.fetch_chat(user_id, session_id, agent_id)

    assert len(result) == 1
    assert result[0].role == "user"
    assert result[0].content == "Hello"

# State Manager Tests
@pytest.mark.asyncio
async def test_update_agent_state(state_manager):
    agent_id = "test_agent"
    state = {
        "status": "active",
        "current_task": "processing_request",
        "memory_usage": 45.2
    }
    
    success = await state_manager.update_agent_state(agent_id, state)
    assert success == True
    
    # Verify state was stored in memory
    assert agent_id in state_manager.system_state.agent_states
    stored_state = state_manager.system_state.agent_states[agent_id]
    assert stored_state["status"] == "active"
    assert stored_state["current_task"] == "processing_request"

@pytest.mark.asyncio
async def test_track_conversation_state(state_manager):
    user_id = "test_user"
    session_id = "test_session"
    message = ConversationMessage(
        role=ParticipantRole.USER.value,
        content="Test message"
    )
    metadata = {"intent": "query", "confidence": 0.95}
    
    success = await state_manager.track_conversation_state(
        user_id=user_id,
        session_id=session_id,
        message=message,
        metadata=metadata
    )
    assert success == True
    
    # Verify conversation state was stored
    conversation_key = f"{user_id}:{session_id}"
    assert conversation_key in state_manager.system_state.conversation_states

@pytest.mark.asyncio
async def test_get_system_snapshot(state_manager):
    # Setup some test state
    await state_manager.update_agent_state("agent1", {"status": "active"})
    await state_manager.track_conversation_state(
        "user1", "session1",
        ConversationMessage(role=ParticipantRole.USER.value, content="test")
    )
    
    snapshot = await state_manager.get_system_snapshot()
    assert snapshot.agent_states["agent1"]["status"] == "active"
    assert len(snapshot.conversation_states) == 1
    assert isinstance(snapshot.timestamp, datetime)

@pytest.mark.asyncio
async def test_cleanup_old_states(state_manager):
    # Add some test states with old timestamps
    old_time = datetime.now(timezone.utc).timestamp() - (25 * 3600)  # 25 hours old
    state_manager.system_state.conversation_states["old_convo"] = {
        "timestamp": datetime.fromtimestamp(old_time, timezone.utc)
    }
    
    current_time = datetime.now(timezone.utc).timestamp()
    state_manager.system_state.conversation_states["current_convo"] = {
        "timestamp": datetime.fromtimestamp(current_time, timezone.utc)
    }
    
    success = await state_manager.cleanup_old_states(max_age_hours=24)
    assert success == True
    
    # Verify old state was removed while current state remains
    assert "old_convo" not in state_manager.system_state.conversation_states
    assert "current_convo" in state_manager.system_state.conversation_states