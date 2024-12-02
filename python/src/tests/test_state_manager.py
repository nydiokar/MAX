import pytest
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any
from unittest.mock import AsyncMock

from MAX.types import (
    ConversationMessage,
    ParticipantRole,
)
from MAX.config.database_config import (
    DatabaseConfig,
    MongoDBConfig,
    ChromaDBConfig,
    StateManagerConfig
)
from MAX.managers.state_manager import StateManager

class MockStorage:
    def __init__(self):
        self.storage = {}
        self.save_chat_message = AsyncMock()
        self.fetch_chat = AsyncMock(return_value=[])
        self.fetch_all_chats = AsyncMock(return_value=[])

@pytest.fixture
def mock_db_config():
    config = DatabaseConfig()
    config.mongodb = MongoDBConfig(
        uri="mongodb://localhost:27017",
        database="test_db",
        state_collection="test_states",
        ttl_hours=24
    )
    config.chromadb = ChromaDBConfig(
        collection_name="test_collection"
    )
    config.state_manager = StateManagerConfig(
        enable_vector_storage=True,
        max_state_age_hours=24
    )
    return config

@pytest.fixture
async def state_manager(mock_db_config):
    mongo_storage = MockStorage()
    chroma_storage = MockStorage()
    
    manager = StateManager(mock_db_config)
    # Override the storage instances with our mocks
    manager.mongo_storage = mongo_storage
    manager.vector_storage = chroma_storage
    
    return manager

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
    
    conversation_key = f"{user_id}:{session_id}"
    assert conversation_key in state_manager.system_state.conversation_states

@pytest.mark.asyncio
async def test_cleanup_old_states(state_manager):
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
    assert "old_convo" not in state_manager.system_state.conversation_states
    assert "current_convo" in state_manager.system_state.conversation_states