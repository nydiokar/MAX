import pytest
from datetime import datetime, timezone
from typing import List, Optional, Dict
from collections import defaultdict
from MAX.types import ConversationMessage, TimestampedMessage
from MAX.utils import Logger
from MAX.storage.abstract_storage.chat_storage import ChatStorage  # Changed from chat_storage to .chat_storage
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock
from MAX.types import (
    ConversationMessage,
    ParticipantRole,
    SystemState
)
from MAX.config.database_config import DatabaseConfig

class MockStorage:
    """Combined mock for both MongoDB and ChromaDB storage"""
    def __init__(self):
        self.storage = {}
        self.save_chat_message = AsyncMock(return_value=True)
        self.fetch_chat = AsyncMock(return_value=[])
        self.fetch_all_chats = AsyncMock(return_value=[])
        self.fetch_chat_with_timestamps = AsyncMock(return_value=[])

@pytest.fixture
def mock_db_config():
    return DatabaseConfig()

@pytest.fixture
async def state_manager(mock_db_config):
    from Orch.python.src.MAX.managers.system_state_manager import StateManager
    
    # Create state manager with mock storage
    manager = StateManager(mock_db_config)
    
    # Replace storage with mocks
    mock_storage = MockStorage()
    manager.mongo_storage = mock_storage
    manager.vector_storage = mock_storage
    
    return manager

@pytest.mark.asyncio
async def test_update_agent_state(state_manager):
    """Test agent state updates"""
    agent_id = "test_agent"
    state = {
        "status": "active",
        "current_task": "processing_request",
        "memory_usage": 45.2
    }
    
    success = await state_manager.update_agent_state(agent_id, state)
    assert success is True
    
    # Verify state was stored
    stored_state = state_manager.system_state.agent_states.get(agent_id)
    assert stored_state is not None
    assert stored_state["status"] == "active"
    assert stored_state["current_task"] == "processing_request"
    assert "last_updated" in stored_state

@pytest.mark.asyncio
async def test_track_conversation_state(state_manager):
    """Test conversation state tracking"""
    user_id = "test_user"
    session_id = "test_session"
    message = ConversationMessage(
        role=ParticipantRole.USER,
        content="Test message",
        timestamp=datetime.now(timezone.utc)
    )
    
    success = await state_manager.track_conversation_state(
        user_id=user_id,
        session_id=session_id,
        message=message
    )
    assert success is True
    
    # Verify state was tracked
    conv_key = f"{user_id}:{session_id}"
    assert conv_key in state_manager.system_state.conversation_states

@pytest.mark.asyncio
async def test_system_snapshot(state_manager):
    """Test system state snapshot"""
    # Setup initial state
    await state_manager.update_agent_state("agent1", {"status": "active"})
    
    snapshot = await state_manager.get_system_snapshot()
    assert isinstance(snapshot, SystemState)
    assert "agent1" in snapshot.agent_states
    assert snapshot.agent_states["agent1"]["status"] == "active"
    assert isinstance(snapshot.timestamp, datetime)

@pytest.mark.asyncio
async def test_state_cleanup(state_manager):
    """Test cleanup of old states"""
    # Add old state
    old_time = datetime.now(timezone.utc).timestamp() - (25 * 3600)
    state_manager.system_state.conversation_states["old_conv"] = {
        "timestamp": datetime.fromtimestamp(old_time, timezone.utc)
    }
    
    # Add current state
    state_manager.system_state.conversation_states["current_conv"] = {
        "timestamp": datetime.now(timezone.utc)
    }
    
    await state_manager.cleanup_old_states(max_age_hours=24)
    
    # Verify cleanup
    assert "old_conv" not in state_manager.system_state.conversation_states
    assert "current_conv" in state_manager.system_state.conversation_states

@pytest.mark.asyncio
async def test_error_handling(state_manager):
    """Test error handling"""
    # Simulate storage error
    state_manager.mongo_storage.save_chat_message.side_effect = Exception("Storage error")
    
    success = await state_manager.update_agent_state("agent1", {"status": "active"})
    assert success is False

if __name__ == "__main__":
    pytest.main(["-v"]) 