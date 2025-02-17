import pytest
import asyncio
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock

class MockStorage:
    """Mock storage for testing."""
    def __init__(self):
        self.data = {}

    async def fetch_chat(self, *args):
        return []

    async def save_chat_messages(self, *args):
        pass

    async def fetch_all_chats(self, *args):
        return []

class MockLLM:
    """Mock LLM for testing."""
    async def generate(self, *args, **kwargs):
        return "Mock LLM response"

class MockTransport:
    """Mock transport for testing."""
    async def send_message(self, *args, **kwargs):
        return True

    async def receive_message(self, *args, **kwargs):
        return {"text": "Mock received message"}

@pytest.fixture
def mock_storage():
    return MockStorage()

@pytest.fixture
def mock_llm():
    return MockLLM()

@pytest.fixture
def mock_transport():
    return MockTransport()

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_async_response():
    """Create a mock async response."""
    mock_response = AsyncMock()
    mock_response.content = [{"text": "Mock response content"}]
    return mock_response

@pytest.fixture
def mock_conversation_context():
    """Create a mock conversation context."""
    return {
        "user_id": "test_user",
        "session_id": "test_session",
        "channel": "test",
        "metadata": {}
    }
