import pytest
import logging
from pathlib import Path

from MAX.agents import RecursiveThinkerAgent, RecursiveThinkerOptions
from MAX.types import ConversationMessage
from MAX.config.models import AnthropicModels

# Configure logging to prevent sensitive data leaks
logging.getLogger('MAX').setLevel(logging.ERROR)

@pytest.fixture(autouse=True)
def setup_logging():
    """Configure secure logging"""
    logger = logging.getLogger('MAX')
    logger.handlers = []
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)

@pytest.fixture
def agent_options():
    return RecursiveThinkerOptions(
        name="RecursiveThinker",
        description="A recursive thinking agent",
        model_id=AnthropicModels.HAIKU,
        streaming=False
    )

@pytest.mark.asyncio
async def test_basic_request(agent_options):
    """Test basic request processing"""
    agent = RecursiveThinkerAgent(agent_options)
    response = await agent.process_request(
        input_text="Hello",
        user_id="test_user",
        session_id="test_session",
        chat_history=[]
    )
    assert isinstance(response, ConversationMessage)
    assert response.content[0]["text"]

@pytest.mark.asyncio
async def test_memory_storage(agent_options):
    """Test basic memory storage"""
    agent = RecursiveThinkerAgent(agent_options)
    await agent.process_request(
        input_text="Remember this",
        user_id="test_user",
        session_id="test_session",
        chat_history=[]
    )
    assert "test_session" in agent.conversation_memories
    assert len(agent.conversation_memories["test_session"]) > 0

@pytest.mark.asyncio
async def test_context_usage(agent_options):
    """Test conversation context is used"""
    agent = RecursiveThinkerAgent(agent_options)
    await agent.process_request(
        input_text="My name is Alice",
        user_id="test_user",
        session_id="test_session",
        chat_history=[]
    )
    response = await agent.process_request(
        input_text="What's my name?",
        user_id="test_user",
        session_id="test_session",
        chat_history=[]
    )
    assert "Alice" in response.content[0]["text"]