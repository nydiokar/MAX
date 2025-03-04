import pytest
from MAX.orchestrator import MultiAgentOrchestrator
from MAX.config.orchestrator_config import OrchestratorConfig
from MAX.agents.default_agent import DefaultAgent
from MAX.types import ConversationMessage, ParticipantRole

@pytest.fixture
async def live_orchestrator():
    """Create a real orchestrator with actual DefaultAgent."""
    config = OrchestratorConfig(
        LOG_EXECUTION_TIMES=True,
        USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=True,
        MEMORY_ENABLED=True
    )
    orchestrator = MultiAgentOrchestrator(options=config)
    
    # Add a real DefaultAgent
    default_agent = DefaultAgent()
    orchestrator.add_agent(default_agent)
    
    yield orchestrator
    await orchestrator.shutdown()

@pytest.mark.asyncio
async def test_basic_conversation(live_orchestrator):
    """Test a basic conversation with real agent responses."""
    response = await live_orchestrator.route_request(
        user_input="What is 2+2?",
        user_id="test_user",
        session_id="test_session"
    )
    
    assert response is not None
    assert response.output is not None
    assert isinstance(response.output, ConversationMessage)
    assert "4" in response.output.content[0]["text"].lower()

@pytest.mark.asyncio
async def test_conversation_memory(live_orchestrator):
    """Test that the agent remembers previous context."""
    # First message
    await live_orchestrator.route_request(
        user_input="My name is John",
        user_id="test_user",
        session_id="test_session"
    )
    
    # Follow-up message
    response = await live_orchestrator.route_request(
        user_input="What's my name?",
        user_id="test_user",
        session_id="test_session"
    )
    
    assert "John" in response.output.content[0]["text"]

@pytest.mark.asyncio
async def test_complex_query(live_orchestrator):
    """Test handling of more complex queries."""
    response = await live_orchestrator.route_request(
        user_input="Explain what recursion is in programming",
        user_id="test_user",
        session_id="test_session"
    )
    
    assert response is not None
    assert len(response.output.content[0]["text"]) > 100  # Should be a detailed response
    assert "recursion" in response.output.content[0]["text"].lower()

@pytest.mark.asyncio
async def test_error_recovery(live_orchestrator):
    """Test how the system handles and recovers from errors."""
    response = await live_orchestrator.route_request(
        user_input="",  # Empty input should be handled gracefully
        user_id="test_user",
        session_id="test_session"
    )
    
    assert response is not None
    assert response.output is not None
    assert "error" in response.metadata or "warning" in response.metadata

@pytest.mark.asyncio
async def test_performance_tracking(live_orchestrator):
    """Test that performance metrics are being tracked."""
    await live_orchestrator.route_request(
        user_input="Hello, how are you?",
        user_id="test_user",
        session_id="test_session"
    )
    
    assert len(live_orchestrator.execution_times) > 0
    assert any("execution" in key.lower() for key in live_orchestrator.execution_times.keys())

@pytest.mark.asyncio
async def test_multiple_turns(live_orchestrator):
    """Test multiple turn conversation."""
    responses = []
    
    # First turn
    responses.append(await live_orchestrator.route_request(
        user_input="What is Python?",
        user_id="test_user",
        session_id="test_session"
    ))
    
    # Second turn
    responses.append(await live_orchestrator.route_request(
        user_input="What are its main features?",
        user_id="test_user",
        session_id="test_session"
    ))
    
    # Third turn
    responses.append(await live_orchestrator.route_request(
        user_input="Give me an example of Python code",
        user_id="test_user",
        session_id="test_session"
    ))
    
    assert all(response is not None for response in responses)
    assert all(len(response.output.content[0]["text"]) > 0 for response in responses)
    # The last response should contain code
    assert "```" in responses[-1].output.content[0]["text"] 