import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from MAX.orchestrator import MultiAgentOrchestrator
from MAX.config.orchestrator_config import OrchestratorConfig
from MAX.types import ConversationMessage, ParticipantRole
from MAX.types.workflow_types import WorkflowStage
from MAX.classifiers import ClassifierResult
from MAX.agents import Agent, AgentResponse
from MAX.storage import InMemoryChatStorage
from MAX.agents.default_agent import DefaultAgent

@pytest.fixture
def mock_classifier():
    mock = Mock()
    mock.classify = Mock()
    return mock

@pytest.fixture
def mock_storage():
    return InMemoryChatStorage()

@pytest.fixture
def mock_agent():
    mock = AsyncMock(spec=Agent)
    mock.name = "MOCK_AGENT"
    mock.id = "mock_agent"
    mock.description = "Mock agent for testing"
    mock.is_streaming_enabled.return_value = False
    mock.process_request.return_value = ConversationMessage(
        role=ParticipantRole.ASSISTANT,
        content=[{"text": "Mock response"}]
    )
    return mock

@pytest.fixture
async def orchestrator(mock_storage, mock_agent):
    """Create a test orchestrator with basic configuration."""
    config = OrchestratorConfig(
        LOG_EXECUTION_TIMES=True,
        USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=True,
        MEMORY_ENABLED=True
    )
    orchestrator = MultiAgentOrchestrator(
        options=config,
        storage=mock_storage
    )
    orchestrator.add_agent(mock_agent)
    yield orchestrator
    await orchestrator.shutdown()

@pytest.mark.asyncio
async def test_agent_registration(orchestrator, mock_agent):
    """Test basic agent registration functionality."""
    assert mock_agent.id in orchestrator.get_all_agents()
    
    # Test adding another agent
    new_agent = DefaultAgent()
    success = orchestrator.add_agent(new_agent)
    assert success
    assert new_agent.id in orchestrator.get_all_agents()

@pytest.mark.asyncio
async def test_message_routing(orchestrator, mock_agent):
    """Test basic message routing functionality."""
    response = await orchestrator.route_request(
        user_input="test message",
        user_id="test_user",
        session_id="test_session"
    )
    
    assert response is not None
    assert isinstance(response, AgentResponse)
    assert mock_agent.process_request.called
    assert "Mock response" in response.output.content[0]["text"]

@pytest.mark.asyncio
async def test_memory_storage(orchestrator):
    """Test basic memory storage functionality."""
    if not orchestrator.memory_manager:
        pytest.skip("Memory manager not enabled")
    
    test_message = ConversationMessage(
        role=ParticipantRole.USER,
        content=[{"text": "Test memory storage"}]
    )
    
    # Test storage
    await orchestrator.memory_manager.store_message(
        message=test_message,
        agent_id="test_agent",
        session_id="test_session"
    )
    
    # Test retrieval
    context = await orchestrator.memory_manager.get_relevant_context(
        query="Test memory",
        agent_id="test_agent",
        session_id="test_session"
    )
    
    assert context is not None
    assert "Test memory storage" in context

@pytest.mark.asyncio
async def test_error_handling(orchestrator, mock_agent):
    """Test error handling in message routing."""
    mock_agent.process_request.side_effect = Exception("Test error")
    
    response = await orchestrator.route_request(
        user_input="trigger error",
        user_id="test_user",
        session_id="test_session"
    )
    
    assert response is not None
    assert "error" in response.metadata
    assert response.metadata["error"] == "Test error"

@pytest.mark.asyncio
async def test_performance_metrics(orchestrator):
    """Test collection of performance metrics."""
    # Send a test request
    await orchestrator.route_request(
        user_input="Test performance metrics",
        user_id="test_user",
        session_id="test_session"
    )
    
    # Verify execution times are collected
    assert len(orchestrator.execution_times) > 0
    assert "Classifying user intent" in orchestrator.execution_times
    assert "Agent execution" in orchestrator.execution_times

def test_agent_availability(orchestrator):
    """Test agent availability checking."""
    test_agent = DefaultAgent()
    orchestrator.add_agent(test_agent)
    
    # Test availability check
    assert orchestrator._is_agent_available(test_agent)
    
    # Test with busy agent
    test_agent.current_tasks = ["task1", "task2"]
    test_agent.max_concurrent_tasks = 2
    assert not orchestrator._is_agent_available(test_agent)

@pytest.mark.asyncio
async def test_agent_selection_with_classifier(orchestrator, mock_classifier, mock_agent):
    """Test agent selection using intent classification"""
    # Setup classifier response
    mock_classifier.classify.return_value = ClassifierResult(
        selected_agent=mock_agent,
        confidence=0.9,
        intents=["test_intent"]
    )

    # Test selection
    result = await orchestrator.route_request(
        user_input="test request",
        user_id="test_user",
        session_id="test_session"
    )

    assert isinstance(result, AgentResponse)
    assert mock_classifier.classify.called
    assert result.metadata.agent_id == mock_agent.id

@pytest.mark.asyncio
async def test_workflow_stage_progression(orchestrator, mock_classifier, mock_agent):
    """Test Memory → Reasoning → Execution workflow progression"""
    # Setup successful agent selection
    mock_classifier.classify.return_value = ClassifierResult(
        selected_agent=mock_agent,
        confidence=0.9,
        intents=["test_intent"]
    )

    # Mock agent response
    mock_agent.process_request.return_value = ConversationMessage(
        role=ParticipantRole.ASSISTANT.value,
        content=[{"text": "test response"}]
    )

    # Test full workflow
    result = await orchestrator.route_request(
        user_input="test workflow request",
        user_id="test_user",
        session_id="test_session"
    )

    # Verify workflow stages were executed
    supervisor_calls = orchestrator.supervisor.activate_team.call_args_list
    assert len(supervisor_calls) >= 3  # Memory, Reasoning, Execution stages

    stages = [call[1]["workflow_stage"] for call in supervisor_calls]
    assert WorkflowStage.MEMORY.value in stages
    assert WorkflowStage.REASONING.value in stages
    assert WorkflowStage.EXECUTION.value in stages

@pytest.mark.asyncio
async def test_agent_fallback_mechanism(orchestrator, mock_classifier):
    """Test fallback mechanism when primary agent selection fails"""
    # Setup classifier response with no agent
    mock_classifier.classify.return_value = ClassifierResult(
        selected_agent=None,
        confidence=0,
        intents=["test_intent"],
        fallback_reason="No suitable agent found"
    )

    # Test fallback behavior
    result = await orchestrator.route_request(
        user_input="test request",
        user_id="test_user",
        session_id="test_session"
    )

    assert isinstance(result, AgentResponse)
    assert "NO_SELECTED_AGENT" in result.output.content[0]["text"]

@pytest.mark.asyncio
async def test_multi_agent_response_handling(orchestrator, mock_classifier):
    """Test handling of responses from multiple agents"""
    # Setup mock team response
    mock_responses = [
        ConversationMessage(
            role=ParticipantRole.ASSISTANT.value,
            content=[{"text": "Response 1"}]
        ),
        ConversationMessage(
            role=ParticipantRole.ASSISTANT.value,
            content=[{"text": "Response 2"}]
        )
    ]

    # Mock supervisor aggregation
    orchestrator.supervisor.aggregate_responses.return_value = {
        "content": "Aggregated response",
        "confidence": 0.9
    }

    # Test response handling
    result = await orchestrator.route_request(
        user_input="test multi-agent request",
        user_id="test_user",
        session_id="test_session"
    )

    assert isinstance(result, AgentResponse)
    assert orchestrator.supervisor.aggregate_responses.called

@pytest.mark.asyncio
async def test_error_handling_and_recovery(orchestrator, mock_classifier, mock_agent):
    """Test error handling during agent execution"""
    # Setup classifier
    mock_classifier.classify.return_value = ClassifierResult(
        selected_agent=mock_agent,
        confidence=0.9
    )

    # Make agent raise an error
    mock_agent.process_request.side_effect = Exception("Test error")

    # Test error handling
    result = await orchestrator.route_request(
        user_input="test error case",
        user_id="test_user",
        session_id="test_session"
    )

    assert isinstance(result, AgentResponse)
    assert result.metadata.additional_params.get("error_type") is not None
    assert orchestrator.supervisor.update_task_status.called

@pytest.mark.asyncio
async def test_conversation_state_management(orchestrator, mock_storage):
    """Test conversation state management across requests"""
    # Test state tracking
    await orchestrator.route_request(
        user_input="first message",
        user_id="test_user",
        session_id="test_session"
    )

    # Verify state was stored
    history = await mock_storage.fetch_chat(
        user_id="test_user",
        session_id="test_session",
        agent_id=orchestrator.supervisor.id
    )

    assert len(history) > 0
    assert history[0].role == ParticipantRole.USER.value
