import pytest
from unittest.mock import Mock, patch
from MAX.orchestrator import MultiAgentOrchestrator
from MAX.config.orchestrator_config import OrchestratorConfig
from MAX.types import ConversationMessage, ParticipantRole
from MAX.types.workflow_types import WorkflowStage
from MAX.classifiers import ClassifierResult
from MAX.agents import Agent, AgentResponse
from MAX.storage import InMemoryChatStorage

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
    mock = Mock(spec=Agent)
    mock.name = "MOCK_AGENT"
    mock.id = "mock_agent"
    mock.description = "Mock agent for testing"
    mock.is_streaming_enabled = Mock(return_value=False)
    return mock

@pytest.fixture
def orchestrator(mock_classifier, mock_storage, mock_agent):
    orchestrator = MultiAgentOrchestrator(
        options=OrchestratorConfig(),
        storage=mock_storage,
        classifier=mock_classifier
    )
    orchestrator.add_agent(mock_agent)
    return orchestrator

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
