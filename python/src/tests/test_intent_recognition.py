import pytest
import asyncio
from typing import Dict, Any, List
from unittest.mock import AsyncMock, Mock, patch

from MAX.types import (
    ConversationMessage,
    ParticipantRole,
    OrchestratorConfig,
    AgentResponse
)
from MAX.classifiers import Classifier, ClassifierResult
from MAX.agents import (
    Agent, 
    AgentOptions,
    TaskExpertOptions, 
    TaskExpertAgent
)
from MAX.orchestrator import MultiAgentOrchestrator
from MAX.storage import InMemoryChatStorage

class MockAgent(Agent):
    def __init__(self, name: str, description: str):
        options = AgentOptions(
            name=name,
            description=description
        )
        super().__init__(options)
        self._process_request_mock = AsyncMock(return_value=AgentResponse(
            metadata={
                "agent_id": self.id,
                "agent_name": self.name,
                "user_id": "test_user",
                "session_id": "test_session",
                "additional_params": {}
            },
            output=ConversationMessage(
                role=ParticipantRole.ASSISTANT.value,
                content="Mock response"
            ),
            streaming=False
        ))

    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: List[ConversationMessage],
        additional_params: Dict[str, Any] = None
    ) -> AgentResponse:
        return await self._process_request_mock(
            input_text, 
            user_id, 
            session_id, 
            chat_history, 
            additional_params
        )

    def is_streaming_enabled(self) -> bool:
        return False

class MockClassifier(Classifier):
    def __init__(self):
        self.classify = AsyncMock()
        self.agents = {}  # Will be set by orchestrator
        self.default_agent = MockAgent(
            "Default Agent",
            "A mock default agent for testing"
        )

    def set_agents(self, agents: Dict[str, Agent]):
        self.agents = agents

    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: List[ConversationMessage],
        additional_params: Dict[str, Any] = None
    ) -> AgentResponse:
        return AgentResponse(
            metadata={
                "agent_id": "mock-classifier",
                "agent_name": "Mock Classifier",
                "user_id": user_id,
                "session_id": session_id,
                "additional_params": additional_params or {}
            },
            output=ConversationMessage(
                role=ParticipantRole.ASSISTANT.value,
                content="Mock classifier response"
            ),
            streaming=False
        )

@pytest.fixture
def mock_task_expert_options():
    with patch('multi_agent_orchestrator.agents.TaskExpertOptions') as mock_opts:
        # Create a mock that will accept any kwargs
        mock_opts.return_value = Mock(
            name="DEFAULT",
            description="A knowledgeable generalist capable of addressing a wide range of topics."
        )
        yield mock_opts

@pytest.fixture
def mock_storage():
    return InMemoryChatStorage()

@pytest.fixture
def mock_agents():
    return {
        "crypto": MockAgent("Crypto Expert", "Handles cryptocurrency and market analysis"),
        "general": MockAgent("General Assistant", "Handles general queries and tasks")
    }

@pytest.fixture
def mock_classifier():
    return MockClassifier()

@pytest.fixture
async def orchestrator(mock_agents, mock_classifier, mock_storage, mock_task_expert_options):
    with patch('multi_agent_orchestrator.orchestrator.StateManager') as mock_state_manager, \
         patch('multi_agent_orchestrator.orchestrator.Logger') as mock_logger, \
         patch('multi_agent_orchestrator.orchestrator.TaskExpertAgent') as mock_task_expert, \
         patch('multi_agent_orchestrator.orchestrator.AnthropicClassifier') as mock_anthropic_classifier, \
         patch('multi_agent_orchestrator.orchestrator.TaskExpertOptions', mock_task_expert_options), \
         patch('asyncio.create_task') as mock_create_task:
        
        # Configure mocks
        mock_state_manager.return_value = Mock(
            restore_state_from_storage=AsyncMock(),
            track_conversation_state=AsyncMock(),
            _run_periodic_cleanup=AsyncMock()
        )
        
        mock_logger.return_value = Mock(
            info=Mock(),
            error=Mock(),
            print_chat_history=Mock(),
            log_header=Mock()
        )
        
        # Mock the TaskExpertAgent to return our mock general agent
        mock_task_expert.return_value = mock_agents["general"]
        
        # Mock AnthropicClassifier to prevent any real API calls
        mock_anthropic_classifier.return_value = mock_classifier
        
        # Mock asyncio.create_task to prevent event loop errors
        mock_create_task.return_value = Mock()

        # Create orchestrator with all dependencies mocked
        config = OrchestratorConfig(
            USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=True,
            LOG_AGENT_CHAT=False,
            LOG_CLASSIFIER_OUTPUT=False
        )

        orchestrator = MultiAgentOrchestrator(
            options=config,
            storage=mock_storage,
            classifier=mock_classifier,
            logger=mock_logger.return_value
        )

        # Register agents
        for agent in mock_agents.values():
            orchestrator.add_agent(agent)

        yield orchestrator

@pytest.mark.asyncio
async def test_crypto_intent_recognition(orchestrator, mock_agents, mock_classifier):
    # Setup
    user_input = "What's the current market trend for Bitcoin?"
    mock_classifier.classify.return_value = ClassifierResult(
        selected_agent=mock_agents["crypto"],
        confidence=0.92
    )
    
    # Test
    response = await orchestrator.route_request(
        user_input=user_input,
        user_id="test_user",
        session_id="test_session"
    )
    
    # Verify
    mock_classifier.classify.assert_called_once()
    assert response.metadata.agent_id == "crypto-expert"
    assert response.metadata.agent_name == "Crypto Expert"

@pytest.mark.asyncio
async def test_fallback_to_default_agent(orchestrator, mock_agents, mock_classifier):
    # Setup
    user_input = "What's the weather like today?"
    mock_classifier.classify.return_value = ClassifierResult(
        selected_agent=None,
        confidence=0.0
    )
    
    # Test
    response = await orchestrator.route_request(
        user_input=user_input,
        user_id="test_user",
        session_id="test_session"
    )
    
    # Verify
    assert response.metadata.agent_id == "general-assistant"
    assert response.metadata.agent_name == "General Assistant"

@pytest.mark.asyncio
async def test_conversation_context_maintenance(orchestrator, mock_agents, mock_classifier):
    # Setup - First message
    mock_classifier.classify.return_value = ClassifierResult(
        selected_agent=mock_agents["crypto"],
        confidence=0.95
    )
    
    # First message
    await orchestrator.route_request(
        user_input="How's Bitcoin performing?",
        user_id="test_user",
        session_id="test_session"
    )
    
    # Follow-up message should go to same agent
    mock_classifier.classify.return_value = ClassifierResult(
        selected_agent=mock_agents["crypto"],
        confidence=0.85
    )
    
    response = await orchestrator.route_request(
        user_input="What about its trading volume?",
        user_id="test_user",
        session_id="test_session"
    )
    
    assert response.metadata.agent_id == "crypto-expert"

@pytest.mark.asyncio
async def test_low_confidence_handling(orchestrator, mock_agents, mock_classifier):
    # Setup
    user_input = "hmm..."
    mock_classifier.classify.return_value = ClassifierResult(
        selected_agent=mock_agents["general"],
        confidence=0.3
    )
    
    # Test
    response = await orchestrator.route_request(
        user_input=user_input,
        user_id="test_user",
        session_id="test_session"
    )
    
    # Verify we still get a response even with low confidence
    assert response.metadata.agent_id == "general-assistant"

@pytest.mark.asyncio
async def test_error_handling(orchestrator, mock_agents, mock_classifier):
    # Setup
    user_input = "Test query"
    mock_classifier.classify.side_effect = Exception("Classification error")
    
    # Test
    response = await orchestrator.route_request(
        user_input=user_input,
        user_id="test_user",
        session_id="test_session"
    )
    
    # Verify error handling
    assert "error_type" in response.metadata.additional_params
    assert response.metadata.additional_params["error_type"] == "classification_failed"