import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from MAX.orchestrator import MultiAgentOrchestrator
from MAX.classifiers import ClassifierResult, AnthropicClassifier, AnthropicClassifierOptions
from MAX.agents import Agent, AgentResponse, AnthropicAgent, AnthropicAgentOptions
from MAX.types import ConversationMessage, ParticipantRole, AgentTypes

class TestMultiAgentOrchestrator:
    @pytest.fixture
    def task_expert_agent(self):
        """Create a real AnthropicAgent for task handling."""
        agent = AnthropicAgent(
            options=AnthropicAgentOptions(
                api_key="test-key",
                name="Task Expert",
                description="A specialized agent for task analysis and execution",
                agent_type=AgentTypes.TASK_EXPERT.value,
                streaming=False,
            )
        )
        # Mock only the process_request method to avoid actual API calls
        agent.process_request = AsyncMock(
            return_value=ConversationMessage(
                role=ParticipantRole.ASSISTANT.value,
                content="Task expert response"
            )
        )
        return agent

    @pytest.fixture
    def general_agent(self):
        """Create a real AnthropicAgent for general tasks."""
        agent = AnthropicAgent(
            options=AnthropicAgentOptions(
                api_key="test-key",
                name="General Assistant",
                description="A general-purpose assistant for handling various tasks",
                agent_type=AgentTypes.DEFAULT.value,
                streaming=False,
            )
        )
        # Mock only the process_request method to avoid actual API calls
        agent.process_request = AsyncMock(
            return_value=ConversationMessage(
                role=ParticipantRole.ASSISTANT.value,
                content="General assistant response"
            )
        )
        return agent

    @pytest.fixture
    def test_classifier(self):
        """Create a real AnthropicClassifier with mocked API calls."""
        classifier = AnthropicClassifier(
            options=AnthropicClassifierOptions(
                api_key="test-key",
                min_confidence_threshold=0.7
            )
        )
        # Mock the API calls but keep the real classification logic
        classifier.client = MagicMock()
        classifier.client.messages.create = AsyncMock()
        return classifier

    @pytest.fixture
    def mock_storage(self):
        storage = MagicMock()
        storage.fetch_all_chats = AsyncMock(return_value=[])
        storage.fetch_chat = AsyncMock(return_value=[])
        storage.save_chat_message = AsyncMock()
        return storage

    @pytest.fixture
    def orchestrator(self, test_classifier, mock_storage):
        orchestrator = MultiAgentOrchestrator(
            classifier=test_classifier,
            storage=mock_storage
        )
        # Disable default agent usage to test proper fallback behavior
        orchestrator.config.USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED = False
        return orchestrator

    @pytest.mark.asyncio
    async def test_route_request_with_successful_classification(
        self, orchestrator, task_expert_agent, test_classifier, mock_storage
    ) -> None:
        # Setup
        user_input = "test input"
        user_id = "test_user"
        session_id = "test_session"
        
        # Setup classifier mock response
        test_classifier.client.messages.create.return_value = MagicMock(
            content=[
                MagicMock(
                    type="tool_use",
                    input={
                        "userinput": "analyze this task",
                        "primary_agent": {
                            "name": "Task Expert",
                            "confidence": 0.9,
                            "reasoning": "Task analysis request"
                        },
                        "fallback_agents": [],
                        "detected_intents": ["task_analysis"]
                    }
                )
            ]
        )
        
        # Add the agent to the orchestrator
        orchestrator.add_agent(task_expert_agent)

        # Test
        response = await orchestrator.route_request(user_input, user_id, session_id)
        
        # Verify
        assert isinstance(response, AgentResponse)
        assert response.metadata.agent_id == task_expert_agent.id
        assert "task_analysis" in response.metadata.additional_params["detected_intents"]

    @pytest.mark.asyncio
    async def test_route_request_with_fallback_to_general(
        self, orchestrator, task_expert_agent, general_agent, test_classifier
    ) -> None:
        # Setup
        user_input = "test input"
        user_id = "test_user"
        session_id = "test_session"
        
        # Add both agents to the orchestrator
        orchestrator.add_agent(task_expert_agent)
        orchestrator.add_agent(general_agent)

        # Setup classifier mock response - simulate low confidence for task expert
        test_classifier.client.messages.create.return_value = MagicMock(
            content=[
                MagicMock(
                    type="tool_use",
                    input={
                        "userinput": "what's the weather like?",
                        "primary_agent": {
                            "name": "Task Expert",
                            "confidence": 0.5,  # Below threshold
                            "reasoning": "Not a task-related query"
                        },
                        "fallback_agents": [
                            {
                                "name": "General Assistant",
                                "confidence": 0.8,
                                "reasoning": "Can handle general queries"
                            }
                        ],
                        "detected_intents": ["general_query", "weather"]
                    }
                )
            ]
        )

        # Test
        response = await orchestrator.route_request(user_input, user_id, session_id)
        
        # Verify
        assert isinstance(response, AgentResponse)
        assert response.metadata.agent_id == general_agent.id
        assert set(response.metadata.additional_params["detected_intents"].split(",")) == {"general_query", "weather"}
        assert "failure_reason" in response.metadata.additional_params

    @pytest.mark.asyncio
    async def test_route_request_with_no_suitable_agents(
        self, orchestrator, test_classifier
    ) -> None:
        # Setup
        user_input = "unrecognizable input"
        user_id = "test_user"
        session_id = "test_session"
        
        # Setup classifier mock response - no suitable agents
        test_classifier.client.messages.create.return_value = MagicMock(
            content=[
                MagicMock(
                    type="tool_use",
                    input={
                        "userinput": "something completely unrecognizable",
                        "primary_agent": {
                            "name": "Unknown",
                            "confidence": 0.1,
                            "reasoning": "Cannot determine appropriate agent"
                        },
                        "fallback_agents": [],
                        "detected_intents": ["unknown_intent"]
                    }
                )
            ]
        )

        # Test
        response = await orchestrator.route_request(user_input, user_id, session_id)
        
        # Verify
        assert isinstance(response, AgentResponse)
        assert response.metadata.agent_id == "no_agent_selected"
        assert "classification_failed" in response.metadata.additional_params["error_type"]
        assert "unknown_intent" in response.metadata.additional_params["detected_intents"]
        assert "No suitable agent found" in response.metadata.additional_params["failure_reason"]
