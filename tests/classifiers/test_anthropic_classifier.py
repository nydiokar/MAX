import pytest
import json
from MAX.classifiers import AnthropicClassifier, AnthropicClassifierOptions, ClassifierResult
from MAX.agents import Agent
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List
from MAX.types import ConversationMessage
from MAX.utils import Logger

class TestAnthropicClassifier:
    @pytest.fixture
    def mock_task_expert(self):
        """Create a mock task expert agent."""
        agent = MagicMock(spec=Agent)
        agent.id = "task-expert"
        agent.name = "Task Expert"
        agent.description = "A specialized agent for task analysis and execution"
        agent.process_request = AsyncMock()
        return agent

    @pytest.fixture
    def mock_general_assistant(self):
        """Create a mock general assistant agent."""
        agent = MagicMock(spec=Agent)
        agent.id = "general-assistant"
        agent.name = "General Assistant"
        agent.description = "A general-purpose assistant for handling various tasks"
        agent.process_request = AsyncMock()
        return agent

    @pytest.fixture
    def mock_anthropic_response(self):
        # Create a proper Message object structure
        message = MagicMock()
        
        # Create the tool call content
        tool_call = MagicMock()
        tool_call.function = MagicMock()
        tool_call.function.name = "analyzePrompt"
        tool_call.function.arguments = json.dumps({
            "userinput": "test input",
            "primary_agent": {
                "name": "Task Expert",
                "confidence": 0.9,
                "reasoning": "Test reasoning"
            },
            "fallback_agents": [],
            "detected_intents": ["test_intent"]
        })

        # Create content item with proper type attribute
        content_item = MagicMock()
        content_item.type = "tool_calls"
        content_item.tool_calls = [tool_call]

        # Set up the message content
        message.content = [content_item]
        
        # Debug what we're actually creating
        Logger.debug(f"Mock content type: {content_item.type}")
        Logger.debug(f"Mock tool calls: {content_item.tool_calls}")
        Logger.debug(f"Full mock message: {message.content}")
        
        return message

    @pytest.fixture
    def classifier(self, mock_anthropic_response, mock_task_expert):
        options = AnthropicClassifierOptions(
            api_key="test_key",
            min_confidence_threshold=0.7
        )
        classifier = AnthropicClassifier(options)
        
        # Set up the agents dictionary
        classifier.agents = {
            "task-expert": mock_task_expert
        }
        
        # Mock the Anthropic client with explicit return value
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        classifier.client = mock_client
        
        # Debug the mock setup
        Logger.debug(f"Mock response in classifier setup: {mock_anthropic_response.content}")
        
        return classifier

    def create_mock_response(self, response_data: dict) -> MagicMock:
        """Helper to create properly structured mock Claude response."""
        # Create the most inner function mock first
        function_mock = MagicMock()
        function_mock.name = "analyzePrompt"
        function_mock.arguments = json.dumps(response_data)

        # Create tool call mock
        tool_call = MagicMock()
        tool_call.type = "function"
        tool_call.function = function_mock

        # Create tool calls response content
        tool_calls_content = MagicMock()
        tool_calls_content.type = "tool_calls"
        tool_calls_content.tool_calls = [tool_call]

        # Create final response
        response = MagicMock()
        response.content = [tool_calls_content]
        
        return response

    @pytest.mark.asyncio
    async def test_agent_selection_with_high_confidence(self, classifier):
        try:
            # Debug before making the request
            Logger.debug(f"Classifier client: {classifier.client}")
            Logger.debug(f"Classifier agents: {classifier.agents}")
            
            result = await classifier.process_request("test input", [])
            
            # Debug the actual response
            Logger.debug(f"Test result: {result}")
            
            assert result.selected_agent is not None
            assert result.confidence >= 0.7
        except Exception as e:
            Logger.error(f"Test failed with error: {str(e)}")
            Logger.debug(f"Classifier state: {classifier.__dict__}")
            raise

    @pytest.fixture
    def mock_anthropic_response_low_confidence(self):
        message = MagicMock()
        
        tool_call = MagicMock()
        tool_call.function = MagicMock()
        tool_call.function.name = "analyzePrompt"
        tool_call.function.arguments = json.dumps({
            "userinput": "test input",
            "primary_agent": {
                "name": "Task Expert",
                "confidence": 0.5,  # Low confidence
                "reasoning": "Test reasoning"
            },
            "fallback_agents": [{
                "name": "Task Expert",  # Same as mock_task_expert
                "confidence": 0.8,
                "reasoning": "Fallback reasoning"
            }],
            "detected_intents": ["general_query"]
        })

        content_item = MagicMock()
        content_item.type = "tool_calls"
        content_item.tool_calls = [tool_call]
        message.content = [content_item]
        
        return message

    @pytest.mark.asyncio
    async def test_agent_selection_with_low_confidence(self, classifier, mock_task_expert, mock_anthropic_response_low_confidence):
        # Override the classifier's response for this specific test
        classifier.client.messages.create.return_value = mock_anthropic_response_low_confidence
        
        # Ensure the agent is available
        classifier.check_agent_availability = AsyncMock(return_value=True)
        
        # Make sure we're using the same mock_task_expert instance
        classifier.agents = {
            "task-expert": mock_task_expert
        }
        
        result = await classifier.process_request("test input", [])
        
        # Debug output
        Logger.debug(f"Result agent: {result.selected_agent}")
        Logger.debug(f"Expected agent: {mock_task_expert}")
        Logger.debug(f"Agents dict: {classifier.agents}")
        
        assert result.selected_agent == mock_task_expert
        assert result.confidence == 0.8
        assert result.fallback_reason == "Primary agent confidence (0.5) below threshold"

    @pytest.mark.asyncio
    async def test_agent_availability_check(self, classifier, mock_task_expert):
        """Test agent availability checking."""
        # Test available agent
        assert await classifier.check_agent_availability(mock_task_expert) is True
        
        # Test unavailable agent
        unavailable_agent = MagicMock(spec=Agent)
        unavailable_agent.id = "unavailable"
        unavailable_agent.name = "Unavailable Agent"
        unavailable_agent.description = "An agent without process_request"
        delattr(unavailable_agent, 'process_request')
        assert await classifier.check_agent_availability(unavailable_agent) is False

    @pytest.mark.asyncio
    async def test_agent_selection_with_multiple_intents(self, classifier, mock_task_expert, mock_general_assistant):
        """Test classifier selecting agent based on multiple detected intents."""
        # Setup
        classifier.agents = {
            "task-expert": mock_task_expert,
            "general-assistant": mock_general_assistant
        }
        
        # Create mock response data
        response_data = {
            "userinput": "analyze this task and create a plan",
            "primary_agent": {
                "name": "Task Expert",
                "confidence": 0.95,
                "reasoning": "Request involves task analysis and planning"
            },
            "fallback_agents": [
                {
                    "name": "General Assistant",
                    "confidence": 0.7,
                    "reasoning": "Can handle general planning tasks"
                }
            ],
            "detected_intents": ["task_analysis", "planning", "workflow_creation"]
        }
        
        # Setup mock response
        mock_response = self.create_mock_response(response_data)
        classifier.client.messages.create = AsyncMock(return_value=mock_response)
        
        # Test
        result = await classifier.process_request("analyze this task and create a plan", [])
        
        # Verify
        assert isinstance(result, ClassifierResult)
        assert result.selected_agent == mock_task_expert
        assert result.confidence == 0.95
        assert len(result.intents) == 3
        assert "task_analysis" in result.intents
        assert "planning" in result.intents
        assert "workflow_creation" in result.intents
