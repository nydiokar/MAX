import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import sys
from dataclasses import dataclass, field
from MAX.agents.supervisor_agent import SupervisorAgent, SupervisorAgentOptions
from MAX.agents.agent import Agent
from MAX.agents.options import AgentOptions, BaseAgentOptions
from MAX.types.workflow_types import WorkflowStage
from tests.utils.test_helpers import MockChatStorage
from MAX.types.collaboration_management_types import (
    TaskDivisionPlan,
    AgentResponse,
    ResponseType,
    AggregationStrategy
)
from MAX.storage.in_memory_chat_storage import InMemoryChatStorage
from MAX.utils.tool import AgentTools, AgentTool
from MAX.agents.teams.team_registry import TeamRegistry, TeamSpec, TeamConfiguration, TeamType
from MAX.types.collaboration_types import CollaborationRole
from datetime import datetime
from typing import Optional, Any, List, Union, Dict
from pydantic import BaseModel, Field, validator
from tests.utils.test_helpers import (
    MockBaseAgent,
    MockChatStorage,
    create_mock_team_registry,
    create_supervisor_agent
)

class BaseAgentOptions(BaseModel):
    """Base options with required fields"""
    name: str
    description: str

class AgentOptions(BaseAgentOptions):
    """Standard agent options"""
    model_id: Optional[str] = None
    region: Optional[str] = None
    callbacks: Optional[Any] = None
    save_chat: bool = True

class SupervisorAgentOptions(BaseModel):
    """Supervisor specific options with validation"""
    # Required fields
    lead_agent: Any
    
    # Fields that will be set from lead_agent
    name: Optional[str] = None
    description: Optional[str] = None
    
    # Optional fields
    team_registry: Optional[TeamRegistry] = None
    active_team_id: Optional[str] = None
    storage: Optional[MockChatStorage] = None
    trace: Optional[bool] = None
    extra_tools: Optional[Union[AgentTools, List[AgentTool]]] = None
    model_id: Optional[str] = None
    region: Optional[str] = None
    callbacks: Optional[Any] = None
    save_chat: bool = True

    @validator('lead_agent')
    def validate_lead_agent(cls, v):
        if not hasattr(v, 'name') or not hasattr(v, 'description'):
            raise ValueError('lead_agent must have name and description attributes')
        return v

    @validator('extra_tools')
    def validate_extra_tools(cls, v):
        if v is not None:
            if isinstance(v, AgentTools):
                tools_to_check = v.tools
            elif isinstance(v, list):
                tools_to_check = v
            else:
                raise ValueError('extra_tools must be Tools object or list of Tool objects')
            
            if not all(isinstance(tool, AgentTool) for tool in tools_to_check):
                raise ValueError('All tools must be AgentTool instances')
        return v

    class Config:
        arbitrary_types_allowed = True  # Allows Any type for lead_agent

    def model_post_init(self, __context: Any) -> None:
        """Set name and description from lead_agent after initialization"""
        if self.lead_agent:
            self.name = self.lead_agent.name
            self.description = self.lead_agent.description

# Base Agent class that both mock agents will inherit from
class MockBaseAgent:
    def __init__(self):
        self.name = "MOCK_AGENT"
        self.description = "Mock agent for testing"
        self.provider_type = "mock"
        self.capabilities = ["text_generation"]
        self.tool_config = None

    async def generate_response(self, *args, **kwargs):
        return {"content": f"Mock response from {self.name}", "confidence": 0.9}

    async def process_task(self, *args, **kwargs):
        return {"status": "success", "result": f"Processed by {self.name}"}

class MockAnthropicAgent(MockBaseAgent):
    def __init__(self):
        super().__init__()
        self.name = "ANTHROPIC_AGENT"
        self.description = "Mock Anthropic agent for testing"
        self.provider_type = "anthropic"
        self.capabilities = ["text_generation", "reasoning", "memory"]

class MockBedrockAgent(MockBaseAgent):
    def __init__(self):
        self.options = AgentOptions(
            name="BEDROCK_AGENT",
            description="Mock Bedrock agent for testing",
            provider_type="bedrock",
            capabilities=["text_generation", "execution"],
            model_id="anthropic.claude-3",
            region="us-east-1",
            callbacks=None,
            save_chat=True
        )
        self.name = self.options.name
        self.description = self.options.description
        self.provider_type = self.options.provider_type
        self.capabilities = self.options.capabilities
        self.tool_config = None

class MockChatStorage:
    """Mock implementation of ChatStorage with all required methods"""
    def __init__(self):
        self.messages = {}
        self.system_state = {}
        self.task_states = {}

    async def initialize(self):
        return True

    async def cleanup(self):
        return True

    async def check_health(self):
        return {"status": "healthy"}

    async def save_system_state(self, state: dict):
        self.system_state = state
        return True

    async def get_system_state(self):
        return self.system_state

    async def save_task_state(self, task_id: str, state: dict):
        self.task_states[task_id] = state
        return True

    async def search_similar_messages(self, query: str, limit: int = 5):
        return []

    # Add any other methods from ChatStorage that need to be implemented
    async def save_chat(self, user_id: str, session_id: str, messages: List[dict]):
        self.messages[(user_id, session_id)] = messages
        return True

    async def fetch_chat(self, user_id: str, session_id: str):
        return self.messages.get((user_id, session_id), [])

    async def fetch_all_chats(self, user_id: str, session_id: str):
        return self.messages.get((user_id, session_id), [])

# Fixtures for all required components
@pytest.fixture
def mock_storage():
    return MockChatStorage()

@pytest.fixture
def mock_team():
    return MockTeam()

@pytest.fixture
def mock_team_registry():
    registry = Mock(spec=TeamRegistry)
    registry.register_team = AsyncMock(return_value="team-123")
    registry.get_team = AsyncMock(return_value={"id": "team-123", "agents": []})
    registry.active_teams = {"team-123": {"id": "team-123", "agents": []}}
    return registry

@pytest.fixture
def mock_tools():
    return None  # Start with no extra tools

@pytest.fixture
def mock_anthropic_agent():
    agent = MockAnthropicAgent()
    agent.tool_config = None
    agent.generate_response = AsyncMock(return_value={
        "content": "Mock response",
        "confidence": 0.9
    })
    return agent

@pytest.fixture
def mock_bedrock_agent():
    return MockBedrockAgent()

# Main patches for the test module
@pytest.fixture(autouse=True)
def mock_imports():
    # Create mock modules
    mock_anthropic = MagicMock()
    mock_anthropic.AnthropicAgent = MockAnthropicAgent
    
    mock_bedrock = MagicMock()
    mock_bedrock.BedrockLLMAgent = MockBedrockAgent
    
    mock_agents = MagicMock()
    mock_agents.AnthropicAgent = MockAnthropicAgent
    mock_agents.BedrockLLMAgent = MockBedrockAgent
    
    with patch.dict(sys.modules, {
        'MAX.agents': mock_agents,
        'MAX.agents.anthropic_agent': mock_anthropic,
        'MAX.agents.bedrock_agent': mock_bedrock
    }):
        yield

@pytest.fixture
def supervisor_agent(mock_anthropic_agent, mock_storage, mock_team_registry):
    options = SupervisorAgentOptions(
        lead_agent=mock_anthropic_agent,
        team_registry=mock_team_registry,
        storage=mock_storage,
        trace=True,
        extra_tools=None
    )
    agent = SupervisorAgent(options)
    agent.user_id = 'test-user'
    agent.session_id = 'test-session'
    return agent

class MockTeam:
    def __init__(self, team_id="mock-team-1"):
        self.id = team_id
        self.spec = TeamConfiguration(
            team_type=TeamType.WORKFLOW,
            max_members=5,
            required_roles=["memory_agent", "reasoning_agent"]
        )
        self.members = {}
        self.status = "active"
        self.type = TeamType.WORKFLOW
        self.configuration = self.spec

class TestSupervisorAgent:
    """Test suite for SupervisorAgent"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test"""
        self.storage = MockChatStorage()
        self.team_registry = create_mock_team_registry()
        self.supervisor = create_supervisor_agent(
            storage=self.storage,
            team_registry=self.team_registry
        )
        self.supervisor.user_id = 'test-user'
        self.supervisor.session_id = 'test-session'

    @pytest.mark.asyncio
    async def test_supervisor_agent_initialization(self, supervisor_agent):
        """Test that supervisor agent initializes correctly"""
        assert supervisor_agent.lead_agent is not None
        assert supervisor_agent.team_registry is not None
        assert supervisor_agent.storage is not None
        assert supervisor_agent.trace is True
        assert supervisor_agent.supervisor_tools is not None

    @pytest.mark.asyncio
    async def test_task_division_and_collaboration(self):
        """Test that supervisor can divide and manage tasks"""
        result = await self.supervisor.create_task_division(
            task_description="Test task",
            task_type="SEQUENTIAL"
        )
        assert isinstance(result, str)
        assert "team" in result.lower()

    @pytest.mark.asyncio
    async def test_response_aggregation(self):
        """Test that supervisor can aggregate responses"""
        task_id = "test-task"
        responses = [
            AgentResponse(
                agent_id="agent1",
                content="Response 1",
                response_type=ResponseType.TEXT,
                confidence=0.8,
                timestamp=datetime.utcnow()
            ),
            AgentResponse(
                agent_id="agent2",
                content="Response 2",
                response_type=ResponseType.TEXT,
                confidence=0.9,
                timestamp=datetime.utcnow()
            )
        ]
        self.supervisor.response_buffer[task_id] = responses
        result = await self.supervisor.aggregate_responses(task_id=task_id)
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_team_communication(self, supervisor_agent):
        """Test supervisor can manage team communication"""
        await supervisor_agent.activate_team(
            team_type=TeamType.WORKFLOW.value,
            task_description="test task"
        )
        
        result = await supervisor_agent.send_team_message(content="Test message")
        assert isinstance(result, str)
        assert "Mock response" in result

    @pytest.mark.asyncio
    async def test_weighted_response_aggregation(self):
        """Test weighted response aggregation based on agent capabilities"""
        task_id = "weighted_task"
        responses = [
            AgentResponse(
                agent_id="memory_agent",
                content="Memory response",
                response_type=ResponseType.TEXT,
                confidence=0.8,
                timestamp=datetime.utcnow(),
                metadata={"capability": "memory"}
            ),
            AgentResponse(
                agent_id="reasoning_agent",
                content="Reasoning response", 
                response_type=ResponseType.TEXT,
                confidence=0.9,
                timestamp=datetime.utcnow(),
                metadata={"capability": "reasoning"}
            )
        ]
        
        self.supervisor.response_buffer[task_id] = responses
        weights = {"memory_agent": 1.0, "reasoning_agent": 0.8}
        
        result = await self.supervisor.aggregate_responses(
            task_id=task_id,
            strategy=AggregationStrategy.WEIGHTED.value,
            weights=weights
        )
        
        assert isinstance(result, str)
        assert "Memory response" in result or "Reasoning response" in result

    @pytest.mark.asyncio
    async def test_process_request(self, supervisor_agent):
        """Test processing a user request"""
        result = await supervisor_agent.process_request(
            input_text="Test request",
            user_id="test-user",
            session_id="test-session",
            chat_history=[],
            additional_params={}
        )
        
        assert result is not None
        assert "Mock response" in result.content
