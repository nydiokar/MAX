import pytest
from datetime import datetime
import uuid
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch

from MAX.agents.supervisor_agent import SupervisorAgent, SupervisorAgentOptions
from MAX.agents.teams.team_registry import TeamRegistry, TeamType, TeamConfiguration
from MAX.types.workflow_types import WorkflowStage
from MAX.types.collaboration_types import (
    CollaborationRole,
    ResponseType,
    AgentResponse
)
from MAX.managers.task_division_manager import TaskDivisionManager
from MAX.managers.response_aggregator import ResponseAggregator
from MAX.storage import InMemoryChatStorage

class MockAgent:
    """Mock agent for testing."""
    def __init__(
        self,
        agent_id: str,
        name: str = None,
        description: str = None,
        capabilities: List[str] = None
    ):
        self.id = agent_id
        self.name = name or f"Agent_{agent_id}"
        self.description = description or f"Mock agent {agent_id}"
        self.capabilities = capabilities or []
        self.tool_config = None
        self.save_chat = False

    async def process_request(self, *args, **kwargs):
        return Mock(content=[{'text': 'mock response'}])

@pytest.fixture
def mock_lead_agent():
    return MockAgent(
        "lead_agent",
        "Lead Agent",
        "Lead agent for testing",
        ["coordination", "planning"]
    )

@pytest.fixture
def mock_team():
    return [
        MockAgent(
            "memory_agent",
            capabilities=["memory_access", "context_analysis"]
        ),
        MockAgent(
            "reasoning_agent",
            capabilities=["reasoning", "planning"]
        ),
        MockAgent(
            "execution_agent",
            capabilities=["execution", "implementation"]
        )
    ]

@pytest.fixture
def mock_team_registry():
    registry = TeamRegistry()
    return registry

@pytest.fixture
def supervisor_agent(mock_lead_agent, mock_team, mock_team_registry):
    options = SupervisorAgentOptions(
        lead_agent=mock_lead_agent,
        team_registry=mock_team_registry,
        team=mock_team,
        storage=InMemoryChatStorage()
    )
    return SupervisorAgent(options)

# Supervisor Agent Tests
async def test_supervisor_initialization(supervisor_agent, mock_team):
    """Test supervisor agent initialization."""
    assert supervisor_agent.lead_agent is not None
    assert len(supervisor_agent.team) == len(mock_team)
    assert isinstance(supervisor_agent.task_manager, TaskDivisionManager)
    assert isinstance(supervisor_agent.response_aggregator, ResponseAggregator)

async def test_activate_team(supervisor_agent):
    """Test team activation."""
    result = await supervisor_agent.activate_team(
        team_type=TeamType.WORKFLOW.value,
        task_description="Test task",
        workflow_stage=WorkflowStage.MEMORY.value
    )
    
    assert "team" in result.lower()
    assert supervisor_agent.active_team_id is not None

async def test_create_task_division(supervisor_agent):
    """Test task division creation."""
    result = await supervisor_agent.create_task_division(
        task_description="Complex task",
        task_type="WORKFLOW",
        complexity=2
    )
    
    assert "subtasks" in result.lower()
    assert supervisor_agent.active_task_id is not None

async def test_send_team_message(supervisor_agent, mock_team):
    """Test sending messages to team members."""
    # First activate a team
    await supervisor_agent.activate_team(
        TeamType.WORKFLOW.value,
        "Test task"
    )
    
    result = await supervisor_agent.send_team_message(
        content="Test message",
        roles=[CollaborationRole.CONTRIBUTOR.value]
    )
    
    assert result  # Should get responses

async def test_receive_and_aggregate_responses(supervisor_agent):
    """Test response collection and aggregation."""
    task_id = str(uuid.uuid4())
    supervisor_agent.active_task_id = task_id
    
    # Add test responses
    await supervisor_agent.receive_agent_response(
        task_id=task_id,
        agent_id="memory_agent",
        content="Memory stage complete",
        response_type=ResponseType.TEXT,
        confidence=0.9
    )
    
    await supervisor_agent.receive_agent_response(
        task_id=task_id,
        agent_id="reasoning_agent",
        content="Reasoning stage complete",
        response_type=ResponseType.TEXT,
        confidence=0.85
    )
    
    # Aggregate responses
    result = await supervisor_agent.aggregate_responses(task_id)
    assert result
    assert "complete" in result.lower()

async def test_end_to_end_workflow(supervisor_agent):
    """Test complete workflow execution."""
    # 1. Activate team
    await supervisor_agent.activate_team(
        TeamType.WORKFLOW.value,
        "Complex analysis task"
    )
    
    # 2. Create task division
    division_result = await supervisor_agent.create_task_division(
        task_description="Analyze data and make recommendations",
        task_type="WORKFLOW",
        complexity=1
    )
    assert "subtasks" in division_result.lower()
    
    # 3. Send task to team
    message_result = await supervisor_agent.send_team_message(
        content="Begin analysis task",
        roles=[CollaborationRole.CONTRIBUTOR.value]
    )
    assert message_result
    
    # 4. Simulate responses
    task_id = supervisor_agent.active_task_id
    
    for agent_id in ["memory_agent", "reasoning_agent", "execution_agent"]:
        await supervisor_agent.receive_agent_response(
            task_id=task_id,
            agent_id=agent_id,
            content=f"{agent_id} task complete",
            response_type=ResponseType.TEXT,
            confidence=0.9
        )
        
        await supervisor_agent.update_task_status(
            subtask_id=f"subtask_{agent_id}",
            status="completed"
        )
    
    # 5. Get final results
    status = await supervisor_agent.get_task_status()
    assert "completed" in status.lower()
    
    final_result = await supervisor_agent.aggregate_responses(task_id)
    assert final_result
    assert "complete" in final_result.lower()

# Error handling tests
async def test_invalid_team_activation(supervisor_agent):
    """Test error handling for invalid team activation."""
    with pytest.raises(ValueError):
        await supervisor_agent.activate_team(
            team_type="INVALID",
            task_description="Test"
        )

async def test_missing_task_response_aggregation(supervisor_agent):
    """Test error handling for aggregating non-existent task."""
    result = await supervisor_agent.aggregate_responses("nonexistent_task")
    assert "error" in result.lower()

async def test_task_status_no_active_task(supervisor_agent):
    """Test getting task status with no active task."""
    supervisor_agent.active_task_id = None
    status = await supervisor_agent.get_task_status()
    assert "no active task" in status.lower()
