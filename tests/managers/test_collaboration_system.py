import pytest
from datetime import datetime, timedelta
import uuid
from typing import Dict, List
from unittest.mock import Mock, AsyncMock

from MAX.types.workflow_types import WorkflowStage
from MAX.types.collaboration_types import (
    CollaborationRole,
    CollaborationMessage,
    SubTask
)
from MAX.managers.task_division_manager import TaskDivisionManager, TaskDivisionPlan
from MAX.managers.coordination_protocol import (
    CoordinationProtocol,
    ProtocolType,
    MessagePriority
)
from MAX.managers.response_aggregator import (
    ResponseAggregator,
    AgentResponse,
    ResponseType,
    AggregationStrategy
)

class MockAgent:
    """Mock agent for testing."""
    def __init__(self, agent_id: str, capabilities: List[str] = None):
        self.id = agent_id
        self.name = f"Agent_{agent_id}"
        self.capabilities = capabilities or []

# Test fixtures
@pytest.fixture
def task_division_manager():
    return TaskDivisionManager()

@pytest.fixture
def coordination_protocol():
    return CoordinationProtocol()

@pytest.fixture
def response_aggregator():
    return ResponseAggregator()

@pytest.fixture
def mock_agents():
    return [
        MockAgent("agent1", ["memory_access", "context_analysis"]),
        MockAgent("agent2", ["reasoning", "planning"]),
        MockAgent("agent3", ["execution", "implementation"])
    ]

# Task Division Tests
async def test_task_division_workflow_type(task_division_manager, mock_agents):
    """Test dividing a workflow-type task."""
    task_id = str(uuid.uuid4())
    task_desc = "Analyze and implement feature X"
    
    plan = await task_division_manager.create_task_division(
        parent_task_id=task_id,
        task_description=task_desc,
        available_agents=mock_agents,
        task_type="WORKFLOW",
        complexity_level=1
    )
    
    assert len(plan.subtasks) == 3  # Memory, Reasoning, Execution
    assert all(subtask.parent_task_id == task_id for subtask in plan.subtasks)
    
    # Verify task dependencies
    memory_task = next(t for t in plan.subtasks if "Memory" in t.description)
    reasoning_task = next(t for t in plan.subtasks if "Reasoning" in t.description)
    execution_task = next(t for t in plan.subtasks if "Execution" in t.description)
    
    assert not plan.dependencies.get(memory_task.id, set())  # No dependencies
    assert plan.dependencies[reasoning_task.id] == {memory_task.id}
    assert plan.dependencies[execution_task.id] == {reasoning_task.id}

async def test_task_division_parallel_type(task_division_manager, mock_agents):
    """Test dividing a parallel task."""
    task_id = str(uuid.uuid4())
    plan = await task_division_manager.create_task_division(
        parent_task_id=task_id,
        task_description="Process data chunks",
        available_agents=mock_agents,
        task_type="PARALLEL",
        complexity_level=3
    )
    
    assert len(plan.subtasks) == 3  # Based on complexity
    assert not any(plan.dependencies.values())  # No dependencies in parallel tasks

# Coordination Protocol Tests
async def test_broadcast_message(coordination_protocol):
    """Test broadcasting messages to all participants."""
    task_id = str(uuid.uuid4())
    participants = ["agent1", "agent2", "agent3"]
    
    await coordination_protocol.register_protocol(
        task_id=task_id,
        protocol_type=ProtocolType.BROADCAST,
        participants=participants
    )
    
    message_id = await coordination_protocol.send_message(
        task_id=task_id,
        from_agent="coordinator",
        content={"message": "test broadcast"},
        priority=MessagePriority.NORMAL
    )
    
    # Check each participant got the message
    for participant in participants:
        messages = await coordination_protocol.get_messages(participant)
        assert len(messages) == 1
        assert messages[0].content == {"message": "test broadcast"}

async def test_targeted_message(coordination_protocol):
    """Test sending targeted messages."""
    task_id = str(uuid.uuid4())
    participants = ["agent1", "agent2", "agent3"]
    target_agents = ["agent1", "agent2"]
    
    await coordination_protocol.register_protocol(
        task_id=task_id,
        protocol_type=ProtocolType.TARGETED,
        participants=participants
    )
    
    await coordination_protocol.send_message(
        task_id=task_id,
        from_agent="coordinator",
        content={"message": "test targeted"},
        target_agents=target_agents
    )
    
    # Check only targeted agents got the message
    assert len(await coordination_protocol.get_messages("agent1")) == 1
    assert len(await coordination_protocol.get_messages("agent2")) == 1
    assert len(await coordination_protocol.get_messages("agent3")) == 0

# Response Aggregation Tests
async def test_sequential_aggregation(response_aggregator):
    """Test sequential response aggregation."""
    task_id = str(uuid.uuid4())
    
    # Add sequential responses
    await response_aggregator.add_response(
        task_id,
        AgentResponse(
            agent_id="agent1",
            content="Step 1 result",
            response_type=ResponseType.TEXT,
            timestamp=datetime.utcnow()
        )
    )
    
    await response_aggregator.add_response(
        task_id,
        AgentResponse(
            agent_id="agent2",
            content="Step 2 result",
            response_type=ResponseType.TEXT,
            timestamp=datetime.utcnow() + timedelta(seconds=1)
        )
    )
    
    result = await response_aggregator.aggregate_responses(
        task_id,
        strategy=AggregationStrategy.SEQUENTIAL
    )
    
    assert "Step 1 result" in result.merged_content
    assert "Step 2 result" in result.merged_content
    assert result.strategy_used == AggregationStrategy.SEQUENTIAL

async def test_parallel_aggregation(response_aggregator):
    """Test parallel response aggregation."""
    task_id = str(uuid.uuid4())
    
    # Add parallel responses
    responses = [
        AgentResponse(
            agent_id=f"agent{i}",
            content={"part": i, "result": f"data_{i}"},
            response_type=ResponseType.STRUCTURED,
            timestamp=datetime.utcnow()
        )
        for i in range(3)
    ]
    
    for response in responses:
        await response_aggregator.add_response(task_id, response)
    
    result = await response_aggregator.aggregate_responses(
        task_id,
        strategy=AggregationStrategy.PARALLEL
    )
    
    assert isinstance(result.merged_content, dict)
    assert len(result.merged_content) == 3

async def test_weighted_aggregation(response_aggregator):
    """Test weighted response aggregation."""
    task_id = str(uuid.uuid4())
    weights = {"expert": 2.0, "novice": 0.5}
    
    await response_aggregator.add_response(
        task_id,
        AgentResponse(
            agent_id="expert",
            content="Expert opinion",
            response_type=ResponseType.TEXT,
            timestamp=datetime.utcnow(),
            confidence=0.9
        )
    )
    
    await response_aggregator.add_response(
        task_id,
        AgentResponse(
            agent_id="novice",
            content="Novice opinion",
            response_type=ResponseType.TEXT,
            timestamp=datetime.utcnow(),
            confidence=0.6
        )
    )
    
    result = await response_aggregator.aggregate_responses(
        task_id,
        strategy=AggregationStrategy.WEIGHTED,
        weights=weights
    )
    
    assert result.confidence_score > 0.7  # Expert opinion should have more weight

# End-to-end collaboration test
async def test_end_to_end_collaboration(
    task_division_manager,
    coordination_protocol,
    response_aggregator,
    mock_agents
):
    """Test complete collaboration flow."""
    task_id = str(uuid.uuid4())
    task_desc = "Complex analysis task"
    
    # 1. Divide task
    plan = await task_division_manager.create_task_division(
        parent_task_id=task_id,
        task_description=task_desc,
        available_agents=mock_agents,
        task_type="WORKFLOW",
        complexity_level=1
    )
    
    # 2. Setup coordination
    await coordination_protocol.register_protocol(
        task_id=task_id,
        protocol_type=ProtocolType.BROADCAST,
        participants=[agent.id for agent in mock_agents]
    )
    
    # 3. Execute subtasks and collect responses
    for subtask in plan.subtasks:
        # Simulate agent work
        await response_aggregator.add_response(
            task_id,
            AgentResponse(
                agent_id=subtask.assigned_agent,
                content=f"Result for {subtask.description}",
                response_type=ResponseType.TEXT,
                timestamp=datetime.utcnow()
            )
        )
        
        # Update task status
        subtask.status = "completed"
        subtask.completed_at = datetime.utcnow()
    
    # 4. Aggregate final results
    final_result = await response_aggregator.aggregate_responses(
        task_id,
        strategy=AggregationStrategy.SEQUENTIAL
    )
    
    assert final_result.responses
    assert final_result.merged_content
    assert final_result.confidence_score > 0
