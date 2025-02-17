import pytest
from datetime import datetime
from MAX.managers.unified_state_manager import UnifiedStateManager, StateType, StateEntry
from MAX.types.collaboration_types import SubTask

@pytest.fixture
async def state_manager():
    return UnifiedStateManager()

@pytest.mark.asyncio
async def test_basic_state_update(state_manager):
    # Test basic state update and retrieval
    test_data = {"key": "value"}
    await state_manager.update_state(
        "test_id",
        test_data,
        StateType.TASK
    )
    
    state = await state_manager.get_state("test_id")
    assert state is not None
    assert state.data == test_data
    assert state.version == 1
    assert state.state_type == StateType.TASK

@pytest.mark.asyncio
async def test_version_history(state_manager):
    # Test version history tracking
    test_data_1 = {"version": 1}
    test_data_2 = {"version": 2}
    
    await state_manager.update_state(
        "test_id",
        test_data_1,
        StateType.TASK
    )
    await state_manager.update_state(
        "test_id",
        test_data_2,
        StateType.TASK
    )
    
    current = await state_manager.get_state("test_id")
    old_version = await state_manager.get_state("test_id", version=1)
    
    assert current.data == test_data_2
    assert current.version == 2
    assert old_version.data == test_data_1
    assert old_version.version == 1

@pytest.mark.asyncio
async def test_message_queue(state_manager):
    # Test message queue operations
    test_message = {"content": "test"}
    await state_manager.enqueue_message("queue1", test_message)
    
    messages = await state_manager.dequeue_messages("queue1", max_messages=1)
    assert len(messages) == 1
    assert messages[0] == test_message

@pytest.mark.asyncio
async def test_task_dependencies(state_manager):
    # Test task dependency management
    await state_manager.add_task_dependency("task1", "dep1")
    await state_manager.add_task_dependency("task1", "dep2")
    
    deps = await state_manager.get_task_dependencies("task1")
    assert deps == {"dep1", "dep2"}

@pytest.mark.asyncio
async def test_state_notifications(state_manager):
    # Test state change notifications
    notifications = []
    
    async def test_callback(state_entry):
        notifications.append(state_entry)
    
    await state_manager.subscribe_to_state("test_id", test_callback)
    
    test_data = {"key": "value"}
    await state_manager.update_state(
        "test_id",
        test_data,
        StateType.TASK
    )
    
    assert len(notifications) == 1
    assert notifications[0].data == test_data

@pytest.mark.asyncio
async def test_task_state_update(state_manager):
    # Test task-specific state updates
    task = SubTask(
        id="test_task",
        parent_task_id="parent",
        assigned_agent="agent1",
        description="Test task",
        dependencies=set(),
        status="pending",
        created_at=datetime.utcnow()
    )
    
    await state_manager.update_task_state(
        "test_task",
        task,
        metadata={"priority": "high"}
    )
    
    state = await state_manager.get_state("test_task")
    assert state.data == task
    assert state.metadata["priority"] == "high"

@pytest.mark.asyncio
async def test_cleanup(state_manager):
    # Test state history cleanup
    for i in range(150):
        await state_manager.update_state(
            "test_id",
            {"count": i},
            StateType.TASK
        )
    
    state_manager.cleanup_state(max_history_entries=100)
    
    # Verify only most recent 100 entries are kept
    assert len(state_manager._version_history["test_id"]) == 100
    assert state_manager._version_history["test_id"][0].data["count"] == 50

@pytest.mark.asyncio
async def test_empty_queue(state_manager):
    # Test handling of empty queue
    messages = await state_manager.dequeue_messages("nonexistent_queue")
    assert messages == []

@pytest.mark.asyncio
async def test_unsubscribe(state_manager):
    # Test unsubscription from state changes
    notifications = []
    
    async def test_callback(state_entry):
        notifications.append(state_entry)
    
    await state_manager.subscribe_to_state("test_id", test_callback)
    await state_manager.unsubscribe_from_state("test_id", test_callback)
    
    await state_manager.update_state(
        "test_id",
        {"key": "value"},
        StateType.TASK
    )
    
    assert len(notifications) == 0

@pytest.mark.asyncio
async def test_multiple_subscribers(state_manager):
    # Test multiple subscribers to same state
    notifications1 = []
    notifications2 = []
    
    async def callback1(state_entry):
        notifications1.append(state_entry)
        
    async def callback2(state_entry):
        notifications2.append(state_entry)
    
    await state_manager.subscribe_to_state("test_id", callback1)
    await state_manager.subscribe_to_state("test_id", callback2)
    
    await state_manager.update_state(
        "test_id",
        {"key": "value"},
        StateType.TASK
    )
    
    assert len(notifications1) == 1
    assert len(notifications2) == 1
