import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, MagicMock, patch
from MAX.agents.task_expert.task_expert import TaskExpertAgent, TaskExpertOptions
from MAX.storage.TaskStorageMongoDB import MongoDBTaskStorage
from MAX.storage.utils.types import TaskStatus, Task, TaskPriority

@pytest.fixture
def mock_storage():
    storage = Mock()
    storage.save_task = AsyncMock()
    storage.update_task = AsyncMock()
    storage.get_task = AsyncMock()
    storage.get_tasks = AsyncMock()
    storage.get_prioritized_tasks = AsyncMock()
    return storage

@pytest.fixture
def mock_notification():
    notification = Mock()
    notification.send = AsyncMock()
    notification.send_bulk = AsyncMock()
    return notification

@pytest.fixture
def task_expert_agent(mock_storage, mock_notification):
    options = TaskExpertOptions(
        storage_client=mock_storage,
        notification_service=mock_notification,
        max_retries=1,
        retry_delay=0,
        default_task_ttl=3600
    )
    return TaskExpertAgent(options=options)

async def test_priority_calculation(task_expert_agent):
    # Test base priority scores
    task_data = {
        "priority": TaskPriority.HIGH,
        "due_date": (datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
        "dependencies": ["task-1", "task-2"]
    }
    kpu_context = {"importance_signals": ["urgent", "critical"]}
    
    score = await task_expert_agent.calculate_task_priority(task_data, kpu_context)
    
    # Test due date impact
    urgent_task = dict(task_data)
    urgent_task["due_date"] = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    urgent_score = await task_expert_agent.calculate_task_priority(urgent_task, kpu_context)
    
    assert urgent_score > score, f"Expected urgent score ({urgent_score}) to be higher than regular score ({score})"

async def test_task_queue_reordering(task_expert_agent):
    tasks = [
        Task(
            id="1",
            task_id="task-1",
            title="Low Priority Task",
            description="Test description",
            assigned_agent="agent-1",
            created_by="user-1",
            priority=TaskPriority.LOW,
            dependencies=[],
            status=TaskStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc),
            due_date=datetime.now(timezone.utc) + timedelta(days=7)
        ),
        Task(
            id="2",
            task_id="task-2",
            title="High Priority Task",
            description="Test description",
            assigned_agent="agent-1",
            created_by="user-1",
            priority=TaskPriority.HIGH,
            dependencies=["task-1"],
            status=TaskStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc),
            due_date=datetime.now(timezone.utc) + timedelta(days=1)
        ),
        Task(
            id="3",
            task_id="task-3",
            title="Medium Priority Task",
            description="Test description",
            assigned_agent="agent-1",
            created_by="user-1",
            priority=TaskPriority.MEDIUM,
            dependencies=[],
            status=TaskStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc),
            due_date=datetime.now(timezone.utc)
        )
    ]
    
    reordered = await task_expert_agent.reorder_task_queue(tasks)
    assert len(reordered) == 3
    assert reordered[0].priority == TaskPriority.HIGH  # Highest priority should be first

async def test_storage_prioritization(mock_storage):
    # Create a mock MongoDB client that supports subscripting
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_db = MagicMock()
    
    # Setup the mock chain
    mock_client.__getitem__.return_value = mock_db
    mock_db.__getitem__.return_value = mock_collection
    
    # Create storage instance with mock client
    storage = MongoDBTaskStorage(client=mock_client, db_name="test")
    
    # Mock the tasks as dictionaries (like real MongoDB would return)
    mock_tasks = [
        {
            "id": "1",
            "task_id": "task-1",
            "title": "Task 1",
            "description": "Test description",
            "assigned_agent": "agent-1",
            "created_by": "user-1",
            "priority": TaskPriority.HIGH.value,  # Use .value for enum
            "priority_score": 90,
            "dependencies": [],
            "status": TaskStatus.PENDING.value,  # Use .value for enum
            "created_at": datetime.now(timezone.utc),
            "last_updated": datetime.now(timezone.utc),
            "due_date": datetime.now(timezone.utc) + timedelta(days=1),
            "metadata": {}
        },
        {
            "id": "2",
            "task_id": "task-2",
            "title": "Task 2",
            "description": "Test description 2",
            "assigned_agent": "agent-1",
            "created_by": "user-1",
            "priority": TaskPriority.MEDIUM.value,  # Use .value for enum
            "priority_score": 60,
            "dependencies": [],
            "status": TaskStatus.PENDING.value,  # Use .value for enum
            "created_at": datetime.now(timezone.utc),
            "last_updated": datetime.now(timezone.utc),
            "due_date": datetime.now(timezone.utc) + timedelta(days=2),
            "metadata": {}
        }
    ]
    
    # Setup mock return value
    mock_collection.find.return_value.sort.return_value.limit.return_value = mock_tasks
    
    # Test priority-based querying
    tasks = await storage.get_prioritized_tasks(
        filters={"status": TaskStatus.PENDING},
        limit=5
    )
    
    # Verify order and completeness
    assert len(tasks) == 2
    assert tasks[0].priority_score >= tasks[1].priority_score
    assert all(isinstance(task.due_date, datetime) for task in tasks)
    assert all(task.description for task in tasks)

# Helper class for async mocks
class AsyncMock(Mock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)