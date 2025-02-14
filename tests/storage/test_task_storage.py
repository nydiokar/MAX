import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from MAX.storage.TaskStorageMongoDB import MongoDBTaskStorage
from MAX.storage.utils.types import Task, ExecutionHistoryEntry, TaskStatus, TaskPriority
from motor.motor_asyncio import AsyncIOMotorClient

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def mongodb_client():
    """Create a shared MongoDB client for all tests."""
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    yield client
    client.close()

@pytest.fixture
async def task_storage(mongodb_client):
    """Create a task storage instance with the shared client."""
    storage = MongoDBTaskStorage(mongodb_client, "test_db")
    await storage.initialize()
    
    # Clear any existing data
    await storage.tasks.delete_many({})
    await storage.execution_history.delete_many({})
    
    yield storage
    
    # Clean up test data but don't close the client
    await storage.tasks.delete_many({})
    await storage.execution_history.delete_many({})

@pytest.mark.asyncio
async def test_task_crud_operations(task_storage):
    # Create task
    now = datetime.now(timezone.utc)
    task_data = Task(
        task_id="test-task-1",
        title="Test Task",
        description="Test Description",
        status=TaskStatus.PENDING,
        priority=TaskPriority.HIGH,
        created_at=now,
        updated_at=now,
        assigned_agent="agent-1",
        dependencies=[],
        progress=0,
        metadata={},
        tags=[],
        due_date=now + timedelta(days=7),
        last_updated=now,
        created_by="test-user"
    )
    
    task_id = await task_storage.create_task(task_data)
    assert task_id == "test-task-1"
    
    # Fetch task
    fetched_task = await task_storage.fetch_task(task_id)
    assert fetched_task.title == "Test Task"
    
    # Update task
    update_data = Task(
        task_id="test-task-1",
        title="Test Task",
        description="Test Description",
        status=TaskStatus.IN_PROGRESS,
        priority=TaskPriority.HIGH,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        assigned_agent="agent-1",
        dependencies=[],
        progress=0,
        metadata={},
        tags=[],
        due_date=now + timedelta(days=7),
        last_updated=now,
        created_by="test-user"
    )
    success = await task_storage.update_task(task_id, update_data)
    assert success

@pytest.mark.asyncio
async def test_task_queries(task_storage):
    now = datetime.now(timezone.utc)
    # Create multiple tasks
    for i in range(3):
        task_data = Task(
            task_id=f"task-{i}",
            title=f"Task {i}",
            description=f"Description {i}",
            status=TaskStatus.PENDING,
            priority=TaskPriority.MEDIUM,
            created_at=now,
            updated_at=now,
            assigned_agent="agent-1",
            dependencies=[],
            progress=0,
            metadata={},
            tags=[],
            due_date=now + timedelta(days=7),
            last_updated=now,
            created_by="test-user"
        )
        await task_storage.create_task(task_data)
    
    # Test search with filters
    filter_task = Task(
        task_id="",
        title="",
        description="",
        status=TaskStatus.PENDING,
        priority=TaskPriority.MEDIUM,
        created_at=now,
        updated_at=now,
        assigned_agent="agent-1",
        dependencies=[],
        progress=0,
        metadata={},
        tags=[],
        due_date=now + timedelta(days=7),  # Added required field
        last_updated=now,                   # Added required field
        created_by="test-user"             # Added required field
    )
    results = await task_storage.search_tasks(filters=filter_task)
    assert len(results) == 3

@pytest.mark.asyncio
async def test_validation(task_storage):
    now = datetime.now(timezone.utc)
    task1_data = Task(
        task_id="task-cycle-1",
        title="Task 1",
        description="Test task 1",
        status=TaskStatus.PENDING,
        priority=TaskPriority.LOW,
        created_at=now,
        updated_at=now,
        assigned_agent="agent-1",
        dependencies=[],
        progress=0,
        metadata={},
        tags=[],
        due_date=now + timedelta(days=7),
        last_updated=now,
        created_by="test-user"
    )
    
    task2_data = Task(
        task_id="task-cycle-2",
        title="Task 2",
        description="Test task 2",
        status=TaskStatus.PENDING,
        priority=TaskPriority.LOW,
        created_at=now,
        updated_at=now,
        assigned_agent="agent-1",
        dependencies=[],
        progress=0,
        metadata={},
        tags=[],
        due_date=now + timedelta(days=7),
        last_updated=now,
        created_by="test-user"
    )
    
    await task_storage.create_task(task1_data)
    await task_storage.create_task(task2_data)

@pytest.mark.asyncio
class TestTaskLifecycle:
    """Test cases for task creation, assignment and status updates"""
    
    async def test_create_task(self, task_storage):
        now = datetime.now(timezone.utc)
        task_data = Task(
            task_id="TC-001",
            title="Create new feature",
            description="Implement new feature X",
            status=TaskStatus.PENDING,
            priority=TaskPriority.HIGH,
            assigned_agent="bob",
            dependencies=[],
            progress=0,
            created_at=now,
            updated_at=now,
            metadata={},
            tags=[],
            due_date=now + timedelta(days=7),
            last_updated=now,
            created_by="test-user"
        )
        task_id = await task_storage.create_task(task_data)
        created_task = await task_storage.fetch_task(task_id)
        assert created_task.title == "Create new feature"
        assert created_task.status == TaskStatus.PENDING
        assert created_task.assigned_agent == "bob"

    async def test_reassign_task(self, task_storage):
        now = datetime.now(timezone.utc)
        task_data = Task(
            task_id="TC-002",
            title="Bug fix",
            description="Fix critical bug",
            status=TaskStatus.PENDING,
            priority=TaskPriority.HIGH,
            assigned_agent="alice",
            dependencies=[],
            progress=0,
            created_at=now,
            updated_at=now,
            metadata={},
            tags=[],
            due_date=now + timedelta(days=7),
            last_updated=now,
            created_by="test-user"
        )
        task_id = await task_storage.create_task(task_data)
        
        # Reassign to Bob
        update = task_data.model_copy()
        update.assigned_agent = "bob"
        update.last_updated = datetime.now(timezone.utc)
        await task_storage.update_task(task_id, update)
        
        updated_task = await task_storage.fetch_task(task_id)
        assert updated_task.assigned_agent == "bob"

    async def test_update_status(self, task_storage):
        now = datetime.now(timezone.utc)
        task_data = Task(
            task_id="TC-003",
            title="Documentation",
            description="Update docs",
            status=TaskStatus.PENDING,
            priority=TaskPriority.MEDIUM,
            assigned_agent="charlie",
            dependencies=[],
            progress=0,
            created_at=now,
            updated_at=now,
            metadata={},
            tags=[],
            due_date=now + timedelta(days=7),
            last_updated=now,
            created_by="test-user"
        )
        task_id = await task_storage.create_task(task_data)
        
        # Update to IN_PROGRESS
        update = task_data.model_copy()
        update.status = TaskStatus.IN_PROGRESS
        update.progress = 50
        update.last_updated = datetime.now(timezone.utc)
        await task_storage.update_task(task_id, update)
        
        # Log status change with entry_id
        entry_id = f"entry-{datetime.now(timezone.utc).timestamp()}"
        await task_storage.log_execution_history(ExecutionHistoryEntry(
            entry_id=entry_id,
            task_id=task_id,
            timestamp=datetime.now(timezone.utc),
            status=TaskStatus.IN_PROGRESS.value,
            changed_by="charlie",
            comment="Started work"
        ))
        
        updated_task = await task_storage.fetch_task(task_id)
        assert updated_task.status == TaskStatus.IN_PROGRESS
        assert updated_task.progress == 50
        
        history = await task_storage.fetch_execution_history(task_id)
        assert len(history) == 1
        assert history[0].status == TaskStatus.IN_PROGRESS.value