import pytest
import asyncio
from datetime import datetime, timezone
import os
import shutil
from typing import Optional
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import InvalidOperation
from pydantic import ValidationError
from MAX.storage.TaskStorageMongoDB import MongoDBTaskStorage, TaskStorageError
from MAX.storage.utils.types import Task, TaskStatus, TaskPriority
from MAX.storage.data_hub_cache import DataHubCache, DataHubCacheConfig
from MAX.config.database_config import DatabaseConfig
from MAX.utils import Logger

# Constants for testing
TEST_DB_URI = os.getenv("TEST_MONGODB_URI", "mongodb://localhost:27017")
TEST_DB_NAME = os.getenv("TEST_DB_NAME", "test_task_state_db")

def is_test_db(uri: str, db_name: str) -> bool:
    """Safety check to ensure we're not running tests on production."""
    return "test" in db_name.lower() or "test" in uri.lower()

@pytest.fixture(scope="session")
def event_loop():
    """Create and yield an event loop for pytest-asyncio."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def verify_test_db():
    """Verify we're using a test database before running tests."""
    if not is_test_db(TEST_DB_URI, TEST_DB_NAME):
        pytest.fail("Tests must be run against a test database! Check TEST_DB_NAME and TEST_DB_URI")

@pytest.fixture(scope="function")
async def mongodb_client(verify_test_db):
    """Create and yield a MongoDB client with safety checks."""
    client = None
    try:
        client = AsyncIOMotorClient(
            TEST_DB_URI,
            serverSelectionTimeoutMS=5000
        )
        # Test connection
        await client.admin.command('ping')
        
        # Additional safety check
        if not is_test_db(TEST_DB_URI, TEST_DB_NAME):
            raise ValueError("Refusing to run tests on non-test database!")
            
        yield client
        
    except Exception as e:
        pytest.skip(f"MongoDB not available: {str(e)}")
    finally:
        if client:
            await cleanup_collections(client)  # Clean collections before closing
            client.close()

async def cleanup_collections(client):
    """Helper function to clean up collections."""
    try:
        db = client[TEST_DB_NAME]
        await db.tasks.delete_many({})
        await db.execution_history.delete_many({})
    except Exception as e:
        Logger.error(f"Error during collection cleanup: {str(e)}")

@pytest.fixture(scope="function")
async def data_hub_cache(mongodb_client):
    """Create and yield a DataHubCache instance."""
    config = DataHubCacheConfig(
        message_cache_ttl=300,
        state_cache_ttl=60,
        max_cached_sessions=100,
        enable_vector_cache=True
    )
    
    # Create DatabaseConfig with proper initialization
    db_config = DatabaseConfig()
    # Set MongoDB configuration
    db_config.mongodb.uri = TEST_DB_URI
    db_config.mongodb.database = TEST_DB_NAME
    db_config.mongodb.state_collection = "test_system_state"
    db_config.mongodb.chat_collection = "test_chat_history"
    
    # Set ChromaDB configuration
    db_config.chromadb.persist_directory = "./test_chroma_db"
    db_config.chromadb.collection_name = "test_semantic_store"
    
    # Set StateManager configuration
    db_config.state_manager.enable_vector_storage = True
    db_config.state_manager.state_cleanup_interval_hours = 1
    db_config.state_manager.max_state_age_hours = 24
    db_config.state_manager.max_conversation_history = 100
    
    cache = DataHubCache(config=config, db_config=db_config)
    yield cache

@pytest.fixture(scope="function")
async def task_storage(mongodb_client, data_hub_cache):
    """Create and yield a MongoDBTaskStorage instance with cache."""
    storage = MongoDBTaskStorage(
        client=mongodb_client,
        db_name=TEST_DB_NAME,
        cache=data_hub_cache
    )
    
    try:
        # Setup and verify storage
        initialized = await storage.initialize()
        if not initialized:
            pytest.fail("Failed to initialize task storage")
        
        yield storage
        
    except Exception as e:
        pytest.fail(f"Failed to setup task storage: {str(e)}")

@pytest.fixture(autouse=True, scope="function")
async def cleanup_chromadb():
    """Clean up the test ChromaDB directory before and after each test."""
    test_chroma_dir = "./test_chroma_db"
    if os.path.exists(test_chroma_dir):
        shutil.rmtree(test_chroma_dir)
    yield
    if os.path.exists(test_chroma_dir):
        shutil.rmtree(test_chroma_dir)

def validate_task_fields(task: Task) -> bool:
    """Validate that a task has all required fields."""
    required_fields = {
        'task_id', 'title', 'description', 'status', 'priority',
        'assigned_agent', 'created_at', 'due_date',
        'last_updated', 'created_by'
    }
    
    missing_fields = required_fields - set(dir(task))
    if missing_fields:
        print(f"Missing fields: {missing_fields}")
        return False
    return True

def create_test_task(**kwargs) -> Task:
    """Helper function to create a test task with all required fields."""
    try:
        now = datetime.now(timezone.utc)
        default_values = {
            "task_id": str(uuid4()),
            "title": "Test Task",
            "description": "Test Description",
            "status": TaskStatus.PENDING,
            "priority": TaskPriority.MEDIUM,
            "assigned_agent": "TestAgent",
            "created_at": now,
            "updated_at": now,
            "due_date": now,
            "last_updated": now,
            "created_by": "test_user",
            "metadata": {},
            "progress": 0.0,
            "dependencies": [],
            "tags": [],
            "estimated_hours": None
        }
        # Override defaults with any provided values
        default_values.update(kwargs)
        task = Task(**default_values)
        
        # Debug output
        print(f"Created task with fields: {task.__dict__.keys()}")
        if not validate_task_fields(task):
            raise ValueError(f"Task validation failed. Task fields: {task.__dict__.keys()}")
            
        return task
    except Exception as e:
        raise ValueError(f"Failed to create task: {str(e)}")

@pytest.mark.asyncio
async def test_task_lifecycle(task_storage: MongoDBTaskStorage, data_hub_cache: DataHubCache):
    """Test complete task lifecycle with caching."""
    
    # 1. Create Task
    try:
        # Create initial task
        new_task = create_test_task(
            title="Test Task",
            description="Test task description",
            metadata={"test_key": "test_value"}
        )
        
        task_id = await task_storage.create_task(new_task)
        assert task_id is not None
        
        # Verify cache miss on first fetch
        cache_stats_before = data_hub_cache.get_cache_stats()
        cached_task = await data_hub_cache.get_cached_task(task_id)
        assert cached_task is None, "Task should not be in cache yet"
        
        # 2. Fetch and verify task
        fetched_task = await task_storage.fetch_task(task_id)
        assert fetched_task is not None, "Task fetch failed"
        assert fetched_task.title == new_task.title
        assert fetched_task.status == TaskStatus.PENDING
        
        # 3. Update task
        fetched_task.status = TaskStatus.IN_PROGRESS
        fetched_task.progress = 0.5
        update_success = await task_storage.update_task(task_id, fetched_task)
        assert update_success, "Task update failed"
        
        # Verify cache invalidation
        cached_task = await data_hub_cache.get_cached_task(task_id)
        assert cached_task is None, "Cache should be invalidated after update"
        
        # 4. Verify update
        updated_task = await task_storage.fetch_task(task_id)
        assert updated_task.status == TaskStatus.IN_PROGRESS
        assert updated_task.progress == 0.5
        
        # 5. Check execution history
        history = await task_storage.fetch_execution_history(task_id)
        assert len(history) > 0, "Execution history should be recorded"
        assert history[0].status == TaskStatus.IN_PROGRESS.value
        
        # 6. Test dependencies
        dependent_task = create_test_task(
            title="Dependent Task",
            description="Depends on the first task",
            priority=TaskPriority.LOW
        )
        
        dependent_id = await task_storage.create_task(dependent_task)
        assert dependent_id is not None
        
        # Add dependency
        added = await task_storage.add_task_dependency(dependent_id, task_id)
        assert added, "Failed to add dependency"
        
        # Check for cycles
        has_cycle = await task_storage.check_dependency_cycle(task_id, dependent_id)
        assert not has_cycle, "Dependency cycle detected"
        
    except TaskStorageError as e:
        pytest.fail(f"Task storage error: {str(e)}")
    except Exception as e:
        pytest.fail(f"Unexpected error: {str(e)}")

@pytest.mark.asyncio
async def test_task_search(task_storage: MongoDBTaskStorage):
    """Test task search functionality."""
    try:
        # Create multiple tasks with different priorities but same agent
        now = datetime.now(timezone.utc)
        task_ids = []
        
        # Create test tasks
        for i, priority in enumerate([TaskPriority.HIGH, TaskPriority.MEDIUM, TaskPriority.LOW]):
            task = Task(
                task_id=f"search-task-{i}",
                title=f"Test Task {i}",
                description=f"Description {i}",
                status=TaskStatus.PENDING,
                priority=priority,
                assigned_agent="SearchAgent",
                created_at=now,
                updated_at=now,
                due_date=now,
                last_updated=now,
                created_by="test_user",
                metadata={},
                progress=0,
                dependencies=[],
                tags=[]
            )
            task_id = await task_storage.create_task(task)
            task_ids.append(task_id)
        
        # Verify tasks creation
        assert len(task_ids) == 3, f"Expected 3 tasks, but created {len(task_ids)}"
        
        # Test 1: Search by priority
        filter_task = Task(
            task_id="",
            title="",
            description="",
            status=TaskStatus.PENDING,
            priority=TaskPriority.HIGH,
            assigned_agent="",
            created_at=now,
            updated_at=now,
            due_date=now,
            last_updated=now,
            created_by="test_user",
            metadata={},
            progress=0,
            dependencies=[],
            tags=[]
        )
        
        high_priority_tasks = await task_storage.search_tasks(filters=filter_task)
        
        # Verify high priority search results
        assert len(high_priority_tasks) == 1, "Expected 1 high priority task"
        assert all(t.priority == TaskPriority.HIGH for t in high_priority_tasks)
        
        # Test 2: Search by agent
        filter_task = Task(
            task_id="",
            title="",
            description="",
            status=TaskStatus.PENDING,
            priority=TaskPriority.MEDIUM,
            assigned_agent="SearchAgent",
            created_at=now,
            updated_at=now,
            due_date=now,
            last_updated=now,
            created_by="test_user",
            metadata={},
            progress=0,
            dependencies=[],
            tags=[]
        )
        
        agent_tasks = await task_storage.search_tasks(filters=filter_task)
        
        # Debug output
        print(f"\nFound {len(agent_tasks)} tasks for agent SearchAgent:")
        for task in agent_tasks:
            print(f"Task: {task.title} - {task.priority} - {task.assigned_agent}")
        
        # Verify agent search results
        assert len(agent_tasks) == 3, f"Expected 3 tasks for SearchAgent, but found {len(agent_tasks)}"
        assert all(t.assigned_agent == "SearchAgent" for t in agent_tasks)
        
    except Exception as e:
        print(f"\nDetailed error information:")
        print(f"Task IDs created: {task_ids}")
        print(f"Exception: {str(e)}")
        pytest.fail(f"Task search error: {str(e)}")

@pytest.mark.asyncio
async def test_error_handling(task_storage: MongoDBTaskStorage):
    """Test error handling scenarios."""
    try:
        # Test invalid task creation (missing required fields)
        with pytest.raises((TaskStorageError, ValidationError)):
            invalid_task = Task(  # Create invalid task directly without helper
                task_id=str(uuid4()),
                title="",  # Empty title should trigger validation
                description="Test Description",
                status=TaskStatus.PENDING,
                priority=TaskPriority.MEDIUM,
                assigned_agent=""  # Empty agent should also trigger validation
            )
            await task_storage.create_task(invalid_task)
        
        # Test updating non-existent task
        valid_task = create_test_task(
            title="Non-existent",
            description="This task doesn't exist"
        )
        
        update_result = await task_storage.update_task("non_existent_id", valid_task)
        assert not update_result, "Update should fail for non-existent task"
        
    except Exception as e:
        pytest.fail(f"Unexpected error in error handling test: {str(e)}")
