import pytest
from unittest.mock import AsyncMock
from datetime import datetime, timezone, timedelta
from MAX.agents.task_expert.task_tool_registry import TaskToolRegistry
from MAX.storage.utils.types import Task, TaskStatus, TaskPriority
from MAX.agents.task_expert.task_tool_registry import (
    TaskToolRegistry,
    CreateTaskTool,
    UpdateTaskTool,
    DeleteTaskTool,
    TaskOutput,
    CreateTaskInput,
    UpdateTaskInput,
    DeleteTaskInput
)

class MockStorage:
    def __init__(self):
        self.create_task = AsyncMock()
        self.fetch_task = AsyncMock()
        self.update_task = AsyncMock()
        self.delete_task = AsyncMock()
        self.search_tasks = AsyncMock()

@pytest.fixture
def storage():
    return MockStorage()

@pytest.fixture
def registry(storage):
    class TestableTaskToolRegistry(TaskToolRegistry):
        def __init__(self, storage):
            super().__init__()
            self.storage = storage
            # Override tool registration to inject storage
            self._register_task_tools()
        
        def _register_task_tools(self):
            self.register(CreateTaskTool(
                name="create_task",
                description="Create a new task",
                category="SYSTEM",
                input_schema=CreateTaskInput,
                output_schema=TaskOutput,
                storage=self.storage
            ))
            
            self.register(UpdateTaskTool(
                name="update_task",
                description="Update an existing task",
                category="SYSTEM",
                input_schema=UpdateTaskInput,
                output_schema=TaskOutput,
                storage=self.storage
            ))
            
            self.register(DeleteTaskTool(
                name="delete_task",
                description="Delete a task",
                category="SYSTEM",
                input_schema=DeleteTaskInput,
                output_schema=TaskOutput,
                storage=self.storage
            ))
    
    return TaskToolRegistry(storage=storage) 

@pytest.mark.asyncio
class TestTaskToolRegistry:
    async def test_create_task_success(self, registry, storage):
        # Setup
        task_id = "test-123"
        storage.create_task.return_value = task_id
        storage.fetch_task.return_value = Task(
            task_id=task_id,
            title="Test Task",
            description="Test Description",
            status=TaskStatus.PENDING,
            priority=TaskPriority.MEDIUM,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc),
            assigned_agent="test-agent",
            due_date=datetime.now(timezone.utc) + timedelta(days=7),
            created_by="test-user"
        )

        # Execute
        result = await registry.execute("create_task", {
            "title": "Test Task",
            "description": "Test Description",
            "priority": TaskPriority.MEDIUM
        })

        assert result["task"].task_id == task_id
        assert result["task"].title == "Test Task"
        storage.create_task.assert_called_once()

    async def test_update_task_success(self, registry, storage):
        task_id = "test-123"
        storage.fetch_task.return_value = Task(
            task_id=task_id,
            title="Test Task",
            description="Test Description",
            status=TaskStatus.PENDING,
            priority=TaskPriority.MEDIUM,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc),
            assigned_agent="test-agent",
            due_date=datetime.now(timezone.utc) + timedelta(days=7),
            created_by="test-user"
        )
        storage.update_task.return_value = True

        result = await registry.execute("update_task", {
            "task_id": task_id,
            "status": TaskStatus.IN_PROGRESS
        })

        assert result["task"].task_id == task_id
        storage.update_task.assert_called_once()

    async def test_delete_task_success(self, registry, storage):
        task_id = "test-123"
        storage.fetch_task.return_value = Task(
            task_id=task_id,
            title="Test Task",
            description="Test Description",
            status=TaskStatus.COMPLETED,
            priority=TaskPriority.MEDIUM,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc),
            assigned_agent="test-agent",
            due_date=datetime.now(timezone.utc) + timedelta(days=7),
            created_by="test-user"
        )
        storage.search_tasks.return_value = []
        storage.delete_task.return_value = True

        result = await registry.execute("delete_task", {
            "task_id": task_id
        })

        assert result["success"] is True
        storage.delete_task.assert_called_once_with(task_id)

    async def test_task_lifecycle(self, registry, storage):
        task_id = "lifecycle-123"
        
        # Setup mock responses
        storage.create_task.return_value = task_id
        storage.fetch_task.return_value = Task(
            task_id=task_id,
            title="Lifecycle Test",
            description="Testing lifecycle",
            status=TaskStatus.PENDING,
            priority=TaskPriority.MEDIUM,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc),
            assigned_agent="test-agent",
            due_date=datetime.now(timezone.utc) + timedelta(days=7),
            created_by="test-user"
        )
        storage.update_task.return_value = True
        storage.delete_task.return_value = True
        storage.search_tasks.return_value = []

        # Create
        create_result = await registry.execute("create_task", {
            "title": "Lifecycle Test",
            "description": "Testing lifecycle",
            "priority": TaskPriority.MEDIUM
        })
        assert create_result["task"].task_id == task_id

        # Update to In Progress
        await registry.execute("update_task", {
            "task_id": task_id,
            "status": TaskStatus.IN_PROGRESS
        })

        # Update to Completed
        await registry.execute("update_task", {
            "task_id": task_id,
            "status": TaskStatus.COMPLETED
        })

        # Delete
        delete_result = await registry.execute("delete_task", {
            "task_id": task_id
        })
        assert delete_result["success"] is True

    async def test_invalid_tool(self, registry):
        with pytest.raises(ValueError):
            await registry.execute("nonexistent_tool", {})

if __name__ == "__main__":
    pytest.main([__file__, "-v"])