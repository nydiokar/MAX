import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from MAX.agents.task_expert.task_expert import TaskExpertAgent
from MAX.agents.task_expert.errors import (
    TaskExpertError, ValidationError, StorageError,
    LLMError, DependencyError, MonitoringError
)
from MAX.storage.utils.types import TaskPriority


@pytest.mark.asyncio
async def test_empty_input_raises_validation_error():
    """Ensure an empty user request raises ValidationError."""
    # Mock options
    options = MagicMock()
    options.name = "TestAgent"
    options.description = "A test agent"
    options.model_id = "some-model"
    options.storage_client = AsyncMock()
    options.notification_service = AsyncMock()
    agent = TaskExpertAgent(options)

    # Attempt to process an empty input text
    with pytest.raises(ValidationError) as exc_info:
        await agent.process_request("", {})

    assert "Input text cannot be empty" in str(exc_info.value)


@pytest.mark.asyncio
async def test_llm_error_handling():
    """Simulate an LLM error and ensure it's handled as TaskExpertError."""
    options = MagicMock()
    options.name = "TestAgent"
    options.description = "A test agent"
    options.model_id = "some-model"
    options.storage_client = AsyncMock()
    options.notification_service = AsyncMock()
    agent = TaskExpertAgent(options)

    # Mock the LLM operation to raise an error
    async def mock_llm_operation(*args, **kwargs):
        raise LLMError("Failed to process request", model_id="test-model")

    # Replace the error handler's LLM operation
    agent.error_handler.handle_llm_operation = mock_llm_operation

    # Test should raise TaskExpertError
    with pytest.raises(TaskExpertError) as exc_info:
        await agent.process_request("some input", {})
    assert "Failed to process request" in str(exc_info.value)


@pytest.mark.asyncio
async def test_storage_error_handling():
    """Simulate a StorageError and ensure it's handled as TaskExpertError."""
    options = MagicMock()
    options.name = "TestAgent"
    options.description = "A test agent"
    options.model_id = "some-model"
    options.storage_client = AsyncMock()
    options.notification_service = AsyncMock()
    agent = TaskExpertAgent(options)

    # Mock successful LLM operation with valid task data
    async def mock_llm_operation(*args, **kwargs):
        return {
            "tasks": [
                {
                    "type": "agent_task",
                    "details": {
                        "title": "Test Task",
                        "priority": "LOW",  # Required field
                        "description": "Test description"
                    }
                }
            ]
        }

    # Mock storage operation to raise StorageError
    async def mock_storage_operation(*args, **kwargs):
        raise StorageError("Failed to process request", operation="create_task")

    # Set up the mocks
    agent.error_handler.handle_llm_operation = mock_llm_operation
    agent.error_handler.handle_storage_operation = mock_storage_operation

    # Test should raise TaskExpertError
    with pytest.raises(TaskExpertError) as exc_info:
        await agent.process_request("valid input", {})
    assert "Failed to process request" in str(exc_info.value)

@pytest.mark.asyncio
async def test_dependency_error():
    """Simulate a DependencyError when verifying dependencies."""
    options = MagicMock()
    options.name = "TestAgent"
    options.description = "A test agent"
    options.model_id = "some-model"
    options.storage_client = AsyncMock()
    options.notification_service = AsyncMock()
    agent = TaskExpertAgent(options)

    # Return a normal LLM analysis with dependencies
    async def mock_llm_operation(*args, **kwargs):
        return {
            "tasks": [
                {
                    "type": "agent_task",
                    "details": {
                        "title": "Task with invalid dependency",
                        "dependencies": ["some-nonexistent-task"]
                    }
                }
            ]
        }

    # Normal storage operation but dependency check fails
    async def mock_storage_operation(func, action_name, data):
        return "some_task_id"

    async def mock_verify_dependencies(*args, **kwargs):
        raise DependencyError("Simulated dependency failure")

    agent.error_handler.handle_llm_operation = mock_llm_operation
    agent.error_handler.handle_storage_operation = mock_storage_operation
    agent.error_handler.verify_dependencies = mock_verify_dependencies

    # Attempt to process a request
    response = await agent.process_request("task creation with dependency", {})
    # The task creation should fail gracefully and log a warning, 
    # but no exception is raised since we continue on dependency errors.
    assert response["tasks_created"] == []


@pytest.mark.asyncio
async def test_monitoring_error():
    """Simulate a MonitoringError during monitoring setup."""
    options = MagicMock()
    options.name = "TestAgent"
    options.description = "A test agent"
    options.model_id = "some-model"
    options.storage_client = AsyncMock()
    options.notification_service = AsyncMock()
    agent = TaskExpertAgent(options)

    # Return a normal LLM analysis with tasks
    async def mock_llm_operation(*args, **kwargs):
        return {
            "tasks": [
                {
                    "type": "agent_task",
                    "details": {
                        "title": "Valid Task"
                    }
                }
            ],
            "monitoring_requirements": {"some_task_id": {"extra": "some detail"}}
        }

    async def mock_storage_operation(func, action_name, data):
        return "some_task_id"

    # Force monitoring error
    async def mock_setup_monitoring(*args, **kwargs):
        raise MonitoringError(
            message="Simulated monitoring failure",
            monitoring_config={"task_id": "test-task"}
        )

    agent.error_handler.handle_llm_operation = mock_llm_operation
    agent.error_handler.handle_storage_operation = mock_storage_operation
    agent._setup_monitoring = mock_setup_monitoring

    response = await agent.process_request("Monitor this task", {})
    # Ensure monitoring_setup indicates a partial result
    assert response["monitoring_setup"] == "partial"


@pytest.mark.asyncio
async def test_validation_error_due_date_in_past():
    """Ensure that a past due date raises a ValidationError."""
    options = MagicMock()
    options.name = "TestAgent"
    options.description = "A test agent"
    options.model_id = "some-model"
    options.storage_client = AsyncMock()
    options.notification_service = AsyncMock()
    agent = TaskExpertAgent(options)

    # Return a normal LLM analysis with tasks
    async def mock_llm_operation(*args, **kwargs):
        return {
            "tasks": [
                {
                    "type": "agent_task",
                    "details": {
                        "title": "Valid Title",
                        "due_date": (datetime.now() - timedelta(days=1)).isoformat()
                    }
                }
            ]
        }

    agent.error_handler.handle_llm_operation = mock_llm_operation

    # We’ll mock storage operation to see if it’s not even called due to validation error.
    agent.error_handler.handle_storage_operation = AsyncMock(return_value="some_task_id")

    response = await agent.process_request("Create a past-due task", {})
    # Task creation is expected to fail validation, so no tasks are created.
    assert len(response["tasks_created"]) == 0


@pytest.mark.asyncio
async def test_validation_error_invalid_priority():
    """Ensure that an invalid priority raises a ValidationError."""
    options = MagicMock()
    options.name = "TestAgent"
    options.description = "A test agent"
    options.model_id = "some-model"
    options.storage_client = AsyncMock()
    options.notification_service = AsyncMock()
    agent = TaskExpertAgent(options)

    async def mock_llm_operation(*args, **kwargs):
        return {
            "tasks": [
                {
                    "type": "human_task",
                    "details": {
                        "title": "Task with invalid priority",
                        "priority": "IMPOSSIBLE_PRIORITY"  # not in TaskPriority
                    }
                }
            ]
        }

    agent.error_handler.handle_llm_operation = mock_llm_operation
    agent.error_handler.handle_storage_operation = AsyncMock(return_value="some_task_id")

    response = await agent.process_request("Create a task with an invalid priority", {})
    # Expect zero tasks created due to priority validation error.
    assert len(response["tasks_created"]) == 0


@pytest.mark.asyncio
async def test_validation_error_empty_title():
    """Ensure that an empty title triggers a ValidationError."""
    options = MagicMock()
    options.name = "TestAgent"
    options.description = "A test agent"
    options.model_id = "some-model"
    options.storage_client = AsyncMock()
    options.notification_service = AsyncMock()
    agent = TaskExpertAgent(options)

    async def mock_llm_operation(*args, **kwargs):
        return {
            "tasks": [
                {
                    "type": "human_task",
                    "details": {
                        "title": "",
                        "description": "Some description"
                    }
                }
            ]
        }

    agent.error_handler.handle_llm_operation = mock_llm_operation
    agent.error_handler.handle_storage_operation = AsyncMock(return_value="some_task_id")

    response = await agent.process_request("Create a task with empty title", {})
    # No tasks created because the title is invalid
    assert len(response["tasks_created"]) == 0


@pytest.mark.asyncio
async def test_unexpected_exception_handling():
    """Simulate an unexpected exception during task creation."""
    options = MagicMock()
    options.name = "TestAgent"
    options.description = "A test agent"
    options.model_id = "some-model"
    options.storage_client = AsyncMock()
    options.notification_service = AsyncMock()
    agent = TaskExpertAgent(options)

    async def mock_llm_operation(*args, **kwargs):
        return {
            "tasks": [
                {"type": "agent_task", "details": {"title": "Might cause an unexpected error"}}
            ]
        }

    # Simulate an unknown error in storage operation
    async def mock_storage_operation(func, action_name, data):
        raise RuntimeError("Unknown error")

    agent.error_handler.handle_llm_operation = mock_llm_operation
    agent.error_handler.handle_storage_operation = mock_storage_operation

    response = await agent.process_request("Trigger unexpected error", {})
    # The agent should continue to process but no tasks get created
    assert len(response["tasks_created"]) == 0
    # Confirm that it doesn’t raise an uncaught exception but logs an error instead.


@pytest.mark.asyncio
async def test_valid_scenario():
    """
    Ensure that a valid scenario with correct data does not raise any error and tasks are created successfully.
    """
    options = MagicMock()
    options.name = "TestAgent"
    options.description = "A test agent"
    options.model_id = "some-model"
    options.storage_client = AsyncMock()
    options.notification_service = AsyncMock()
    agent = TaskExpertAgent(options)

    # Mock normal LLM analysis
    async def mock_llm_operation(*args, **kwargs):
        return {
            "tasks": [
                {
                    "type": "human_task",
                    "details": {
                        "title": "A perfectly valid task",
                        "description": "All good here",
                        "priority": "HIGH"
                    }
                }
            ],
            "monitoring_requirements": {}
        }

    async def mock_storage_operation(func, action_name, data):
        return "created_task_id"

    agent.error_handler.handle_llm_operation = mock_llm_operation
    agent.error_handler.handle_storage_operation = mock_storage_operation

    response = await agent.process_request("Create a valid task", {})
    assert len(response["tasks_created"]) == 1
    assert response["tasks_created"][0] == "created_task_id"
    assert response["monitoring_setup"] == "complete"
