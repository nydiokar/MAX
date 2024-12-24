# test_task_expert_agent_in_memory.py
import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from datetime import datetime, timedelta

from MAX.agents.task_expert.task_expert import TaskExpertAgent
from MAX.agents.task_expert.options import TaskExpertOptions
from MAX.agents.task_expert.errors import (
    TaskExpertError, ValidationError, StorageError,
    LLMError, DependencyError, MonitoringError
)

from tests.storage.test_in_memory_task_storage import InMemoryTaskStorage, InMemoryNotificationService

#########################
# Mock LLM Implementation
#########################
async def mock_llm_generate(prompt: str) -> str:
    """
    A basic mock LLM generate function that returns valid JSON. 
    You can adjust the returned text to simulate different LLM behaviors.
    """
    # Return an empty tasks array by default, can be changed per test
    return '{"tasks": [], "monitoring_requirements": {}}'


@pytest.fixture
def real_agent_options():
    """
    Pytest fixture to create a real TaskExpertOptions with in-memory storage and notifications.
    Mocks the LLM so no real calls are made.
    """
    with patch("MAX.llms.create_llm_provider") as mock_create_llm:
        mock_create_llm.return_value.generate = mock_llm_generate

        # Build your in-memory storage and notifications
        storage = InMemoryTaskStorage()
        notifications = InMemoryNotificationService()

        # Return real TaskExpertOptions
        yield TaskExpertOptions(
            storage_client=storage,
            notification_service=notifications,
            name="TestAgent",
            description="An agent used in real in-memory tests",
            model_id="mock-model-id",
            fallback_model_id=None,
            temperature=0.7,
            max_tokens=2000,
            top_p=0.9,
            default_task_ttl=7
        )


@pytest.mark.asyncio
async def test_empty_input_raises_validation_error(real_agent_options):
    """Ensure an empty user request raises ValidationError."""
    agent = TaskExpertAgent(real_agent_options)

    # Attempt to process an empty input text
    with pytest.raises(ValidationError) as exc_info:
        await agent.process_request("", {})

    assert "Input text cannot be empty" in str(exc_info.value)


@pytest.mark.asyncio
async def test_llm_error_handling(real_agent_options):
    """Simulate an LLM error and ensure it's handled as TaskExpertError."""
    agent = TaskExpertAgent(real_agent_options)

    # Mock the error_handler to raise an LLMError
    async def mock_llm_operation(*args, **kwargs):
        raise LLMError("Simulated LLM failure", "mock-model-id")

    agent.error_handler.handle_llm_operation = mock_llm_operation

    # Attempt to process a request
    with pytest.raises(TaskExpertError) as exc_info:
        await agent.process_request("some input", {})

    assert "Failed to process request" in str(exc_info.value)


@pytest.mark.asyncio
async def test_storage_error_handling(real_agent_options):
    """Simulate a StorageError and ensure it's handled as TaskExpertError."""
    agent = TaskExpertAgent(real_agent_options)

    # Patch storage operation to raise StorageError
    async def mock_storage_operation(*args, **kwargs):
        raise StorageError("Simulated storage failure", "mock_operation")

    agent.error_handler.handle_storage_operation = mock_storage_operation

    # Mock LLM analysis to return some tasks
    async def mock_llm_operation(*args, **kwargs):
        return {"tasks": [{"type": "agent_task", "details": {"title": "Test"}}]}

    agent.error_handler.handle_llm_operation = mock_llm_operation

    # Attempt to process a request
    with pytest.raises(TaskExpertError) as exc_info:
        await agent.process_request("valid input", {})

    assert "Storage operation failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_dependency_error(real_agent_options):
    """Simulate a DependencyError when verifying dependencies."""
    agent = TaskExpertAgent(real_agent_options)

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
        raise DependencyError("Simulated dependency failure", "some_task_id", "some-nonexistent-task")

    agent.error_handler.handle_llm_operation = mock_llm_operation
    agent.error_handler.handle_storage_operation = mock_storage_operation
    agent.error_handler.verify_dependencies = mock_verify_dependencies

    # Attempt to process a request
    response = await agent.process_request("task creation with dependency", {})
    # The task creation should fail gracefully and log a warning, 
    # but no exception is raised since we continue on dependency errors.
    assert response["tasks_created"] == []


@pytest.mark.asyncio
async def test_monitoring_error(real_agent_options):
    """Simulate a MonitoringError during monitoring setup."""
    agent = TaskExpertAgent(real_agent_options)

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
        raise MonitoringError("Simulated monitoring failure", {"extra": "some detail"})

    agent.error_handler.handle_llm_operation = mock_llm_operation
    agent.error_handler.handle_storage_operation = mock_storage_operation
    agent._setup_monitoring = mock_setup_monitoring

    response = await agent.process_request("Monitor this task", {})
    # Ensure monitoring_setup indicates a partial result
    assert response["monitoring_setup"] == "partial"


@pytest.mark.asyncio
async def test_validation_error_due_date_in_past(real_agent_options):
    """Ensure that a past due date raises a ValidationError."""
    agent = TaskExpertAgent(real_agent_options)

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
async def test_validation_error_invalid_priority(real_agent_options):
    """Ensure that an invalid priority raises a ValidationError."""
    agent = TaskExpertAgent(real_agent_options)

    async def mock_llm_operation(*args, **kwargs):
        return {
            "tasks": [
                {
                    "type": "human_task",
                    "details": {
                        "title": "Task with invalid priority",
                        "priority": "IMPOSSIBLE_PRIORITY"
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
async def test_validation_error_empty_title(real_agent_options):
    """Ensure that an empty title triggers a ValidationError."""
    agent = TaskExpertAgent(real_agent_options)

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
async def test_unexpected_exception_handling(real_agent_options):
    """Simulate an unexpected exception during task creation."""
    agent = TaskExpertAgent(real_agent_options)

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


@pytest.mark.asyncio
async def test_valid_scenario(real_agent_options):
    """
    Ensure that a valid scenario with correct data does not raise any error 
    and tasks are created successfully.
    """
    agent = TaskExpertAgent(real_agent_options)

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
