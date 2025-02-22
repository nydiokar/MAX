# src/tests/conftest.py
import os
import sys
import pytest
import asyncio
from .test_utils import reporter, TEST_CONFIG
from unittest.mock import patch, MagicMock, AsyncMock, Mock
from typing import Dict, Any
from _pytest.terminal import TerminalReporter
from tests.utils.test_helpers import (
    MockChatStorage,
    create_mock_team_registry,
    create_recursive_thinker_agent
)
from MAX.agents import RecursiveThinkerAgent, RecursiveThinkerOptions
from MAX.types import ConversationMessage
# Add the src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )

def pytest_sessionstart(session):
    """Called before test session starts."""
    reporter.test_results = []  # Reset results at start
    print("\n=== Starting Test Session ===")

def pytest_sessionfinish(session, exitstatus):
    """Print final summary after all tests complete."""
    reporter.print_summary()

@pytest.fixture(autouse=True)
def verify_test_environment():
    """Mock the LLM provider for testing"""
    with patch('MAX.llms.anthropic.AnthropicLLM') as mock_llm:
        mock_client = Mock()
        mock_client.generate.return_value = ConversationMessage(
            role="assistant",
            content=[{"text": "Test response"}]
        )
        mock_llm.return_value = mock_client
        yield mock_client

@pytest.fixture
async def mock_db():
    """Mock database operations."""
    mock_collection = Mock()
    mock_collection.find_one.return_value = {
        'context': 'Mocked context data',
        'facts': [
            {
                'content': {'text': 'Mocked fact'},
                'metadata': {'confidence': 0.9}
            }
        ]
    }
    
    with patch('MAX.kpu.KPU._get_db_collection', return_value=mock_collection):
        yield mock_collection

@pytest.fixture
def mock_storage():
    """Provide a mock storage system"""
    with patch('MAX.storage.ChatStorage') as mock_storage:
        mock_instance = Mock()
        mock_storage.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_team_registry():
    """Provide a mock team registry"""
    return create_mock_team_registry()

@pytest.fixture
def recursive_thinker_options():
    """Fixture for recursive thinker options"""
    return RecursiveThinkerOptions(
        name="Test Recursive Thinker",
        description="A test recursive thinker agent",
        max_recursion_depth=3,
        min_confidence_threshold=0.7,
        streaming=False
    )

@pytest.fixture
def recursive_thinker_agent(mock_storage):
    """Provide a pre-configured recursive thinker agent"""
    return create_recursive_thinker_agent(
        storage=mock_storage
    )

@pytest.fixture
async def mock_llm():
    """Provide a mock LLM for testing"""
    mock = AsyncMock()
    mock.generate = AsyncMock(return_value="Test response")
    mock.streaming = False
    mock.api_key = "test-key"
    return mock

def pytest_terminal_summary(terminalreporter: TerminalReporter, exitstatus: int, config: Any) -> None:
    """Add custom summary information to the test report."""
    print("\n=== Test Results ===")
    
    passed = len([i for i in terminalreporter.stats.get('passed', [])])
    failed = len([i for i in terminalreporter.stats.get('failed', [])])
    errors = len([i for i in terminalreporter.stats.get('error', [])])
    
    print(f"\nTest Summary:")
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print(f"! Errors: {errors}")
    
    if passed + failed + errors > 0:
        success_rate = (passed / (passed + failed + errors)) * 100
        print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    if failed > 0 or errors > 0:
        print("\nFailed Tests:")
        for report in terminalreporter.stats.get('failed', []):
            print(f"  ✗ {report.nodeid}")
            if hasattr(report, 'longrepr'):
                print(f"    Error: {str(report.longrepr)}\n")

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Customize test result reporting."""
    outcome = yield
    report = outcome.get_result()
    
    if report.when == "call":
        print(f"\n{'='*80}")
        print(f"Test: {item.name}")
        print(f"Status: {'✓ PASSED' if report.passed else '✗ FAILED'}")
        
        if hasattr(item, 'funcargs'):
            try:
                doc = str(item.function.__doc__)
                if doc:
                    print(f"Description: {doc.strip()}")
            except:
                pass
        
        if not report.passed:
            if hasattr(report, 'longrepr'):
                print(f"Error: {str(report.longrepr)}")
        print(f"{'='*80}")

@pytest.fixture
def task_expert_options():
    options = MagicMock()
    options.name = "TestAgent"
    options.description = "A test agent"
    options.model_id = "some-model"
    options.storage_client = AsyncMock()
    options.notification_service = AsyncMock()
    return options

@pytest.fixture
def mock_memory_system():
    """Provide a mock memory system"""
    with patch('MAX.managers.memory_manager.MemorySystem') as mock_memory:
        mock_instance = Mock()
        mock_memory.return_value = mock_instance
        yield mock_instance
