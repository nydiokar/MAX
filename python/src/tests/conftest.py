# src/tests/conftest.py
import os
import sys
import pytest
import asyncio
from .test_utils import reporter, TEST_CONFIG
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, Any
from _pytest.terminal import TerminalReporter

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

@pytest.fixture
async def mock_db():
    """Mock database operations."""
    with patch('MAX.kpu.KPU._get_db_collection') as mock_collection:
        mock_collection.return_value.find_one.return_value = {
            'context': 'Mocked context data',
            'facts': [
                {
                    'content': {'text': 'Mocked fact'},
                    'metadata': {'confidence': 0.9}
                }
            ]
        }
        yield mock_collection

@pytest.fixture(autouse=True)
def verify_test_environment():
    """Verify test environment before running tests."""
    # Check if MongoDB is running
    try:
        import pymongo
        client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
        client.server_info()  # Will raise exception if cannot connect
        print("MongoDB connection verified")
    except Exception as e:
        pytest.skip(f"MongoDB not available: {e}")

def pytest_terminal_summary(terminalreporter: TerminalReporter, exitstatus: int, config: Any) -> None:
    """Add custom summary information to the test report."""
    print("\n=== KPU Test Results ===")
    
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
            # Add the error message
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
                # Get the docstring of the test function
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
