# src/tests/conftest.py
import os
import sys
import pytest
import asyncio
from .test_utils import reporter, TEST_CONFIG

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
async def test_timeout():
    """Global timeout for all async tests."""
    try:
        # Use timeout from config
        async with asyncio.timeout(TEST_CONFIG["timeouts"]["default"]):
            yield
            await asyncio.sleep(0)  # Allow other tasks to complete
    except asyncio.TimeoutError:
        print(f"Test timed out after {TEST_CONFIG['timeouts']['default']} seconds")
        raise
    except Exception as e:
        print(f"Test error: {e}")
        raise

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