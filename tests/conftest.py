"""Global pytest configuration"""
import os
import sys
import pytest
from typing import Any

# Add the src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )

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
        
        if not report.passed and hasattr(report, 'longrepr'):
            print(f"Error: {str(report.longrepr)}")
        print(f"{'='*80}")

def pytest_terminal_summary(terminalreporter, exitstatus: int, config: Any) -> None:
    """Add custom summary information to the test report."""
    passed = len([i for i in terminalreporter.stats.get('passed', [])])
    failed = len([i for i in terminalreporter.stats.get('failed', [])])
    errors = len([i for i in terminalreporter.stats.get('error', [])])
    
    print("\n=== Test Results ===")
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
                print(f"    Error: {str(report.longrepr)}")
