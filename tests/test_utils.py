from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import time

# Test configuration
TEST_CONFIG = {
    "performance": {
        "max_write_time": 5.0,
        "max_read_time": 2.0,
        "batch_size": 100
    },
    "cleanup": {
        "enabled": True
    },
    "timeouts": {
        "default": 10,  # seconds
        "long_running": 30  # seconds for performance tests
    },
    "database": {
        "mongodb_uri": "mongodb://localhost:27017",
        "test_db": "test_db",
        "test_collection": "test_collection"
    }
}

class TestReporter:
    def __init__(self):
        self.console = Console()
        self.test_results = []
        self.start_time = None

    def start_test(self, test_name: str):
        self.console.print(f"\n[bold blue]┌── Starting Test: {test_name}")
        self.start_time = time.time()

    def log_step(self, step: str, status: bool, details: str = ""):
        symbol = "✓" if status else "✗"
        color = "green" if status else "red"
        self.console.print(f"[bold blue]│  [/{color}]{symbol}[/] {step}")
        if details:
            self.console.print(f"[bold blue]│     [dim]{details}")

    def end_test(self, test_name: str, success: bool):
        duration = time.time() - self.start_time
        status = "[green]PASSED" if success else "[red]FAILED"
        self.console.print(f"[bold blue]└── {status}[/] {test_name} ({duration:.2f}s)\n")
        self.test_results.append((test_name, success, duration))

    def print_summary(self):
        table = Table(title="Test Summary")
        table.add_column("Test Name", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Duration", justify="right")

        passed = 0
        total = len(self.test_results)

        for name, success, duration in self.test_results:
            status = "[green]PASSED" if success else "[red]FAILED"
            if success:
                passed += 1
            table.add_row(name, status, f"{duration:.2f}s")

        self.console.print("\n")
        self.console.print(Panel(f"[bold]Total Tests: {total} | Passed: {passed} | Failed: {total - passed}"))
        self.console.print(table)

# Create a global reporter instance
reporter = TestReporter()
