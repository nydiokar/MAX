from opentelemetry import trace
from opentelemetry.trace import Span
from contextlib import contextmanager
from typing import Optional, Dict, Generator


class FetcherTelemetry:
    def __init__(self, service_name: str):
        self.tracer = trace.get_tracer(service_name)

    @contextmanager
    def span(
        self, name: str, attributes: Dict[str, str]
    ) -> Generator[Span, None, None]:
        with self.tracer.start_span(name, attributes=attributes) as span:
            yield span

    def record_exception(self, span, exception: Exception) -> None:
        """Record an exception in the span."""
        if span:
            span.set_status(trace.Status(trace.StatusCode.ERROR))
            span.record_exception(exception)
