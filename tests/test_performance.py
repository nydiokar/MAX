import pytest
import time
from MAX.orchestrator import MultiAgentOrchestrator
from MAX.config.orchestrator_config import OrchestratorConfig
from MAX.agents.default_agent import DefaultAgent

@pytest.mark.performance
class TestPerformance:
    @pytest.fixture
    async def perf_orchestrator(self):
        config = OrchestratorConfig(
            LOG_EXECUTION_TIMES=True,
            MEMORY_ENABLED=True
        )
        orchestrator = MultiAgentOrchestrator(options=config)
        # Add a real DefaultAgent for testing
        agent = DefaultAgent()
        orchestrator.add_agent(agent)
        yield orchestrator
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.slow  # Mark as slow test
    async def test_response_time(self, perf_orchestrator):
        """Test response time for basic requests."""
        start_time = time.time()
        
        response = await perf_orchestrator.route_request(
            user_input="What is 2+2?",  # Simple question for consistent timing
            user_id="perf_test",
            session_id="perf_session"
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Response should be under 30 seconds (more realistic for LLM)
        assert response_time < 30.0
        assert response is not None
        assert response.output is not None

    @pytest.mark.asyncio
    async def test_memory_performance(self, perf_orchestrator):
        """Test memory operations performance."""
        if not perf_orchestrator.memory_manager:
            pytest.skip("Memory manager not enabled")

        # Test storage performance
        messages = [
            f"Test message {i}" for i in range(5)  # Reduced to 5 messages
        ]
        
        storage_times = []
        for msg in messages:
            start_time = time.time()
            await perf_orchestrator.memory_manager.store_message(
                message=msg,
                agent_id="perf_agent",
                session_id="perf_session"
            )
            storage_times.append(time.time() - start_time)
        
        # Average storage operation should be under 2 seconds
        avg_storage_time = sum(storage_times) / len(storage_times)
        assert avg_storage_time < 2.0

        # Test retrieval performance
        start_time = time.time()
        context = await perf_orchestrator.memory_manager.get_relevant_context(
            query="test message",
            agent_id="perf_agent",
            session_id="perf_session"
        )
        retrieval_time = time.time() - start_time
        
        # Retrieval should be under 2 seconds
        assert retrieval_time < 2.0
        assert context is not None 