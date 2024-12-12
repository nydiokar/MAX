import pytest
import asyncio
import logging
import time
from datetime import datetime
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock

from MAX.types import ConversationMessage, ParticipantRole
from MAX.config.database_config import DatabaseConfig
from MAX.storage.data_hub_cache import DataHubCache, DataHubCacheConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='cache_performance_test.log'
)
logger = logging.getLogger(__name__)

class PerformanceMetrics:
    def __init__(self):
        self.request_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0
        self.start_time = None
        self.end_time = None

    def log_request(self, duration: float, is_hit: bool):
        self.request_times.append(duration)
        self.total_requests += 1
        if is_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def get_summary(self) -> Dict[str, Any]:
        if not self.request_times:
            return {}
            
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_ratio": self.cache_hits / self.total_requests if self.total_requests > 0 else 0,
            "avg_response_time": sum(self.request_times) / len(self.request_times),
            "max_response_time": max(self.request_times),
            "min_response_time": min(self.request_times),
            "total_duration": self.end_time - self.start_time if self.end_time else 0
        }

@pytest.fixture
async def performance_test_setup():
    """Setup test environment with cache and metrics"""
    config = DataHubCacheConfig(
        message_cache_ttl=30,
        state_cache_ttl=10,
        max_cached_sessions=1000,
        max_messages_per_session=100
    )
    db_config = DatabaseConfig()
    cache = DataHubCache(config, db_config)
    metrics = PerformanceMetrics()
    return cache, metrics

async def simulate_user_session(
    cache: DataHubCache,
    user_id: str,
    session_id: str,
    message_count: int,
    mock_storage: Mock,
    metrics: PerformanceMetrics
) -> None:
    """Simulate a user session with multiple message requests"""
    for i in range(message_count):
        start_time = time.time()
        
        try:
            # Simulate some processing delay
            await asyncio.sleep(0.01)
            
            result = await cache.get_chat_messages(
                user_id=user_id,
                session_id=session_id,
                agent_id="test_agent",
                storage=mock_storage
            )
            
            duration = time.time() - start_time
            is_hit = result is not None
            metrics.log_request(duration, is_hit)
            
            logger.debug(f"Request {i} for user {user_id}: {'Hit' if is_hit else 'Miss'} in {duration:.4f}s")
            
        except Exception as e:
            logger.error(f"Error in user session {user_id}: {str(e)}")

@pytest.mark.asyncio
async def test_concurrent_load(performance_test_setup):
    """Test cache performance under concurrent load"""
    cache, metrics = performance_test_setup
    
    # Setup mock storage
    mock_storage = Mock()
    mock_storage.fetch_chat = AsyncMock(return_value=[
        ConversationMessage(
            role=ParticipantRole.USER.value,
            content=[{"text": f"Test message"}]
        )
    ])
    
    # Test parameters
    num_users = 50
    messages_per_user = 20
    
    logger.info(f"Starting concurrent load test with {num_users} users, {messages_per_user} messages each")
    
    # Create user sessions
    metrics.start_time = time.time()
    
    tasks = [
        simulate_user_session(
            cache=cache,
            user_id=f"user_{i}",
            session_id=f"session_{i}",
            message_count=messages_per_user,
            mock_storage=mock_storage,
            metrics=metrics
        )
        for i in range(num_users)
    ]
    
    # Run concurrent sessions
    await asyncio.gather(*tasks)
    
    metrics.end_time = time.time()
    
    # Log results
    summary = metrics.get_summary()
    logger.info("Performance Test Results:")
    logger.info(f"Total Requests: {summary['total_requests']}")
    logger.info(f"Cache Hit Ratio: {summary['hit_ratio']:.2%}")
    logger.info(f"Average Response Time: {summary['avg_response_time']*1000:.2f}ms")
    logger.info(f"Max Response Time: {summary['max_response_time']*1000:.2f}ms")
    logger.info(f"Total Test Duration: {summary['total_duration']:.2f}s")
    
    # Verify performance metrics
    assert summary['hit_ratio'] > 0.7, "Cache hit ratio below target"
    assert summary['avg_response_time'] < 0.1, "Average response time too high"
    
    # Get cache statistics
    cache_stats = cache.get_cache_stats()
    logger.info("Cache Statistics:")
    logger.info(f"Message Cache Size: {cache_stats['message_cache_size']}")
    logger.info(f"Message Hits: {cache_stats['message_hits']}")
    logger.info(f"Message Misses: {cache_stats['message_misses']}")

if __name__ == "__main__":
    pytest.main(["-v", __file__])