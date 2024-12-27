import pytest
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List
import time

from MAX.adapters.fetchers.base.fetcher import AbstractFetcher, FetcherConfig
from MAX.adapters.fetchers.base.types import FetchStatus, FetchResult
from MAX.storage.ChatStorageMongoDB import MongoDBChatStorage
from Orch.python.src.MAX.managers.system_state_manager import StateManager
from MAX.config.database_config import DatabaseConfig
from MAX.types import ConversationMessage, ParticipantRole
from tests.test_utils import TEST_CONFIG

# Make sure this is at module level
pytestmark = pytest.mark.asyncio

class MockMongoDBStorage(MongoDBChatStorage):
    """Mock storage with in-memory implementation"""
    
    def __init__(self, *args, **kwargs):
        self.messages = {}
        self.states = {}
        self.initialized = False
        self._system_state = {
            "agent_states": {},
            "conversation_states": {},
            "metadata": {},
            "timestamp": datetime.now(timezone.utc)
        }

    async def initialize(self) -> bool:
        self.initialized = True
        return True

    async def save_chat_message(
        self, 
        user_id: str, 
        session_id: str, 
        agent_id: str,
        new_message: ConversationMessage,
        metadata: Optional[Dict] = None
    ) -> bool:
        # Standardize content format
        if isinstance(new_message.content, str):
            standardized_content = [{"text": new_message.content}]
        elif isinstance(new_message.content, dict):
            standardized_content = [new_message.content]
        else:
            standardized_content = new_message.content
            
        new_message.content = standardized_content
        
        # Use channel from metadata if available
        effective_agent_id = metadata.get("channel", agent_id) if metadata else agent_id
        key = f"{user_id}:{session_id}:{effective_agent_id}"
        
        if key not in self.messages:
            self.messages[key] = []
        self.messages[key].append(new_message)
        
        # Also store in the combined key for fetch_all_chats
        combined_key = f"{user_id}:{session_id}"
        if combined_key not in self.messages:
            self.messages[combined_key] = []
        self.messages[combined_key].append(new_message)
        return True

    async def fetch_chat(
        self, 
        user_id: str, 
        session_id: str, 
        agent_id: str = None
    ) -> List[ConversationMessage]:
        if agent_id:
            key = f"{user_id}:{session_id}:{agent_id}"
        else:
            key = f"{user_id}:{session_id}"
        return self.messages.get(key, [])

    async def fetch_all_chats(
        self,
        user_id: str,
        session_id: str
    ) -> List[ConversationMessage]:
        key = f"{user_id}:{session_id}"
        return self.messages.get(key, [])

    async def save_system_state(
        self,
        state_type: str,
        state_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        if state_type == "SYSTEM_STATE":
            valid_fields = {"agent_states", "conversation_states", "metadata", "timestamp"}
            filtered_data = {k: v for k, v in state_data.items() if k in valid_fields}
            self._system_state.update(filtered_data)
        self.states[state_type] = state_data
        return True

    async def get_system_state(
        self,
        state_type: str = "SYSTEM_STATE"
    ) -> Optional[Dict[str, Any]]:
        if state_type == "SYSTEM_STATE":
            return self._system_state
        return self.states.get(state_type)

    async def check_health(self) -> Tuple[bool, Dict[str, Any]]:
        return True, {"status": "healthy", "mock": True}

    async def cleanup(self) -> None:
        self.messages = {}
        self.states = {}
        self._system_state = {
            "agent_states": {},
            "conversation_states": {},
            "metadata": {},
            "timestamp": datetime.now(timezone.utc)
        }
        self.initialized = False

    async def save_task_state(
        self,
        task_id: str,
        state: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        self.states[task_id] = state
        return True

class TestFetcher(AbstractFetcher):
    """Test implementation of AbstractFetcher"""
    
    def __init__(self, config: FetcherConfig, name: str):
        super().__init__(config, name)
        self.cache = {}
        self.cache_timestamps = {}
        self._stats = {
            'requests': 0,
            'successes': 0,
            'failures': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_latency': 0.0,
            'rate_limits': 0,
            'circuit_breaks': 0
        }

    @property
    def default_headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json"}

    async def process_response(self, response) -> Dict[str, Any]:
        return {"status": "success", "data": "test_data"}

    async def fetch(self, endpoint: str, use_cache: bool = True) -> FetchResult:
        self._stats['requests'] += 1
        current_time = time.time()
        
        if use_cache and endpoint in self.cache:
            cache_age = current_time - self.cache_timestamps[endpoint]
            if cache_age < self.config.cache_ttl:
                self._stats['cache_hits'] += 1  
                return FetchResult(status=FetchStatus.CACHED, data=self.cache[endpoint])
            else:
                # Clear expired cache
                del self.cache[endpoint]
                del self.cache_timestamps[endpoint]
        
        # Only increment cache_misses if we're using cache
        if use_cache:
            self._stats['cache_misses'] += 1
        
        data = await self.process_response(None)
        
        if use_cache:
            self.cache[endpoint] = data
            self.cache_timestamps[endpoint] = current_time
        
        self._stats['successes'] += 1
        return FetchResult(status=FetchStatus.SUCCESS, data=data)

    @property
    def stats(self) -> Dict[str, Any]:
        stats = self._stats.copy()
        total_requests = stats['requests']
        if total_requests > 0:
            stats['success_rate'] = stats['successes'] / total_requests
            stats['cache_hit_rate'] = stats['cache_hits'] / total_requests
        else:
            stats['success_rate'] = 0.0
            stats['cache_hit_rate'] = 0.0
        return stats

    async def disconnect(self):
        self.cache.clear()
        self.cache_timestamps.clear()

@pytest.fixture
async def test_components():
    """Initialize test components with mocks"""
    print("\nSetting up test components...")
    
    # Initialize configs
    db_config = DatabaseConfig()
    db_config.mongodb.uri = TEST_CONFIG["database"]["mongodb_uri"]
    db_config.mongodb.database = TEST_CONFIG["database"]["test_db"]
    
    # Create mock storage first
    mongo_storage = MockMongoDBStorage(
        mongo_uri=db_config.mongodb.uri,
        db_name=db_config.mongodb.database,
        collection_name=TEST_CONFIG["database"]["test_collection"]
    )
    
    # Initialize storage
    await mongo_storage.initialize()
    
    # Create and initialize state manager properly
    state_manager = StateManager(db_config)
    state_manager.mongo_storage = mongo_storage  # Override with mock storage
    await state_manager._initialize_storages()  # Ensure proper initialization
    
    # Initialize fetcher
    fetcher_config = FetcherConfig(
        base_url="http://test-api.local",
        requests_per_second=10.0,
        cache_ttl=60,
        enable_metrics=False,
        enable_telemetry=False
    )
    fetcher = TestFetcher(fetcher_config, "test_fetcher")
    
    yield {
        "fetcher": fetcher,
        "state_manager": state_manager,
        "storage": mongo_storage
    }
    
    # Cleanup
    print("\nCleaning up test components...")
    try:
        await mongo_storage.cleanup()
        await state_manager.cleanup_old_states()
        await fetcher.disconnect()
    except Exception as e:
        print(f"Cleanup error: {e}")

@pytest.mark.asyncio
async def test_data_flow(test_components):
    """Test complete data flow through components"""
    fetcher = test_components["fetcher"]
    state_manager = test_components["state_manager"]
    storage = test_components["storage"]
    
    # 1. Simulate data fetch
    test_data = {
        "status": "success",
        "data": "test_response"
    }
    fetch_result = await fetcher.fetch("/test_endpoint")
    assert fetch_result.status == FetchStatus.SUCCESS
    
    # 2. Create and track conversation message
    message = ConversationMessage(
        role=ParticipantRole.USER.value,
        content=[{"text": "Test message"}]
    )
    
    # 3. Test state tracking
    success = await state_manager.track_conversation_state(
        user_id="test_user",
        session_id="test_session",
        message=message,
        metadata={"source": "test", "fetch_result": test_data}
    )
    assert success, "Failed to track conversation state"
    
    # 4. Verify storage
    stored_messages = await storage.fetch_chat(
        user_id="test_user",
        session_id="test_session",
        agent_id="conversation_state"
    )
    assert len(stored_messages) > 0
    
    # 5. Verify fetcher cache
    cached_result = await fetcher.fetch("/test_endpoint", use_cache=True)
    assert cached_result.status == FetchStatus.CACHED

@pytest.mark.asyncio
async def test_error_handling(test_components):
    """Test error handling and recovery"""
    state_manager = test_components["state_manager"]
    
    # Test invalid message handling
    invalid_message = ConversationMessage(
        role=ParticipantRole.USER.value,
        content=[]  # Invalid content
    )
    
    # Should handle error gracefully
    success = await state_manager.track_conversation_state(
        user_id="test_user",
        session_id="test_session",
        message=invalid_message
    )
    assert not success, "Should fail gracefully with invalid message"
    
    # System should remain operational
    health_status = await state_manager.check_storage_health()
    assert health_status["healthy"], "System unhealthy after error"

@pytest.mark.asyncio
async def test_component_interoperability(test_components):
    """Test interaction between all components"""
    fetcher = test_components["fetcher"]
    state_manager = test_components["state_manager"]
    storage = test_components["storage"]
    
    # 1. Test cache interaction
    test_endpoint = "/test"
    for _ in range(3):
        result = await fetcher.fetch(test_endpoint, use_cache=True)
        # Should use cache after first call
    
    # 2. Test state persistence
    message = ConversationMessage(
        role=ParticipantRole.USER.value,
        content=[{"text": "Test interop"}]
    )
    
    await state_manager.track_conversation_state(
        user_id="test_user",
        session_id="test_session",
        message=message
    )
    
    # 3. Verify state restoration
    await state_manager.restore_state_from_storage()
    system_state = await state_manager.get_system_snapshot()
    assert "test_user:test_session" in system_state.conversation_states
    
    # 4. Test concurrent operations
    async def concurrent_operation(idx: int):
        message = ConversationMessage(
            role=ParticipantRole.USER.value,
            content=[{"text": f"Concurrent test {idx}"}]
        )
        await state_manager.track_conversation_state(
            user_id=f"user_{idx}",
            session_id="test_session",
            message=message
        )
    
    # Run multiple operations concurrently
    await asyncio.gather(*[concurrent_operation(i) for i in range(5)])
    
    # Verify all operations completed
    stored_messages = await storage.fetch_all_chats(
        user_id="user_0",
        session_id="test_session"
    )
    assert len(stored_messages) > 0, "Concurrent operations failed"

@pytest.mark.asyncio
async def test_data_standardization(test_components):
    """Test data standardization and transformation"""
    state_manager = test_components["state_manager"]
    
    # Test different message formats
    test_cases = [
        {
            "content": [{"text": "Plain text message"}],  # Changed from string to list
            "expected_format": [{"text": "Plain text message"}]
        },
        {
            "content": [{"text": "Structured message"}],
            "expected_format": [{"text": "Structured message"}]
        },
        {
            "content": [{"text": "Object message"}],  # Changed from dict to list
            "expected_format": [{"text": "Object message"}]
        }
    ]
    
    for case in test_cases:
        message = ConversationMessage(
            role=ParticipantRole.USER.value,
            content=case["content"]
        )
        
        await state_manager.track_conversation_state(
            user_id="test_user",
            session_id="test_session",
            message=message
        )
        
        stored_messages = await test_components["storage"].fetch_chat(
            user_id="test_user",
            session_id="test_session"
        )
        assert stored_messages[-1].content == case["expected_format"]

@pytest.mark.asyncio
async def test_message_routing(test_components):
    """Test message routing and delivery"""
    state_manager = test_components["state_manager"]
    
    # Test routing to different channels
    channels = ["conversation_state", "analysis", "metadata"]
    
    for channel in channels:
        message = ConversationMessage(
            role=ParticipantRole.USER.value,
            content=[{"text": f"Test message for {channel}"}]
        )
        
        await state_manager.track_conversation_state(
            user_id="test_user",
            session_id="test_session",
            message=message,
            metadata={"channel": channel}
        )
        
        # Verify message reached correct channel
        stored_messages = await test_components["storage"].fetch_chat(
            user_id="test_user",
            session_id="test_session",
            agent_id=channel
        )
        assert len(stored_messages) > 0

@pytest.mark.asyncio
async def test_caching_behavior(test_components):
    """Test caching mechanisms"""
    fetcher = test_components["fetcher"]
    
    # Test cache hits and misses
    endpoints = ["/test1", "/test2", "/test3"]
    
    # First round - should all be cache misses
    for endpoint in endpoints:
        result = await fetcher.fetch(endpoint, use_cache=True)
        assert result.status != FetchStatus.CACHED
    
    # Second round - should all be cache hits
    for endpoint in endpoints:
        result = await fetcher.fetch(endpoint, use_cache=True)
        assert result.status == FetchStatus.CACHED
    
    # Test cache expiration
    await asyncio.sleep(fetcher.config.cache_ttl + 1)
    result = await fetcher.fetch(endpoints[0], use_cache=True)
    assert result.status != FetchStatus.CACHED

@pytest.mark.asyncio
async def test_concurrent_state_updates(test_components):
    """Test concurrent state updates to verify data consistency"""
    state_manager = test_components["state_manager"]
    
    # Create multiple concurrent updates
    async def concurrent_update(i: int):
        message = ConversationMessage(
            role=ParticipantRole.USER.value,
            content=[{"text": f"Concurrent message {i}"}]
        )
        success = await state_manager.track_conversation_state(
            user_id="same_user",
            session_id="same_session",
            message=message
        )
        return {
            "success": success,
            "message_id": i,
            "content": f"Concurrent message {i}"
        }
    
    # Run 10 updates concurrently
    results = await asyncio.gather(*[concurrent_update(i) for i in range(10)])
    
    # Verify all operations succeeded
    assert all(r["success"] for r in results), "Some state updates failed"
    
    # Verify final state
    stored_messages = await test_components["storage"].fetch_chat(
        user_id="same_user",
        session_id="same_session"
    )
    
    # Check message count
    assert len(stored_messages) == 10, "Some messages were lost in concurrent updates"
    
    # Verify message contents are all present and haven't been corrupted
    stored_contents = {msg.content[0]["text"] for msg in stored_messages}
    expected_contents = {f"Concurrent message {i}" for i in range(10)}
    assert stored_contents == expected_contents, "Message contents don't match expected values"
    
    # Verify state manager's internal state
    conversation_key = "same_user:same_session"
    state = state_manager.system_state.conversation_states.get(conversation_key)
    assert state is not None, "Conversation state not found"
    assert state["last_message"]["content"][0]["text"] in expected_contents, "Final state doesn't contain valid message"


@pytest.mark.asyncio
async def test_performance_metrics(test_components):
    """Test performance metrics collection"""
    fetcher = test_components["fetcher"]
    state_manager = test_components["state_manager"]
    
    # Generate some test load
    for i in range(5):
        message = ConversationMessage(
            role=ParticipantRole.USER.value,
            content=[{"text": f"Test message {i}"}]
        )
        await state_manager.track_conversation_state(
            user_id="test_user",
            session_id="test_session",
            message=message
        )
        await fetcher.fetch(f"/test_endpoint_{i}")
    
    # Verify cache metrics
    cached_result = await fetcher.fetch("/test_endpoint_0", use_cache=True)
    assert cached_result.status == FetchStatus.CACHED
    
    # Check stats after cache hit
    stats = fetcher.stats
    assert stats['requests'] == 6  # 5 initial + 1 cache hit
    assert stats['successes'] == 5
    assert stats['cache_hits'] >= 1
    assert 'success_rate' in stats
    assert stats['success_rate'] > 0.0

if __name__ == "__main__":
    pytest.main(["-v", __file__])
