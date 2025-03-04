import pytest
import aiohttp
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, AsyncGenerator

from MAX.agents import RecursiveThinkerAgent, RecursiveThinkerOptions
from MAX.types import ConversationMessage, ParticipantRole, TimestampedMessage
from MAX.storage import MongoDBChatStorage
from MAX.storage.utils.storage_factory import StorageFactory
from MAX.storage.data_hub_cache import DataHubCache, DataHubCacheConfig
from MAX.retrievers.kb_retriever import KnowledgeBasesRetriever, KnowledgeBasesRetrieverOptions
from MAX.config.database_config import DatabaseConfig

async def is_ollama_available() -> bool:
    """Check if Ollama server is running and has models loaded"""
    try:
        timeout = aiohttp.ClientTimeout(total=5)  # Set overall timeout to 5 seconds
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Check server is up
            async with session.get("http://localhost:11434/api/tags") as resp:
                if resp.status != 200:
                    print("\nOllama server not responding")
                    return False
                tags = await resp.json()
                print(f"\nAvailable models: {tags}")
                
            # Check model can generate with timeout
            test_payload = {
                "model": "llama3.1:8b-instruct-q8_0",
                "prompt": "Say hi",
                "stream": False
            }
            async with session.post("http://localhost:11434/api/generate", json=test_payload) as resp:
                if resp.status != 200:
                    print("\nOllama cannot generate responses")
                    return False
                await resp.json()  # Actually wait for response
                print("\nOllama generation test successful")
                return True
                
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        print(f"\nError checking Ollama: {str(e)}")
        return False

@pytest.fixture
async def storage() -> AsyncGenerator[MongoDBChatStorage, None]:
    """Create MongoDB storage instance for testing"""
    storage_factory = StorageFactory(
        mongo_uri="mongodb://localhost:27017",
        db_name="max_agents_test"  # Use a separate test database
    )
    storage = storage_factory.get_chat_storage()
    await storage.initialize()  # Properly initialize the storage
    yield storage
    await storage.cleanup()  # Cleanup after tests

@pytest.fixture
async def cache(storage: MongoDBChatStorage) -> AsyncGenerator[DataHubCache, None]:
    """Create DataHubCache instance for testing"""
    config = DataHubCacheConfig(
        message_cache_ttl=300,  # 5 minutes
        state_cache_ttl=60,     # 1 minute
        max_cached_sessions=1000,
        max_messages_per_session=100,
        enable_vector_cache=True
    )
    db_config = DatabaseConfig()
    cache = DataHubCache(config, db_config)
    yield cache

@pytest.fixture
async def retriever(storage: MongoDBChatStorage) -> AsyncGenerator[KnowledgeBasesRetriever, None]:
    """Create KnowledgeBasesRetriever instance for testing"""
    options = KnowledgeBasesRetrieverOptions(
        storage_client=storage,
        collection_name="test_collection",
        max_results=5,
        similarity_threshold=0.7
    )
    retriever = KnowledgeBasesRetriever(options)
    yield retriever

@pytest.fixture
def agent_options(storage: MongoDBChatStorage, cache: DataHubCache, retriever: KnowledgeBasesRetriever) -> RecursiveThinkerOptions:
    return RecursiveThinkerOptions(
        name="test_agent",
        description="Test agent",
        model_type="llama3.1:8b-instruct-q8_0",  # Use model we know exists
        storage=storage,
        streaming=False,
        memory_system=cache,  # Use DataHubCache for memory
        retriever=retriever   # Use KnowledgeBasesRetriever for context retrieval
    )

@pytest.fixture
async def agent(agent_options: RecursiveThinkerOptions) -> AsyncGenerator[RecursiveThinkerAgent, None]:
    agent = RecursiveThinkerAgent(agent_options)
    await agent.initialize()
    yield agent
    # Don't cleanup storage so we can verify the data
    # if agent.options.storage:
    #     await agent.options.storage.cleanup()

@pytest.mark.asyncio
async def test_basic_chat(agent: RecursiveThinkerAgent) -> None:
    """Test that agent can understand and respond to messages"""
    print("\nTesting basic chat functionality...")
    
    try:
        # Test greeting and factual question
        response = await agent.process_request(
            input_text="Hi! Can you tell me what's 2+2?",
            user_id="test_user",
            session_id="test_session",
            chat_history=[]
        )
        
        print(f"\nAgent response: {response.content[0]['text']}\n")
        
        # First check if we got an error response
        if response.metadata.get('error'):
            pytest.fail(f"Agent returned error: {response.content[0]['text']}")
            
        assert isinstance(response, ConversationMessage)
        assert response.role == ParticipantRole.ASSISTANT
        assert "4" in response.content[0]["text"]
        
    except Exception as e:
        pytest.fail(f"Test failed with exception: {str(e)}")

@pytest.mark.asyncio
async def test_memory(agent: RecursiveThinkerAgent, cache: DataHubCache) -> None:
    """Test memory system functionality"""
    # Test storage functionality without LLM
    test_messages: List[ConversationMessage] = []
    for i in range(3):
        msg = ConversationMessage(
            role=ParticipantRole.USER,
            content=[{"text": f"Test message {i}"}],
            message_type="text",
            metadata={
                "timestamp": datetime.now().timestamp(),
                "user_id": "test_user",
                "session_id": "test_session"
            }
        )
        # Use cache to store message
        await cache.save_chat_message(
            message=msg,
            user_id="test_user",
            session_id="test_session",
            agent_id=agent.id,
            storage=agent.options.storage
        )
        test_messages.append(msg)
    
    # Verify storage retrieval through cache
    stored_messages = await cache.get_chat_messages(
        user_id="test_user",
        session_id="test_session",
        agent_id=agent.id,
        storage=agent.options.storage
    )
    assert len(stored_messages) >= 3
    
    # Verify message structure
    for msg in stored_messages:
        assert isinstance(msg, ConversationMessage)
        assert "timestamp" in msg.metadata

@pytest.mark.asyncio
async def test_memory_persistence(agent: RecursiveThinkerAgent, cache: DataHubCache) -> None:
    """Test that memory persists across agent instances"""
    print("\nTesting memory persistence...")
    
    # Store a message with first agent instance
    response1 = await agent.process_request(
        input_text="Remember this: The code is 1234",
        user_id="test_user",
        session_id="persistence_test",
        chat_history=[]
    )
    
    # Verify the response
    assert isinstance(response1, ConversationMessage)
    assert response1.role == ParticipantRole.ASSISTANT
    assert "1234" in response1.content[0]["text"]
    
    # Debug: Print storage info
    print(f"\nStorage database: {agent.options.db_name}")
    print(f"Storage collection: {agent.options.collection_name}")
    
    # Verify message was stored in cache
    stored_messages = await cache.get_chat_messages(
        user_id="test_user",
        session_id="persistence_test",
        agent_id=agent.id,
        storage=agent.options.storage
    )
    
    # Debug: Print stored messages
    print(f"\nStored messages in cache: {len(stored_messages)}")
    for msg in stored_messages:
        print(f"- {msg.role}: {msg.content[0]['text']}")
    
    # Verify storage contents
    assert len(stored_messages) >= 2  # Should have user message and response
    assert any("1234" in msg.content[0]["text"] for msg in stored_messages), "Original message not found in storage"

if __name__ == "__main__":
    pytest.main([__file__])
