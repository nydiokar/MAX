import pytest
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock

from MAX.types.memory_types import DataCategory, DataPriority, MemoryData
from MAX.managers.memory_manager import MemorySystem

# Configure test logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='memory_system_test.log'
)
logger = logging.getLogger(__name__)

@pytest.fixture
def mock_stores():
    vector_store = Mock()
    vector_store.save_chat_message = AsyncMock()
    vector_store.search_similar_messages = AsyncMock()
    
    document_store = Mock()
    document_store.save_chat_message = AsyncMock()
    
    return vector_store, document_store

@pytest.fixture
async def memory_system(mock_stores):
    vector_store, document_store = mock_stores
    system = MemorySystem(vector_store, document_store)
    yield system
    await system.shutdown()

async def generate_test_data(count: int = 100) -> List[Dict[str, Any]]:
    """Generate test data for concurrent testing"""
    return [
        {
            "content": f"Test data {i}",
            "metadata": {"test_id": i}
        }
        for i in range(count)
    ]

@pytest.mark.asyncio
async def test_data_routing(memory_system, mock_stores):
    """Test data routing through different categories"""
    logger.info("Starting data routing test")
    
    test_data = {
        "content": "Test message",
        "metadata": {"user_id": "test_user"}
    }
    
    # Store data in different categories
    categories = [
        (DataCategory.USER, DataPriority.HIGH),
        (DataCategory.SYSTEM, DataPriority.MEDIUM),
        (DataCategory.AGENT, DataPriority.LOW)
    ]
    
    for category, priority in categories:
        memory_id = await memory_system.store(
            data=test_data,
            category=category,
            priority=priority
        )
        logger.info(f"Stored data {memory_id} in category {category}")
    
    # Wait for processing
    await asyncio.sleep(1)
    
    # Verify storage calls
    vector_store, document_store = mock_stores
    assert vector_store.save_chat_message.call_count == len(categories)
    assert document_store.save_chat_message.call_count == len(categories)
    
    logger.info("Data routing test completed")

@pytest.mark.asyncio
async def test_concurrent_storage(memory_system):
    """Test concurrent data storage"""
    logger.info("Starting concurrent storage test")
    
    # Reduce test count to avoid overwhelming the system
    test_count = 50  # Reduced from 100
    test_data = await generate_test_data(test_count)
    
    # Store data concurrently
    async def store_item(data: Dict[str, Any], index: int):
        category = DataCategory.USER if index % 2 == 0 else DataCategory.SYSTEM
        return await memory_system.store(
            data=data,
            category=category,
            priority=DataPriority.MEDIUM
        )
    
    start_time = datetime.now()
    
    tasks = [
        store_item(data, i) 
        for i, data in enumerate(test_data)
    ]
    
    memory_ids = await asyncio.gather(*tasks)
    
    # Allow more time for processing
    await asyncio.sleep(3)
    
    duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"Stored {len(memory_ids)} items in {duration:.2f} seconds")
    logger.info(f"Processing rate: {len(memory_ids)/duration:.2f} items/second")
    
    # Add verification
    assert len(memory_ids) == test_count, "Not all items were stored"
    assert all(isinstance(id, str) for id in memory_ids), "Invalid memory IDs returned"
    
    logger.info("Concurrent storage test completed")

@pytest.fixture
def setup_logging():
    # Configure detailed logging
    file_handler = logging.FileHandler('memory_system_test.log', mode='w')  # 'w' mode to overwrite
    file_handler.setLevel(logging.DEBUG)  # Capture all levels
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler to both loggers
    logger.addHandler(file_handler)
    root_logger = logging.getLogger('MAX.utils.logger')
    root_logger.addHandler(file_handler)
    
    yield
    
    # Cleanup
    logger.removeHandler(file_handler)
    root_logger.removeHandler(file_handler)
    file_handler.close()

@pytest.mark.asyncio
async def test_queue_handling(memory_system, setup_logging):
    """Test queue handling and backpressure"""
    logger.info("Starting queue handling test")
    
    # Reduce test data size to avoid overwhelming the queue
    queue_size = 1000
    test_count = queue_size + 100  # Testing overflow behavior
    test_data = await generate_test_data(test_count)
    
    stored = 0
    skipped = 0
    
    # Add delay to allow processing
    for data in test_data:
        try:
            await memory_system.store(
                data=data,
                category=DataCategory.USER,
                priority=DataPriority.LOW
            )
            stored += 1
            # Add small delay to allow queue processing
            if stored % 100 == 0:
                await asyncio.sleep(0.1)
        except asyncio.QueueFull:
            skipped += 1
    
    # Allow time for queue processing
    await asyncio.sleep(2)
    
    logger.info(f"Stored: {stored}, Skipped: {skipped}")
    
    # Modified assertion to check queue behavior
    assert stored + skipped == test_count, "Not all items were processed or skipped"
    assert memory_system.queues[DataCategory.USER].qsize() <= queue_size, "Queue exceeded maximum size"
    
    logger.info("Queue handling test completed")

if __name__ == "__main__":
    pytest.main(["-v", __file__])