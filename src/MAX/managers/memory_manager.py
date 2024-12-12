import asyncio
import logging
from datetime import datetime, UTC
from typing import Dict, Any, Optional, List
import uuid

from MAX.types.memory_types import DataCategory, DataPriority, MemoryData
from MAX.storage import ChatStorage, ChromaDBChatStorage
from MAX.utils import Logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='memory_system.log'
)
logger = logging.getLogger(__name__)

class MemorySystem:
    """Enhanced Memory System for centralized data management"""
    
    def __init__(
        self,
        vector_store: ChromaDBChatStorage,
        document_store: ChatStorage,
        logger: Optional[Logger] = None,
        queue_size: int = 1000
    ):
        self.vector_store = vector_store
        self.document_store = document_store
        self.logger = logger or Logger()
        
        # Initialize queues for different categories
        self.queues: Dict[DataCategory, asyncio.Queue] = {
            category: asyncio.Queue(maxsize=queue_size)
            for category in DataCategory
        }
        
        # Processing tasks
        self.processors = []
        self._running = True
        
        # Start processors
        for category in DataCategory:
            processor = asyncio.create_task(self._process_queue(category))
            self.processors.append(processor)
            
        self.logger.info("Memory system initialized")

    async def store(
        self,
        data: Dict[str, Any],
        category: DataCategory,
        priority: DataPriority = DataPriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store data in memory system"""
        memory_id = str(uuid.uuid4())
        
        memory_data = MemoryData(
            id=memory_id,
            timestamp=datetime.now(UTC),
            category=category,
            priority=priority,
            content=data,
            metadata=metadata or {}
        )
        
        try:
            await self.queues[category].put(memory_data)
            self.logger.info(f"Queued data {memory_id} for {category}")
            return memory_id
        except Exception as e:
            self.logger.error(f"Error storing data: {str(e)}")
            raise

    async def _process_queue(self, category: DataCategory) -> None:
        """Process data from category queue"""
        queue = self.queues[category]
        
        while self._running:
            try:
                # Get data from queue
                data = await queue.get()
                self.logger.debug(f"Processing {data.id} from {category} queue")
                
                # Store in both storages
                await self._store_data(data)
                
                queue.task_done()
                self.logger.info(f"Processed {data.id} from {category} queue")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error processing {category} queue: {str(e)}")
                await asyncio.sleep(1)

    async def _store_data(self, data: MemoryData) -> None:
        """Store data in storage systems"""
        try:
            # Prepare storage format
            storage_data = {
                "text": str(data.content),
                "metadata": {
                    "id": data.id,
                    "category": data.category,
                    "priority": data.priority,
                    "timestamp": data.timestamp.isoformat(),
                    **data.metadata
                }
            }
            
            # Store in vector store
            await self.vector_store.save_chat_message(
                user_id=data.metadata.get('user_id', 'system'),
                session_id=str(data.category),
                agent_id=data.metadata.get('agent_id', 'system'),
                new_message=storage_data
            )
            
            # Store in document store
            await self.document_store.save_chat_message(
                user_id=data.metadata.get('user_id', 'system'),
                session_id=str(data.category),
                agent_id=data.metadata.get('agent_id', 'system'),
                new_message=storage_data
            )
            
        except Exception as e:
            self.logger.error(f"Error storing data {data.id}: {str(e)}")
            raise

    async def query(
        self,
        query: str,
        category: Optional[DataCategory] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Query stored data"""
        try:
            filters = {"category": category.value} if category else None
            
            results = await self.vector_store.search_similar_messages(
                query=query,
                metadata_filter=filters,
                limit=limit
            )
            
            return results
        except Exception as e:
            self.logger.error(f"Error querying data: {str(e)}")
            raise

    async def shutdown(self) -> None:
        """Shutdown memory system"""
        self._running = False
        
        # Cancel processors
        for processor in self.processors:
            processor.cancel()
            
        # Wait for all processors
        await asyncio.gather(*self.processors, return_exceptions=True)
        
        # Clear queues
        for queue in self.queues.values():
            while not queue.empty():
                try:
                    queue.get_nowait()
                    queue.task_done()
                except asyncio.QueueEmpty:
                    break
        
        self.logger.info("Memory system shut down")