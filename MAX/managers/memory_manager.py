import asyncio
import logging
import uuid
from datetime import datetime, UTC
from typing import Dict, Any, Optional, List

from MAX.types.memory_types import DataCategory, DataPriority, MemoryData
from MAX.storage import ChatStorage, ChromaDBChatStorage
from MAX.utils import Logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="memory_system.log",
)
logger = logging.getLogger(__name__)


class MemorySystem:
    """Enhanced Memory System for centralized data management."""

    def __init__(
        self,
        vector_store: ChromaDBChatStorage,
        document_store: ChatStorage,
        logger: Optional[Logger] = None,
        queue_size: int = 1000,
    ):
        """
        Initialize the memory system.

        :param vector_store: A storage system for vector-based data retrieval.
        :param document_store: A storage system for textual/document data.
        :param logger: Optional custom logger instance.
        :param queue_size: Max size for each data category queue.
        """
        self.vector_store = vector_store
        self.document_store = document_store
        self.logger = logger or Logger()
        self.running = True

        self.queues: Dict[DataCategory, asyncio.Queue] = self._init_queues(
            queue_size
        )
        self.processors: List[asyncio.Task] = self._init_processors()

        self.logger.info("Memory system initialized")

    def _init_queues(
        self, queue_size: int
    ) -> Dict[DataCategory, asyncio.Queue]:
        """Initialize a queue for each DataCategory."""
        return {
            category: asyncio.Queue(maxsize=queue_size)
            for category in DataCategory
        }

    def _init_processors(self) -> List[asyncio.Task]:
        """Initialize an asynchronous processor task for each data category."""
        tasks = []
        for category in DataCategory:
            processor = asyncio.create_task(self._process_queue(category))
            tasks.append(processor)
        return tasks

    async def store(
        self,
        data: Dict[str, Any],
        category: DataCategory,
        priority: DataPriority = DataPriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store data with metadata in the system queues."""
        memory_id = str(uuid.uuid4())
        merged_metadata = {
            **(metadata or {}),
            "category": category.value,
            "priority": priority.value,
            "timestamp": datetime.now(UTC).isoformat(),
            "source": (metadata or {}).get("source", "internal"),
        }

        memory_data = MemoryData(
            id=memory_id,
            timestamp=datetime.now(UTC),
            category=category,
            priority=priority,
            content=data,
            metadata=merged_metadata,
        )

        await self.queues[category].put(memory_data)
        self.logger.info(f"Queued {category} data with ID: {memory_id}")
        return memory_id

    async def get_vector_store(self) -> ChromaDBChatStorage:
        """Retrieve the vector store instance."""
        return self.vector_store

    async def get_document_store(self) -> ChatStorage:
        """Retrieve the document store instance."""
        return self.document_store

    async def _process_queue(self, category: DataCategory) -> None:
        """Continuously process items from a category-specific queue."""
        queue = self.queues[category]

        while self.running:
            try:
                data = await queue.get()
                self.logger.debug(
                    f"Processing data ID: {data.id} from {category} queue"
                )

                await self._store_data(data)

                queue.task_done()
                self.logger.info(
                    f"Processed data ID: {data.id} from {category} queue"
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    f"Error processing {category} queue: {str(e)}"
                )
                await asyncio.sleep(1)

    async def _store_data(self, data: MemoryData) -> None:
        """Store a single MemoryData item in both the vector and document stores."""
        try:
            storage_data = {
                "text": str(data.content),
                "metadata": {
                    "id": data.id,
                    "category": data.category,
                    "priority": data.priority,
                    "timestamp": data.timestamp.isoformat(),
                    **data.metadata,
                },
            }

            user_id = data.metadata.get("user_id", "system")
            agent_id = data.metadata.get("agent_id", "system")
            session_id = str(data.category)

            # Store in vector store
            await self.vector_store.save_chat_message(
                user_id=user_id,
                session_id=session_id,
                agent_id=agent_id,
                new_message=storage_data,
            )

            # Store in document store
            await self.document_store.save_chat_message(
                user_id=user_id,
                session_id=session_id,
                agent_id=agent_id,
                new_message=storage_data,
            )

        except Exception as e:
            self.logger.error(f"Error storing data ID: {data.id}: {str(e)}")
            raise

    async def query(
        self,
        query: str,
        category: Optional[DataCategory] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Query stored data using the vector store."""
        try:
            filters = {"category": category.value} if category else None
            results = await self.vector_store.search_similar_messages(
                query=query, metadata_filter=filters, limit=limit
            )
            return results
        except Exception as e:
            self.logger.error(f"Error querying data: {str(e)}")
            raise

    async def shutdown(self) -> None:
        """Shut down the memory system, canceling all processors and clearing queues."""
        self.running = False

        # Cancel processor tasks
        for processor in self.processors:
            processor.cancel()

        # Wait for processors to finish
        await asyncio.gather(*self.processors, return_exceptions=True)

        # Clear all queues
        for queue in self.queues.values():
            while not queue.empty():
                try:
                    queue.get_nowait()
                    queue.task_done()
                except asyncio.QueueEmpty:
                    break

        self.logger.info("Memory system shut down")
