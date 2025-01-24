from MAX.storage.ChatStorageMongoDB import MongoDBChatStorage
from MAX.storage.TaskStorageMongoDB import MongoDBTaskStorage
from typing import Union
from MAX.storage.utils.types import Task, ExecutionHistoryEntry


class StorageFactory:
    """Factory to manage different storage implementations."""

    def __init__(self, mongo_uri: str, db_name: str):
        self.mongo_uri = mongo_uri
        self.db_name = db_name

    def get_chat_storage(self) -> MongoDBChatStorage:
        """Get the MongoDB implementation for chat storage."""
        return MongoDBChatStorage(
            self.mongo_uri, self.db_name, collection_name="conversations"
        )

    def get_task_storage(self) -> MongoDBTaskStorage:
        """Get the MongoDB implementation for task storage."""
        return MongoDBTaskStorage(self.mongo_uri, self.db_name)

    def get_storage(
        self, storage_type: str
    ) -> Union[MongoDBChatStorage, MongoDBTaskStorage]:
        """Get a specific storage implementation by type."""
        if storage_type == "chat":
            return self.get_chat_storage()
        elif storage_type == "task":
            return self.get_task_storage()
        else:
            raise ValueError(f"Unknown storage type: {storage_type}")
