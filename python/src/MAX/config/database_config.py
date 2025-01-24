from typing import Optional
from dataclasses import dataclass


@dataclass
class MongoDBConfig:
    uri: str = "mongodb://localhost:27017"
    database: str = "max_plus"
    state_collection: str = "system_state"
    chat_collection: str = "chat_history"
    ttl_hours: Optional[int] = 24


@dataclass
class ChromaDBConfig:
    persist_directory: str = "./chroma_db"
    collection_name: str = "semantic_store"


@dataclass
class StateManagerConfig:
    enable_vector_storage: bool = True
    state_cleanup_interval_hours: int = 24
    max_state_age_hours: int = 72
    max_conversation_history: int = 100


class DatabaseConfig:
    def __init__(self):
        self.mongodb = MongoDBConfig()
        self.chromadb = ChromaDBConfig()
        self.state_manager = StateManagerConfig()
