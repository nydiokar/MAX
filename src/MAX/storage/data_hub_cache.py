from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import asyncio
from cachetools import TTLCache, LRUCache
from dataclasses import dataclass
import json
from MAX.types import ConversationMessage, TimestampedMessage
from MAX.utils import Logger
from MAX.storage import ChatStorage, ChromaDBChatStorage
from MAX.config.database_config import DatabaseConfig
from Orch.python.src.MAX.storage import ChromaDB

# INCLUDE THE NEW STORAGE METHOD!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


@dataclass
class DataHubCacheConfig:
    """Configuration for Data Hub cache system"""
    message_cache_ttl: int = 300  # 5 minutes for chat messages
    state_cache_ttl: int = 60     # 1 minute for state data
    max_cached_sessions: int = 1000
    max_messages_per_session: int = 100
    enable_vector_cache: bool = True
    sync_interval: int = 30

class DataHubCache:
    """Centralized cache manager for MAX+ Data Hub"""
    
    def __init__(self, config: DataHubCacheConfig, db_config: DatabaseConfig):
        self.config = config
        self.db_config = db_config
        self.logger = Logger()
        
        # Initialize cache stores
        self.message_cache = TTLCache(
            maxsize=config.max_cached_sessions * config.max_messages_per_session,
            ttl=config.message_cache_ttl
        )
        
        self.state_cache = TTLCache(
            maxsize=config.max_cached_sessions,
            ttl=config.state_cache_ttl
        )
        
        self.vector_cache = LRUCache(maxsize=1000) if config.enable_vector_cache else None
        
        # Track cache statistics
        self.stats = {
            'message_hits': 0,
            'message_misses': 0,
            'state_hits': 0,
            'state_misses': 0,
            'vector_hits': 0,
            'vector_misses': 0
        }

    def _get_session_key(self, user_id: str, session_id: str, agent_id: Optional[str] = None) -> str:
        """Generate cache key for a session"""
        if agent_id:
            return f"{user_id}:{session_id}:{agent_id}"
        return f"{user_id}:{session_id}"

    async def get_chat_messages(
        self,
        user_id: str,
        session_id: str,
        agent_id: str,
        storage: ChatStorage
    ) -> List[ConversationMessage]:
        """Get chat messages with caching"""
        cache_key = self._get_session_key(user_id, session_id, agent_id)
        
        # Check cache first
        cached_messages = self.message_cache.get(cache_key)
        if cached_messages is not None:
            self.stats['message_hits'] += 1
            return cached_messages
            
        self.stats['message_misses'] += 1
        
        # Fetch from storage
        messages = await storage.fetch_chat(user_id, session_id, agent_id)
        
        # Cache the result
        self.message_cache[cache_key] = messages
        
        return messages

    async def save_chat_message(
        self,
        message: ConversationMessage,
        user_id: str,
        session_id: str,
        agent_id: str,
        storage: ChatStorage
    ) -> bool:
        """Save chat message and update cache"""
        try:
            # Save to storage first
            success = await storage.save_chat_message(
                user_id=user_id,
                session_id=session_id,
                agent_id=agent_id,
                new_message=message
            )
            
            if not success:
                return False
                
            # Update cache
            cache_key = self._get_session_key(user_id, session_id, agent_id)
            cached_messages = self.message_cache.get(cache_key, [])
            cached_messages.append(message)
            
            # Trim cache if needed
            if len(cached_messages) > self.config.max_messages_per_session:
                cached_messages = cached_messages[-self.config.max_messages_per_session:]
                
            self.message_cache[cache_key] = cached_messages
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving chat message to cache: {str(e)}")
            return False

    async def get_state(
        self,
        user_id: str,
        session_id: str,
        state_manager
    ) -> Optional[Dict[str, Any]]:
        """Get session state with caching"""
        cache_key = f"state:{self._get_session_key(user_id, session_id)}"
        
        # Check cache
        cached_state = self.state_cache.get(cache_key)
        if cached_state is not None:
            self.stats['state_hits'] += 1
            return cached_state
            
        self.stats['state_misses'] += 1
        
        # Fetch from state manager
        state = await state_manager.get_system_state()
        if state:
            self.state_cache[cache_key] = state
            
        return state

    async def update_state(
        self,
        user_id: str,
        session_id: str,
        state: Dict[str, Any],
        state_manager
    ) -> bool:
        """Update session state and cache"""
        try:
            # Update state manager
            success = await state_manager.save_system_state(
                state_type="session_state",
                state_data=state
            )
            
            if not success:
                return False
                
            # Update cache
            cache_key = f"state:{self._get_session_key(user_id, session_id)}"
            self.state_cache[cache_key] = state
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating state cache: {str(e)}")
            return False

    async def cache_vector_search(
        self,
        query: str,
        results: List[Dict[str, Any]],
        ttl: Optional[int] = None
    ) -> None:
        """Cache vector search results"""
        if not self.config.enable_vector_cache:
            return
            
        cache_key = f"vector:{query}"
        self.vector_cache[cache_key] = {
            'results': results,
            'timestamp': datetime.now(),
            'ttl': ttl or self.config.message_cache_ttl
        }

    async def get_cached_vector_search(
        self,
        query: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached vector search results"""
        if not self.config.enable_vector_cache:
            return None
            
        cache_key = f"vector:{query}"
        cached = self.vector_cache.get(cache_key)
        
        if cached is None:
            self.stats['vector_misses'] += 1
            return None
            
        # Check TTL
        if cached['ttl']:
            age = (datetime.now() - cached['timestamp']).total_seconds()
            if age > cached['ttl']:
                del self.vector_cache[cache_key]
                return None
                
        self.stats['vector_hits'] += 1
        return cached['results']

    async def clear_session_cache(self, user_id: str, session_id: str):
        """Clear all cached data for a session"""
        base_key = self._get_session_key(user_id, session_id)
        
        # Clear message cache
        message_keys = [k for k in self.message_cache if k.startswith(base_key)]
        for key in message_keys:
            del self.message_cache[key]
            
        # Clear state cache
        state_key = f"state:{base_key}"
        if state_key in self.state_cache:
            del self.state_cache[state_key]

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get current cache statistics"""
        return {
            **self.stats,
            'message_cache_size': len(self.message_cache),
            'state_cache_size': len(self.state_cache),
            'vector_cache_size': len(self.vector_cache) if self.vector_cache else 0
        }