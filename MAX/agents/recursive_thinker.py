from typing import Dict, List, Union, AsyncIterable, Optional, Any, TypeVar, Generic, Protocol, AsyncGenerator
from dataclasses import dataclass, replace
from datetime import datetime
from unittest.mock import AsyncMock
import os
from anthropic import AsyncAnthropic, Anthropic

from MAX.agents.agent import Agent, AgentOptions
from MAX.types import (
    ConversationMessage,
    ParticipantRole,
    DataCategory,
    MessageType,
    MessageContent,
    AgentProviderType,
    MemoryEntry
)
from MAX.llms.anthropic import AnthropicLLM
from MAX.config.llms.base import BaseLlmConfig, ResourceConfig
from MAX.config.llms.anthropic import AnthropicConfig
from MAX.llms.utils.exceptions import LLMConfigError
from MAX.agents.options import RecursiveThinkerOptions
from MAX.llms.base import AsyncLLMBase
from MAX.utils.logger import Logger
from MAX.storage import ChatStorage
from MAX.managers import MemorySystem

logger = Logger.get_logger()

# Define generic type for message content
T = TypeVar('T')

class LLMProvider(Protocol):
    async def generate(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        stream: bool = False
    ) -> Union[str, AsyncIterable[MessageContent]]: ...

@dataclass
class RecursiveThinkerOptions(AgentOptions):
    max_recursion_depth: int = 3
    min_confidence_threshold: float = 0.7
    model_id: str = "claude-3-haiku-20240307"
    temperature: float = 0.7
    max_tokens: int = 4096
    streaming: bool = True
    storage: Optional[ChatStorage] = None
    memory_system: Optional[MemorySystem] = None  # Use existing MemorySystem
    resources: ResourceConfig = None

    
@dataclass 
class SimpleMemory:
    """Simple memory store for demonstration"""
    def __init__(self):
        self.conversations: Dict[str, List[ConversationMessage]] = {}
    
    async def add(self, session_id: str, message: ConversationMessage) -> None:
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        self.conversations[session_id].append(message)
    
    async def get_history(self, session_id: str, limit: int = 5) -> List[ConversationMessage]:
        return self.conversations.get(session_id, [])[-limit:]

class RecursiveThinkerAgent(Agent):
    """
    Agent that uses recursive thinking to break down and solve problems
    while maintaining conversation memory.
    """
    
    def __init__(self, options: RecursiveThinkerOptions):
        self.streaming = getattr(options, 'streaming', False)
        self.max_recursion_depth = options.max_recursion_depth
        self.min_confidence_threshold = options.min_confidence_threshold
        self.conversation_memories = {}
        self._current_recursion_depth = 0
        super().__init__(options)
        
        # Store all options as instance attributes
        self.options = options
        self.temperature = getattr(options, 'temperature', 0.7)
        self.max_tokens = getattr(options, 'max_tokens', 4096)
        self.storage = options.storage
        self.memory_system = options.memory_system
        
        # Initialize LLM with proper config
        llm_config = BaseLlmConfig(
            model=options.model_id,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            streaming=self.streaming,
            resources=options.resources
        )
        self._llm = self._create_llm_provider(llm_config)

    def _create_llm_provider(self, llm_config: BaseLlmConfig) -> AsyncLLMBase:
        """Create Anthropic client directly like in anthropic_agent.py"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable must be set")

        if self.streaming:
            return AsyncAnthropic(api_key=api_key)
        else:
            return Anthropic(api_key=api_key)

    async def _store_memory(
        self,
        session_id: str,
        message: ConversationMessage,
        priority: int = None
    ) -> None:
        """Store a memory entry with optional priority"""
        if priority is None:
            priority = getattr(self.options.resources, 'priority', 1)  # Get from ResourceConfig or default to 1
        
        if session_id not in self.conversation_memories:
            self.conversation_memories[session_id] = []
        self.conversation_memories[session_id].append(message)

    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: List[ConversationMessage],
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> Union[ConversationMessage, AsyncIterable[Any]]:
        try:
            # Store user's input
            user_message = ConversationMessage(
                role=ParticipantRole.USER,
                content=[{"text": input_text}],
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id,
                    "session_id": session_id
                },
                message_type=MessageType.TEXT
            )
            
            if session_id not in self.conversation_memories:
                self.conversation_memories[session_id] = []
            self.conversation_memories[session_id].append(user_message)
            
            # Format messages
            messages = [
                {
                    "role": ("user" if msg.role == ParticipantRole.USER.value else "assistant"),
                    "content": msg.content[0]["text"] if msg.content else "",
                }
                for msg in self.conversation_memories[session_id]
            ]
            messages.append({"role": "user", "content": input_text})

            context = await self._get_conversation_context(session_id)
            system_prompt = self._create_system_prompt(context, input_text)

            input_params = {
                "model": self.options.model_id,
                "messages": messages,
                "system": system_prompt,
                "max_tokens": 1000,
                "temperature": 0.7,
            }

            try:
                if self.streaming:
                    async with self._llm.messages.stream(**input_params) as stream:
                        async for event in stream:
                            if event.type == "text":
                                self.callbacks.on_llm_new_token(event.text)
                        response = await stream.get_final_message()
                        response_text = response.content[0].text
                else:
                    # For non-streaming, use create() without await
                    response = self._llm.messages.create(**input_params)
                    response_text = response.content[0].text

                response_message = ConversationMessage(
                    role=ParticipantRole.ASSISTANT,
                    content=[{"text": response_text}],
                    metadata={
                        "timestamp": datetime.now().isoformat(),
                        "user_id": user_id,
                        "session_id": session_id,
                        "recursion_depth": self._current_recursion_depth
                    },
                    message_type=MessageType.TEXT
                )
                
                self.conversation_memories[session_id].append(response_message)
                return response_message

            except Exception as e:
                raise Exception(f"Error invoking Anthropic: {str(e)}")

        except Exception as e:
            error_message = ConversationMessage(
                role=ParticipantRole.ASSISTANT,
                content=[{"text": f"Error processing request: {str(e)}"}],
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id,
                    "session_id": session_id,
                    "error": True
                },
                message_type=MessageType.TEXT
            )
            if session_id not in self.conversation_memories:
                self.conversation_memories[session_id] = []
            self.conversation_memories[session_id].append(error_message)
            return error_message

    async def _get_conversation_context(self, session_id: str) -> str:
        """Get formatted conversation context"""
        if session_id not in self.conversation_memories:
            return ""
        messages = self.conversation_memories[session_id][-5:]  # Last 5 messages
        return "\n".join([
            f"{msg.role}: {msg.content[0]['text']}"
            for msg in messages
        ])

    def _create_system_prompt(self, context: str, input_text: str) -> str:
        return f"""Previous conversation:
{context}

You are a recursive thinker that:
1. Breaks problems into clear subcomponents
2. Analyzes each component from multiple angles
3. Considers implications and trade-offs
4. Synthesizes insights into a cohesive solution
5. Recursively improves by questioning assumptions

Current user input: {input_text}
Confidence threshold: {self.min_confidence_threshold}
Current recursion depth: {self._current_recursion_depth}/{self.max_recursion_depth}
"""

    def is_streaming_enabled(self) -> bool:
        return self.streaming

    async def _get_llm(self) -> AsyncLLMBase:
        """Get or create properly typed LLM instance"""
        return self._llm

    async def _get_conversation_context(
        self,
        session_id: str,
        limit: int = 5
    ) -> List[ConversationMessage]:
        """Get recent conversation context with proper typing"""
        if not self.storage:
            return []
            
        messages = await self.storage.get_chat_history(
            session_id=session_id,
            limit=limit
        )
        return messages

    async def _store_memory(
        self,
        data: Dict[str, Any],
        category: DataCategory,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store memory with proper typing"""
        if self.memory_system:
            await self.memory_system.store(
                data=data,
                category=category,
                metadata=metadata or {}
            )

"""
RecursiveThinkerAgent with Conversation Memory

This agent implements a conversation memory system that enables:
- Session-based memory storage
- Context retrieval for multi-step reasoning
- Persistent conversation history within sessions

Key Components:
- conversation_memories: Dict storing messages by session_id
- _store_memory: Adds new messages to memory
- _get_conversation_context: Retrieves recent context
- process_request: Handles message processing with context

Example Usage:
    agent = RecursiveThinkerAgent(options)
    
    # First interaction
    response1 = await agent.process_request(
        "My name is Alice",
        user_id="user1",
        session_id="session1",
        chat_history=[]
    )
    
    # Second interaction (will remember Alice)
    response2 = await agent.process_request(
        "What's my name?",
        user_id="user1",
        session_id="session1",
        chat_history=[response1]
    )
"""
