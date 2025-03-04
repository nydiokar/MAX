"""RecursiveThinker agent implementation."""
from dataclasses import dataclass
from typing import Dict, List, Union, AsyncIterable, Optional, Any
from datetime import datetime

from MAX.agents.agent import Agent, AgentOptions, AgentResponse
from MAX.types import (
    ConversationMessage,
    ParticipantRole,
    MessageType,
    DataCategory
)
from MAX.llms.base import AsyncLLMBase
from MAX.llms.ollama import AsyncOllamaLLM, OllamaModelType
from MAX.utils.logger import Logger
from MAX.storage import ChatStorage
from MAX.storage.utils.storage_factory import StorageFactory

logger = Logger.get_logger()

@dataclass
class RecursiveThinkerOptions(AgentOptions):
    """Options for recursive thinking agent"""
    streaming: bool = True
    storage: Optional[ChatStorage] = None
    model_type: Union[str, OllamaModelType] = OllamaModelType.GENERAL
    mongo_uri: str = "mongodb://localhost:27017"
    db_name: str = "max_agents"
    collection_name: str = "conversations"
    max_history_size: int = 100
    knowledge_threshold: int = 50  # Minimum length for knowledge-worthy responses
    llm: Optional[AsyncLLMBase] = None  # Add LLM option

    def __post_init__(self):
        super().__post_init__() if hasattr(super(), '__post_init__') else None
        if not self.name:
            self.name = "RecursiveThinker"
        if not self.description:
            self.description = "An agent that thinks recursively about problems"
        
        if not self.storage:
            storage_factory = StorageFactory(self.mongo_uri, self.db_name)
            self.storage = storage_factory.get_chat_storage()
            
        if not self.llm:
            self.llm = AsyncOllamaLLM(
                model_type=self.model_type,
                temperature=0.7,
                max_tokens=2048
            )

class RecursiveThinkerAgent(Agent):
    """Recursive thinking agent implementation."""
    
    def __init__(self, options: RecursiveThinkerOptions):
        super().__init__(options)
        self.options = options
        self.conversation_memories = {}
        self._llm = options.llm
        
    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: Optional[List[ConversationMessage]] = None,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> Union[AgentResponse, AsyncIterable[Any]]:
        """Process a request with recursive thinking approach."""
        try:
            # 1. Get conversation context and relevant memories
            context = await self._get_relevant_context(
                input_text=input_text,
                session_id=session_id,
                chat_history=chat_history
            )

            # 2. Process the request with context
            response = await self._llm.generate(
                system_prompt=self._create_system_prompt(context, input_text),
                messages=chat_history or [],
                stream=self.options.streaming
            )

            # Convert to AgentResponse
            agent_response = AgentResponse(
                message=response,
                message_type=MessageType.TEXT,
                data_category=DataCategory.CONVERSATION,
                metadata={
                    "user_id": user_id,
                    "session_id": session_id,
                    "agent_id": self.id,
                    "timestamp": datetime.now().isoformat()
                }
            )

            # Store with proper response type
            if self._is_knowledge_worthy(response):
                await self._store_interaction(
                    message=agent_response,
                    user_id=user_id,
                    session_id=session_id,
                    is_important=True
                )
            else:
                await self._store_interaction(
                    message=agent_response,
                    user_id=user_id,
                    session_id=session_id,
                    is_important=False
                )

            return agent_response

        except Exception as e:
            logger.error(f"Error in process_request: {str(e)}")
            error_response = self._create_error_response(str(e), user_id, session_id)
            await self._store_interaction(error_response, user_id, session_id, is_important=False)
            return error_response

    def _is_knowledge_worthy(self, message: ConversationMessage) -> bool:
        """Determine if a message should be stored in long-term memory."""
        if not isinstance(message, ConversationMessage):
            return False
            
        return (
            message.role == ParticipantRole.ASSISTANT and
            not message.metadata.get("error", False) and
            len(message.content[0]["text"]) > self.options.knowledge_threshold and
            not message.metadata.get("is_fallback", False)
        )

    async def _get_relevant_context(
        self,
        input_text: str,
        session_id: str,
        chat_history: Optional[List[ConversationMessage]] = None,
    ) -> str:
        """Get relevant context combining memory and chat history."""
        try:
            context_parts = []
            
            # 1. Get recent conversation history
            if chat_history:
                recent_context = [
                    msg.content[0]["text"] 
                    for msg in chat_history[-3:]
                ]
                context_parts.extend(recent_context)

            # 2. Search semantic memory
            memory_results = await self._llm.memory.search(
                query=input_text,
                limit=5
            )
            if memory_results:
                memory_context = [result.text for result in memory_results]
                context_parts.extend(memory_context)

            # 3. Get local conversation memory as fallback
            if not context_parts and session_id in self.conversation_memories:
                local_context = [
                    msg.content[0]["text"] 
                    for msg in self.conversation_memories[session_id][-3:]
                ]
                context_parts.extend(local_context)
            
            return "\n".join(context_parts)

        except Exception as e:
            logger.error(f"Error getting context: {str(e)}")
            return ""

    async def _store_interaction(
        self,
        message: ConversationMessage,
        user_id: str,
        session_id: str,
        is_important: bool = False
    ) -> None:
        """Store interaction with importance-based storage strategy."""
        try:
            # Always store in conversation history
            if session_id not in self.conversation_memories:
                self.conversation_memories[session_id] = []
            
            if len(self.conversation_memories[session_id]) >= self.options.max_history_size:
                self.conversation_memories[session_id].pop(0)
            
            self.conversation_memories[session_id].append(message)

            # Store important interactions in semantic memory
            if is_important:
                await self._llm.memory.store(
                    text=message.content[0]["text"],
                    metadata={
                        "user_id": user_id,
                        "session_id": session_id,
                        "agent_id": self.id,
                        "timestamp": datetime.now().isoformat(),
                        "importance": "high"
                    }
                )

            # Store in persistent storage if available
            if self.options.storage:
                await self.options.storage.save_chat_message(
                    user_id=user_id,
                    session_id=session_id,
                    agent_id=self.id,
                    new_message=message,
                    max_history_size=self.options.max_history_size
                )

        except Exception as e:
            logger.error(f"Error storing interaction: {str(e)}")

    def _create_error_response(self, error_msg: str, user_id: str, session_id: str) -> ConversationMessage:
        """Create a standardized error response."""
        return ConversationMessage(
            role=ParticipantRole.ASSISTANT,
            content=[{"text": f"I encountered an error: {error_msg}. Please try rephrasing your request."}],
            metadata={
                "error": True,
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
        )

    def _create_system_prompt(self, context: str, input_text: str) -> str:
        """Create system prompt with recursive thinking approach."""
        return f"""Previous context and knowledge:
{context}

You are a recursive thinking assistant that:
1. Breaks down complex problems into smaller parts
2. Considers multiple approaches before responding
3. Maintains conversation context
4. Builds on previous knowledge

Thinking process:
1. Analyze the input carefully
2. Consider relevant context
3. Break down complex issues
4. Synthesize a clear response

Current user input: {input_text}

Provide your response in a clear, structured format.
"""

    def is_streaming_enabled(self) -> bool:
        """Check if streaming is enabled"""
        return self.options.streaming

    async def aclose(self) -> None:
        """Clean up resources"""
        if self.options.storage:
            try:
                await self.options.storage.cleanup()
                logger.info("Storage system cleaned up successfully")
            except Exception as e:
                logger.error(f"Failed to cleanup storage: {str(e)}")
