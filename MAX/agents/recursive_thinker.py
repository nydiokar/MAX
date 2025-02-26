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
from MAX.llms.ollama import OllamaLLM, OllamaModelType
from MAX.utils.logger import Logger
from MAX.storage import ChatStorage
from MAX.managers import MemorySystem

logger = Logger.get_logger()

@dataclass
class RecursiveThinkerOptions(AgentOptions):
    """Options for basic conversation agent"""
    streaming: bool = True
    storage: Optional[ChatStorage] = None
    memory_system: Optional[MemorySystem] = None
    model_type: Union[str, OllamaModelType] = OllamaModelType.GENERAL

    def __post_init__(self):
        super().__post_init__() if hasattr(super(), '__post_init__') else None
        if not self.name:
            self.name = "ChatAgent"
        if not self.description:
            self.description = "A basic conversational agent powered by Ollama"

class RecursiveThinkerAgent(Agent):
    """Basic conversational agent implementation."""
    
    def __init__(self, options: RecursiveThinkerOptions):
        super().__init__(options)
        self.options = options
        self.conversation_memories = {}
        
    def _create_llm_provider(self) -> AsyncLLMBase:
        """Create an instance of the LLM provider"""
        try:
            # Handle string model names directly
            if isinstance(self.options.model_type, str):
                model_type = self.options.model_type
            else:
                # For enum types, get the string value directly
                model_type = self.options.model_type.value if hasattr(self.options.model_type, 'value') else str(self.options.model_type)
            
            logger.debug(f"Creating LLM with model type: {model_type}")
            return OllamaLLM(
                model_type=model_type,
                streaming=self.options.streaming
            )
        except Exception as e:
            logger.error(f"Error creating LLM provider: {str(e)}")
            raise
        
    async def initialize(self) -> bool:
        """Initialize the agent and create LLM instance"""
        self._llm = self._create_llm_provider()
        return True

    async def _store_memory(
        self,
        session_id: str,
        message: ConversationMessage,
        agent_id: str = None,
        data: Optional[Dict[str, Any]] = None,
        max_history_size: Optional[int] = None
    ) -> None:
        """Store message in all available memory systems"""
        # Store in conversation memory with size limit
        if session_id not in self.conversation_memories:
            self.conversation_memories[session_id] = []
        
        if max_history_size and len(self.conversation_memories[session_id]) >= max_history_size:
            self.conversation_memories[session_id].pop(0)  # Remove oldest
        self.conversation_memories[session_id].append(message)

        # Store in persistent storage if available
        if self.options.storage:
            try:
                await self.options.storage.save_chat_message(
                    user_id=message.metadata.get("user_id", "unknown"),
                    session_id=session_id,
                    agent_id=agent_id or self.id,
                    new_message=message,
                    max_history_size=max_history_size
                )
            except Exception as e:
                logger.error(f"Failed to store message in storage: {e}")

        # Store in memory system if available
        if self.options.memory_system and data:
            try:
                await self.options.memory_system.store(
                    data=data,
                    category=DataCategory.CONVERSATION,
                    metadata=message.metadata or {}
                )
            except Exception as e:
                logger.error(f"Failed to store in memory system: {e}")

    async def _get_conversation_context(
        self,
        session_id: str, 
        user_id: str = None,
        limit: int = 5
    ) -> List[ConversationMessage]:
        """Get conversation history from the most appropriate source"""
        messages: List[ConversationMessage] = []
        
        if self.options.storage:
            try:
                messages = await self.options.storage.fetch_chat(
                    user_id=user_id or "unknown",
                    session_id=session_id,
                    agent_id=self.id,
                    max_history_size=limit
                )
            except Exception as e:
                logger.error(f"Failed to fetch from storage: {e}")

        # Fallback to in-memory if storage failed or returned nothing
        if not messages and session_id in self.conversation_memories:
            messages = self.conversation_memories[session_id][-limit:]
        
        return messages

    async def _format_messages(
        self,
        messages: List[ConversationMessage]
    ) -> List[Dict[str, str]]:
        """Format messages for Ollama API"""
        formatted = []
        for msg in messages:
            role = "user" if msg.role == ParticipantRole.USER else "assistant"
            content = msg.content[0]["text"] if msg.content else ""
            formatted.append({
                "role": role,
                "content": content
            })
        return formatted

    async def _extract_confidence(self, response_text: str) -> float:
        """Extract confidence score from response"""
        try:
            if "Confidence: " in response_text:
                confidence_str = response_text.split("Confidence: ")[1].split()[0]
                return float(confidence_str)
        except:
            pass
        return 0.5  # Default confidence

    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: Optional[List[ConversationMessage]] = None,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> Union[ConversationMessage, AsyncIterable[Any]]:
        """Process a chat request"""
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
            await self._store_memory(session_id, user_message, max_history_size=5)

            # Get conversation history
            history = await self._get_conversation_context(
                session_id=session_id,
                user_id=user_id,
                limit=5
            )
            if chat_history:
                history.extend(chat_history)

            # Format messages for LLM
            messages = await self._format_messages(history)
            messages.append({"role": "user", "content": input_text})
            
            system_prompt = self._create_system_prompt(
                "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages[:-1]),
                input_text
            )

            # Get LLM response
            try:
                response = await self._llm.generate(
                    system_prompt=system_prompt,
                    messages=messages,
                    stream=self.options.streaming
                )
                
                if self.options.streaming:
                    response_text = ""
                    async for chunk in response:
                        if isinstance(chunk, ConversationMessage):
                            response_text += chunk.content[0]["text"]
                            if self.callbacks:
                                self.callbacks.on_llm_new_token(chunk.content[0]["text"])
                else:
                    # Handle AgentResponse type
                    response_text = response.content if isinstance(response, AgentResponse) else str(response)

                # Create response message
                response_message = ConversationMessage(
                    role=ParticipantRole.ASSISTANT,
                    content=[{"text": response_text}],  # Now properly formatted
                    metadata={
                        "timestamp": datetime.now().isoformat(),
                        "user_id": user_id,
                        "session_id": session_id
                    },
                    message_type=MessageType.TEXT
                )

                # Store in memory
                await self._store_memory(
                    session_id=session_id,
                    message=response_message,
                    data={"message": response_text},
                    max_history_size=5
                )
                
                return response_message

            except Exception as e:
                raise Exception(f"Error invoking Ollama: {str(e)}")

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
            await self._store_memory(session_id, error_message)
            return error_message

    def _create_system_prompt(self, context: str, input_text: str) -> str:
        """Create base system prompt"""
        return f"""Previous conversation:
{context}

You are a helpful assistant that:
1. Provides clear and direct answers
2. Maintains conversation context
3. Formats responses in a clear structure

Instructions:
1. Always be polite and helpful
2. Keep responses concise but informative
3. Use context when relevant

Current user input: {input_text}
"""

    def is_streaming_enabled(self) -> bool:
        """Check if streaming is enabled"""
        return self.options.streaming

    async def aclose(self) -> None:
        """Clean up resources"""
        # Currently no cleanup needed for Ollama
        pass
