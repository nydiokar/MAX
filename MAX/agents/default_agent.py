from typing import Dict, List, Union, AsyncIterable, Optional, Any
from datetime import datetime
from MAX.agents.agent import Agent, AgentOptions, AgentCallbacks
from MAX.types import ConversationMessage
from MAX.llms.ollama import AsyncOllamaLLM, OllamaModelType

class DefaultAgent(Agent):
    """A simple, responsive Ollama-based agent for default handling."""
    
    def __init__(self, options: Optional[AgentOptions] = None):
        if options is None:
            options = AgentOptions(
                name="DEFAULT",
                description="A responsive general-purpose agent for handling default cases.",
                save_chat=True,
                resources=None
            )
        super().__init__(options)

    def _create_llm_provider(self) -> AsyncOllamaLLM:
        """Create Ollama LLM provider with default settings."""
        return AsyncOllamaLLM(
            model_type=OllamaModelType.GENERAL,
            streaming=False
        )

    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: Optional[List[ConversationMessage]] = None,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> Union[ConversationMessage, AsyncIterable[Any]]:
        """Process a request using Ollama."""
        try:
            # Format messages for LLM
            messages = []
            if chat_history:
                messages.extend([
                    {"role": "user" if msg.role == "user" else "assistant", 
                     "content": msg.content[0]["text"]}
                    for msg in chat_history
                ])
            messages.append({"role": "user", "content": input_text})
            
            # Get response from LLM
            response_text = await self._llm._generate(
                prompt=input_text,
                messages=messages
            )

            # Create response message
            return ConversationMessage(
                role="assistant",
                content=[{"text": response_text}],
                metadata={
                    "user_id": user_id,
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat()
                }
            )

        except Exception as e:
            return ConversationMessage(
                role="assistant",
                content=[{"text": f"Error: {str(e)}"}],
                metadata={
                    "user_id": user_id,
                    "session_id": session_id,
                    "error": True
                }
            ) 