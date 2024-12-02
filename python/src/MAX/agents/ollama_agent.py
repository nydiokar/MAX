from typing import List, Dict, Optional, AsyncIterable, Any, Tuple, Union
from MAX.agents import Agent, AgentOptions
from MAX.types import ConversationMessage, ParticipantRole
from MAX.utils import Logger
import ollama
from dataclasses import dataclass

@dataclass
class OllamaAgentOptions(AgentOptions):
    model_id: str = "llama3.1:8b-instruct-q8_0"
    streaming: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    system_prompt: Optional[str] = None

class OllamaAgent(Agent):
    def __init__(self, options: OllamaAgentOptions):
        super().__init__(options)
        self.model_id = options.model_id
        self.streaming = options.streaming
        self.temperature = options.temperature
        self.top_p = options.top_p
        self.system_prompt = options.system_prompt
        self.logger = Logger()

    async def generate_response(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generic method to generate responses with custom parameters"""
        try:
            messages = []
            if system_prompt or self.system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt or self.system_prompt
                })
            
            messages.append({"role": "user", "content": prompt})
            
            response = ollama.chat(
                model=self.model_id,
                messages=messages,
                options={
                    "temperature": temperature or self.temperature,
                    "top_p": self.top_p
                }
            )
            
            return response['message']
            
        except Exception as e:
            self.logger.error(f"Error in generate_response: {str(e)}")
            raise

    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: List[ConversationMessage],
        additional_params: Optional[Dict[str, str]] = None
    ) -> Union[ConversationMessage, AsyncIterable[Any]]:
        """Process a request with chat history"""
        try:
            messages = []
            
            # Add system prompt if specified
            if self.system_prompt:
                messages.append({
                    "role": "system",
                    "content": self.system_prompt
                })
            
            # Add chat history
            messages.extend([
                {"role": msg.role, "content": msg.content[0]['text']}
                for msg in chat_history
            ])
            
            # Add current input
            messages.append({
                "role": ParticipantRole.USER.value,
                "content": input_text
            })

            if self.streaming:
                return await self.handle_streaming_response(messages)
            else:
                response = ollama.chat(
                    model=self.model_id,
                    messages=messages,
                    options={
                        "temperature": self.temperature,
                        "top_p": self.top_p
                    }
                )
                
                return ConversationMessage(
                    role=ParticipantRole.ASSISTANT.value,
                    content=[{
                        "text": response['message'].get('content', ''),
                        "metadata": {
                            "model": self.model_id,
                            "temperature": self.temperature,
                            **(additional_params if additional_params else {})
                        }
                    }]
                )
        except Exception as e:
            self.logger.error(f"Error in process_request: {str(e)}")
            raise

    async def handle_streaming_response(self, messages: List[Dict[str, str]]) -> AsyncIterable[Any]:
        """Handle streaming responses"""
        try:
            stream = ollama.chat(
                model=self.model_id,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "top_p": self.top_p
                },
                stream=True
            )
            
            for chunk in stream:
                yield chunk['message']['content']
                
        except Exception as e:
            self.logger.error(f"Error in streaming response: {str(e)}")
            raise
