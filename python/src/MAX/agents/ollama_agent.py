from typing import List, Dict, Optional, AsyncIterable, Any
from MAX.agents import Agent, AgentOptions
from MAX.types import ConversationMessage, ParticipantRole
from MAX.utils import Logger  # Using the existing Logger
import ollama
from dataclasses import dataclass

@dataclass
class OllamaAgentOptions(AgentOptions):
    model_id: str = "llama3.1:8b-instruct-q8_0"
    streaming: bool = False
    temperature: float = 0.7
    top_p: float = 0.9

class OllamaAgent(Agent):
    def __init__(self, options: OllamaAgentOptions):
        super().__init__(options)
        self.model_id = options.model_id
        self.streaming = options.streaming
        self.temperature = options.temperature
        self.top_p = options.top_p
        self.logger = Logger()  # Using the existing Logger
        
        # Log initialization
        self.logger.info(f"Initialized OllamaAgent with model: {self.model_id}")
        self.logger.info(f"Streaming: {self.streaming}")
        self.logger.info(f"Temperature: {self.temperature}")

    async def handle_streaming_response(self, messages: List[Dict[str, str]]) -> ConversationMessage:
        self.logger.debug("Starting streaming response")
        text = ''
        try:
            response = ollama.chat(
                model=self.model_id,
                messages=messages,
                stream=True,
                options={
                    "temperature": self.temperature,
                    "top_p": self.top_p
                }
            )
            
            for part in response:
                text += part['message']['content']
                self.callbacks.on_llm_new_token(part['message']['content'])
                self.logger.debug(f"Received token: {part['message']['content'][:20]}...")

            self.logger.info("Streaming response completed")
            return ConversationMessage(
                role=ParticipantRole.ASSISTANT.value,
                content=[{"text": text}]
            )

        except Exception as error:
            self.logger.error(f"Error in streaming response: {str(error)}")
            raise error

    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: List[ConversationMessage],
        additional_params: Optional[Dict[str, str]] = None
    ) -> ConversationMessage | AsyncIterable[Any]:
        self.logger.info(f"Processing request for user {user_id}, session {session_id}")
        self.logger.debug(f"Input text: {input_text}")
        
        # Log chat history if present
        if chat_history:
            self.logger.print_chat_history(chat_history, self.model_id)

        messages = [
            {"role": msg.role, "content": msg.content[0]['text']}
            for msg in chat_history
        ]
        messages.append({"role": ParticipantRole.USER.value, "content": input_text})

        try:
            if self.streaming:
                self.logger.info("Using streaming mode")
                return await self.handle_streaming_response(messages)
            else:
                self.logger.info("Using non-streaming mode")
                response = ollama.chat(
                    model=self.model_id,
                    messages=messages,
                    options={
                        "temperature": self.temperature,
                        "top_p": self.top_p
                    }
                )
                self.logger.info("Request processed successfully")
                return ConversationMessage(
                    role=ParticipantRole.ASSISTANT.value,
                    content=[{"text": response['message']['content']}]
                )
        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}")
            raise

    def is_streaming_enabled(self) -> bool:
        return self.streaming