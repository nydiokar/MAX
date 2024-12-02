from typing import Dict, Any, Union, AsyncIterable, Optional
from dataclasses import dataclass
from MAX.types import ConversationMessage, ParticipantRole
from MAX.utils import Logger
from llama_cpp import Llama

@dataclass
class LlamaConfig:
    model_path: str
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    n_threads: int = 4
    n_ctx: int = 2048

    def post_init(self):
        # Validate parameters
        if not (0 <= self.temperature <= 1):
            raise ValueError("Temperature must be between 0 and 1.")
        if not (0 < self.max_tokens <= 2048):
            raise ValueError("Max tokens must be between 1 and 2048.")
        if not (0 <= self.top_p <= 1):
            raise ValueError("Top_p must be between 0 and 1.")
        if not (0 < self.top_k <= 100):
            raise ValueError("Top_k must be a positive integer.")
        if not (0 < self.repeat_penalty <= 2):
            raise ValueError("Repeat penalty must be between 0 and 2.")

class LlamaMixin:
    """Mixin to add Llama capabilities to any agent."""
    
    def init_llama(self, llama_config: LlamaConfig):
        """Initialize Llama-specific attributes."""
        self.llama_config = llama_config
        self.llama_model = Llama(
            model_path=llama_config.model_path,
            n_ctx=llama_config.n_ctx,
            n_threads=llama_config.n_threads
        )
        self.logger = Logger()

    async def generate_llama_response(
        self,
        prompt: str,
        system_prompt: str,
        streaming: bool = False
    ) -> Union[str, AsyncIterable[str]]:
        """
        Generate a response using the Llama model.

        Args:
            prompt (str): The user input prompt to generate a response for.
            system_prompt (str): The system prompt to provide context for the assistant.
            streaming (bool): If True, the response will be generated in a streaming manner.

        Returns:
            Union[str, AsyncIterable[str]]: The generated response as a string or an async iterable of strings if streaming is enabled.

        Raises:
            Exception: If there is an error during response generation.
        """
        try:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
            
            if streaming:
                async def stream_response():
                    async for token in self.llama_model.create_completion(
                        full_prompt,
                        max_tokens=self.llama_config.max_tokens,
                        temperature=self.llama_config.temperature,
                        top_p=self.llama_config.top_p,
                        top_k=self.llama_config.top_k,
                        repeat_penalty=self.llama_config.repeat_penalty,
                        stream=True
                    ):
                        yield token['choices'][0]['text']
                return stream_response()
            else:
                response = self.llama_model.create_completion(
                    full_prompt,
                    max_tokens=self.llama_config.max_tokens,
                    temperature=self.llama_config.temperature,
                    top_p=self.llama_config.top_p,
                    top_k=self.llama_config.top_k,
                    repeat_penalty=self.llama_config.repeat_penalty,
                    stream=False
                )
                return response['choices'][0]['text']

        except Exception as e:
            self.logger.error(f"Error generating Llama response: {str(e)}")
            raise e