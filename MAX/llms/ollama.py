"""Ollama LLM implementation with integrated configuration."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, AsyncIterator, Any, AsyncIterable
import aiohttp
import json
from enum import Enum
from datetime import datetime
import ollama  # Official Ollama client
from MAX.types.base_types import (
    ConversationMessage,
    AgentResponse,
    ParticipantRole,
    MessageType
)
from MAX.llms.base import AsyncLLMBase

class OllamaModelType(str, Enum):
    """Available Ollama model types with their corresponding model names"""
    GENERAL = "llama3.1:8b-instruct-q8_0"
    CODE = "llama3.1:8b-instruct-q8_0"
    FAST = "llama3.1:8b-instruct-q8_0"
    INSTRUCT = "llama3.1:8b-instruct-q8_0"

@dataclass
class OllamaLLM:
    """Combined Ollama configuration and implementation"""
    model_type: Union[OllamaModelType, str] = OllamaModelType.GENERAL
    temperature: float = 0.7
    max_tokens: int = 2048
    streaming: bool = False
    context_window: int = 4096
    stop_sequences: Optional[List[str]] = None
    timeout: float = 30.0
    api_base: str = "http://localhost:11434/api"
    max_parallel_calls: int = 3
    local_only: bool = True

    @property
    def model(self) -> str:
        """Get the actual model name from the model type"""
        if isinstance(self.model_type, str):
            return self.model_type
        return self.model_type.value

    async def _make_request(self, endpoint: str, data: Dict) -> Dict:
        """Make regular API request to Ollama"""
        url = f"{self.api_base}/{endpoint}"
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=data) as resp:
                if resp.status != 200:
                    raise Exception(f"Ollama API error: {resp.status} - {await resp.text()}")
                return await resp.json()

    async def _make_streaming_request(self, endpoint: str, data: Dict) -> AsyncIterator[Dict]:
        """Make streaming API request to Ollama"""
        url = f"{self.api_base}/{endpoint}"
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=data) as resp:
                if resp.status != 200:
                    raise Exception(f"Ollama API error: {resp.status} - {await resp.text()}")
                async for line in resp.content:
                    if line:
                        yield json.loads(line)

    def _create_request_data(self, system_prompt: str, messages: List[Dict[str, str]], stream: bool = False) -> Dict:
        """Create request data dictionary"""
        return {
            "model": self.model,
            "messages": messages,
            "system": system_prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": self.stop_sequences,
            "stream": stream,
        }

    def _create_response(self, content: str, metadata: Dict = None) -> AgentResponse:
        """Create standardized agent response"""
        return AgentResponse(
            content=content,
            confidence=1.0,  # Ollama doesn't provide confidence scores
            metadata={
                "model": self.model,
                "finish_reason": "stop",
                **(metadata or {})
            },
            timestamp=datetime.now(),
            message_type=MessageType.TEXT
        )

    async def generate(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
    ) -> Union[AgentResponse, AsyncIterator[ConversationMessage]]:
        """Generate response from Ollama"""
        data = self._create_request_data(system_prompt, messages, stream)
        
        if not stream:
            response = await self._make_request("chat", data)
            if "message" in response and "content" in response["message"]:
                return self._create_response(response["message"]["content"])
            return self._create_response(str(response))
        else:
            async def stream_response():
                async for chunk in self._make_streaming_request("chat", data):
                    if "message" in chunk and "content" in chunk["message"]:
                        yield ConversationMessage(
                            role=ParticipantRole.ASSISTANT,
                            content=[{"text": chunk["message"]["content"]}],
                            metadata={"model": self.model},
                            message_type=MessageType.TEXT
                        )
            return stream_response()

    @classmethod
    def create(cls, **kwargs) -> 'OllamaLLM':
        """Factory method to create OllamaLLM instances"""
        return cls(**kwargs)

class AsyncOllamaLLM(OllamaLLM, AsyncLLMBase):
    """Async Ollama LLM implementation using official client."""
    
    def __init__(
        self, 
        model_type: Union[OllamaModelType, str] = OllamaModelType.GENERAL,
        **kwargs
    ):
        """
        Initialize Ollama LLM.
        
        Args:
            model_type: Type of model to use (from OllamaModelType enum or string)
            **kwargs: Additional configuration options
        """
        super().__init__(model_type=model_type, **kwargs)
        self.client = ollama.AsyncClient(host=self.api_base.replace("/api", ""))

    async def _generate(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        stream: Optional[bool] = None,
        **kwargs
    ) -> Union[str, AsyncIterable[str]]:
        """
        Generate text using Ollama API.
        
        Args:
            prompt: The input prompt
            messages: Optional list of conversation messages
            stream: Whether to stream the response (overrides instance setting)
            **kwargs: Additional arguments to pass to Ollama
        """
        try:
            # Use instance values if not provided
            stream = stream if stream is not None else self.streaming

            # Format messages for Ollama
            formatted_messages = []
            if messages:
                formatted_messages = messages
            else:
                formatted_messages = [{"role": "user", "content": prompt}]

            options = {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "stop": self.stop_sequences,
                **kwargs
            }

            if stream:
                async def generate_stream():
                    async for chunk in self.client.chat(
                        model=self.model_type.value if hasattr(self.model_type, 'value') else self.model_type,
                        messages=formatted_messages,
                        stream=True,
                        options=options
                    ):
                        if chunk and "message" in chunk and "content" in chunk["message"]:
                            yield chunk["message"]["content"]
                return generate_stream()
            else:
                response = await self.client.chat(
                    model=self.model_type.value if hasattr(self.model_type, 'value') else self.model_type,
                    messages=formatted_messages,
                    stream=False,
                    options=options
                )
                return response["message"]["content"]

        except Exception as e:
            raise Exception(f"Ollama API error: {str(e)}")

    async def acomplete(
        self,
        prompt: str,
        **kwargs
    ) -> Any:
        """Complete a prompt using Ollama."""
        return await self._generate(prompt=prompt, **kwargs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.close()
