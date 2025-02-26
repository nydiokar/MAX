"""Ollama LLM implementation with integrated configuration."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, AsyncIterator
import aiohttp
import json
from enum import Enum
from datetime import datetime

from MAX.types.base_types import (
    ConversationMessage,
    AgentResponse,
    ParticipantRole,
    MessageType
)

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
