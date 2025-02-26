import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import datetime

from MAX.config.base_llm import ResourceConfig, BaseLlmConfig
from MAX.config.llms.ollama import OllamaConfig, OllamaModelType
from MAX.llms.ollama import OllamaLLM
from MAX.types.base_types import ParticipantRole, MessageType, ConversationMessage, AgentResponse

# Helper function to create OllamaConfig without triggering model property issues
def create_ollama_config(**kwargs):
    """Create OllamaConfig instance without setting model directly"""
    # Start with default values
    config_kwargs = {
        "model_type": OllamaModelType.GENERAL,
        "temperature": 0.7,
        "max_tokens": 2048,
        "streaming": False,
        "resources": None,
        "api_key": None,
        "api_base_url": None,
        "context_window": 4096,
        "stop_sequences": None,
        "timeout": 30.0
    }
    # Update with provided kwargs
    config_kwargs.update(kwargs)
    # Create instance with __new__ to bypass __init__
    config = object.__new__(OllamaConfig)
    # Set attributes directly
    for key, value in config_kwargs.items():
        setattr(config, key, value)
    return config

async def test_ollama_initialization():
    """Test Ollama configuration and initialization"""
    # Use minimal config, relying on defaults
    resources = ResourceConfig(
        local_only=True,
        priority=1,
        max_parallel_calls=3
    )
    config = create_ollama_config(resources=resources)
    
    llm = OllamaLLM(config)
    assert llm.provider_name == "ollama"
    assert llm.config.model_type == OllamaModelType.GENERAL  # Default
    assert llm.config.model == "llama3.1:8b-instruct-q8_0"  # From model_type
    assert llm.API_BASE == "http://localhost:11434/api"
    assert llm.available is True
    assert llm._current_tasks == 0

async def test_format_messages():
    """Test the _format_messages method"""
    config = create_ollama_config()
    llm = OllamaLLM(config)
    
    # Test with system prompt
    system_prompt = "You are a helpful assistant."
    messages = [{"role": "user", "content": "Hello"}]
    formatted = llm._format_messages(system_prompt, messages)
    
    assert len(formatted) == 2
    assert formatted[0]["role"] == "system"
    assert formatted[0]["content"] == system_prompt
    assert formatted[1] == messages[0]
    
    # Test without system prompt
    formatted = llm._format_messages("", messages)
    assert len(formatted) == 1
    assert formatted[0] == messages[0]

async def test_create_request_data():
    """Test the _create_request_data method"""
    config = create_ollama_config(
        model_type=OllamaModelType.CODE,
        temperature=0.5,
        max_tokens=1000,
        stop_sequences=["STOP"]
    )
    llm = OllamaLLM(config)
    
    messages = [{"role": "user", "content": "Hello"}]
    data = llm._create_request_data(messages, stream=True)
    
    assert data["model"] == "codellama"
    assert data["temperature"] == 0.5
    assert data["max_tokens"] == 1000
    assert data["stop"] == ["STOP"]
    assert data["stream"] is True
    assert data["messages"] == messages

async def test_create_response():
    """Test the _create_response method"""
    config = create_ollama_config(model_type=OllamaModelType.GENERAL)
    llm = OllamaLLM(config)
    
    content = "This is a test response"
    metadata = {"test_key": "test_value"}
    
    response = llm._create_response(content, metadata)
    
    assert isinstance(response, AgentResponse)
    assert response.content == content
    assert response.confidence == 1.0
    assert response.metadata["model"] == "llama3.1:8b-instruct-q8_0"
    assert response.metadata["finish_reason"] == "stop"
    assert response.metadata["test_key"] == "test_value"
    assert isinstance(response.timestamp, datetime)
    assert response.message_type == MessageType.TEXT

@patch('aiohttp.ClientSession.post')
async def test_ollama_generation(mock_post):
    """Test Ollama response generation with mocked API calls"""
    # Mock the response for non-streaming request
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={
        "message": {"content": "4"}
    })
    mock_post.return_value.__aenter__.return_value = mock_resp
    
    resources = ResourceConfig(
        local_only=True,
        priority=1,
        max_parallel_calls=3
    )
    config = create_ollama_config(resources=resources)
    
    llm = OllamaLLM(config)
    
    # Test messages
    messages = [
        {
            "role": "user",
            "content": "What is 2+2? Answer in one word."
        }
    ]
    
    # Test non-streaming response
    response = await llm.generate(
        system_prompt="You are a helpful assistant. Be concise.",
        messages=messages,
        stream=False
    )
    
    assert response.content is not None
    assert isinstance(response.content, str)
    assert response.content == "4"
    assert response.confidence == 1.0
    assert "model" in response.metadata
    assert response.metadata["finish_reason"] == "stop"
    
    # Setup mock for streaming response
    mock_content = AsyncMock()
    mock_content.__aiter__.return_value = [
        b'{"message": {"content": "4"}}',
        b'{"message": {"content": " is correct"}}',
    ]
    mock_resp.content = mock_content
    
    # Test streaming response
    chunks = []
    async for chunk in await llm.generate(
        system_prompt="You are a helpful assistant. Be concise.",
        messages=messages,
        stream=True
    ):
        assert chunk.role == ParticipantRole.ASSISTANT
        assert chunk.content is not None
        chunks.append(chunk)
    
    assert len(chunks) == 2
    assert chunks[0].content == "4"
    assert chunks[1].content == " is correct"

@patch('aiohttp.ClientSession.post')
async def test_generate_with_config(mock_post):
    """Test generate_with_config method"""
    # Mock the streaming response
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_content = AsyncMock()
    mock_content.__aiter__.return_value = [
        b'{"message": {"content": "Test"}}',
        b'{"message": {"content": " response"}}',
    ]
    mock_resp.content = mock_content
    mock_post.return_value.__aenter__.return_value = mock_resp
    
    # Create original config and custom config
    original_config = create_ollama_config(model_type=OllamaModelType.GENERAL)
    custom_config = create_ollama_config(model_type=OllamaModelType.CODE)
    
    llm = OllamaLLM(original_config)
    
    # Test with custom config
    messages = [{"role": "user", "content": "Test"}]
    chunks = []
    
    # generate_with_config returns an async generator, so we iterate directly
    async for chunk in llm.generate_with_config(
        config=custom_config,
        messages=messages
    ):
        chunks.append(chunk)
    
    assert len(chunks) == 2
    assert chunks[0].content == "Test"
    assert chunks[1].content == " response"
    
    # Verify the config was temporarily changed and then restored
    assert llm.config == original_config
    assert llm.config.model_type == OllamaModelType.GENERAL

@patch('aiohttp.ClientSession.post')
async def test_ollama_error_handling(mock_post):
    """Test Ollama error handling with non-existent model"""
    # Mock an API error response
    mock_resp = AsyncMock()
    mock_resp.status = 404
    mock_resp.text = AsyncMock(return_value="Model not found")
    mock_post.return_value.__aenter__.return_value = mock_resp
    
    config = create_ollama_config(
        model_type=OllamaModelType.NONEXISTENT,
        timeout=5.0  # Short timeout for faster test
    )
    
    llm = OllamaLLM(config)
    
    with pytest.raises(Exception) as exc_info:
        await llm.generate(
            system_prompt="Test",
            messages=[{"role": "user", "content": "Test"}],
            stream=False
        )
    assert "Ollama API error" in str(exc_info.value)

async def test_invalid_config_type():
    """Test error handling when an invalid config type is provided"""
    # Create an OllamaLLM with valid config
    valid_config = create_ollama_config()
    llm = OllamaLLM(valid_config)
    
    # Try to use generate_with_config with an invalid config type
    invalid_config = BaseLlmConfig()  # Not an OllamaConfig
    
    with pytest.raises(TypeError) as exc_info:
        # We need to try to iterate over the generator to trigger the error
        async for _ in llm.generate_with_config(
            config=invalid_config,
            messages=[{"role": "user", "content": "Test"}]
        ):
            pass  # This should raise before we get here
    
    assert "Expected OllamaConfig" in str(exc_info.value)

if __name__ == "__main__":
    asyncio.run(test_ollama_generation())
