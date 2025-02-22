import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from MAX.agents import RecursiveThinkerAgent, RecursiveThinkerOptions
from MAX.types import (
    ConversationMessage,
    ParticipantRole,
    MemoryEntry,
    ConversationMemory,
    DataCategory,
    DataPriority,
    MessageType,
    ResourceConfig,
    MessageContent
)
from MAX.managers import MemorySystem
from MAX.storage import ChatStorage, ChromaDBChatStorage
from MAX.llms.anthropic import AnthropicLLM, LLMProviderError
from MAX.llms.utils.exceptions import LLMConfigError
from tests.mocks.storage import MockChatStorage
from typing import AsyncGenerator, List, Optional, Dict, Any, Union
from MAX.config.llms.base import BaseLlmConfig
from MAX.llms.base import AsyncLLMBase

class MockLLM(AsyncLLMBase):
    """Mock LLM that matches the AsyncLLMBase interface"""
    def __init__(self, config: BaseLlmConfig):
        super().__init__(config)
        self.streaming = False
        
    async def _generate(self, messages: List[Dict[str, str]]) -> str:
        return "Test response"

    async def generate(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        stream: bool = False
    ) -> Union[str, AsyncGenerator[Dict[str, str], None]]:
        if stream:
            async def stream_tokens():
                for token in ["Test", " ", "response"]:
                    yield {"text": token}
            return stream_tokens()
        return "Test response"

@pytest.fixture
async def mock_llm():
    mock = AsyncMock()
    mock.generate = AsyncMock(return_value="Test response")
    return mock

@pytest.fixture
def mock_memory_system():
    memory_system = AsyncMock(spec=MemorySystem)
    memory_system.query = AsyncMock(return_value=[
        {"content": "Previous memory about blue being favorite color"},
        {"content": "Memory about discussing French cities"}
    ])
    memory_system.store = AsyncMock(return_value="memory_id_123")
    return memory_system

@pytest.fixture
def mock_response() -> ConversationMessage:
    return ConversationMessage(
        role=ParticipantRole.ASSISTANT,
        content=[{"text": "Test response"}]
    )

class TypedMockStorage:
    """Type-safe mock storage for testing"""
    async def save_chat(
        self,
        session_id: str,
        message: ConversationMessage,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        pass

    async def get_chat_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[ConversationMessage]:
        return []

@pytest.fixture
def agent_options():
    return RecursiveThinkerOptions(
        name="RecursiveThinker",
        description="A recursive thinking agent",
        model_id="claude-3-haiku-20240307",
        streaming=False
    )

@pytest.fixture
async def agent(agent_options):
    """Create agent with proper LLM setup"""
    agent = RecursiveThinkerAgent(agent_options)
    # Initialize the LLM properly
    agent._llm = MockLLM(BaseLlmConfig())
    # Initialize the memory dict
    agent.conversation_memories = {}
    return agent

@pytest.fixture
def chat_history():
    """Fixture for sample chat history"""
    return [
        ConversationMessage(
            role=ParticipantRole.USER.value,
            content=[{"text": "What is quantum computing?"}],
            metadata={"timestamp": "2024-01-01T10:00:00Z"}
        ),
        ConversationMessage(
            role=ParticipantRole.ASSISTANT.value,
            content=[{"text": "Quantum computing uses quantum mechanics principles..."}],
            metadata={"timestamp": "2024-01-01T10:00:01Z"}
        )
    ]

@pytest.mark.asyncio
async def test_basic_request(agent_options):
    """Test basic request processing"""
    agent = RecursiveThinkerAgent(agent_options)
    response = await agent.process_request(
        input_text="Hello",
        user_id="test_user",
        session_id="test_session",
        chat_history=[]
    )
    assert isinstance(response, ConversationMessage)
    assert response.content[0]["text"]

@pytest.mark.asyncio
async def test_memory_storage(agent_options):
    """Test basic memory storage"""
    agent = RecursiveThinkerAgent(agent_options)
    await agent.process_request(
        input_text="Remember this",
        user_id="test_user",
        session_id="test_session",
        chat_history=[]
    )
    assert "test_session" in agent.conversation_memories
    assert len(agent.conversation_memories["test_session"]) > 0

@pytest.mark.asyncio
async def test_context_usage(agent_options):
    """Test conversation context is used"""
    agent = RecursiveThinkerAgent(agent_options)
    await agent.process_request(
        input_text="My name is Alice",
        user_id="test_user",
        session_id="test_session",
        chat_history=[]
    )
    response = await agent.process_request(
        input_text="What's my name?",
        user_id="test_user",
        session_id="test_session",
        chat_history=[]
    )
    assert "Alice" in response.content[0]["text"]

@pytest.mark.asyncio
async def test_basic_request_processing(agent):
    response = await agent.process_request(
        input_text="Test input",
        user_id="test_user",
        session_id="test_session",
        chat_history=[]
    )
    assert isinstance(response, ConversationMessage)
    assert response.role == ParticipantRole.ASSISTANT

@pytest.mark.asyncio
async def test_streaming_response(agent):
    agent.streaming = True
    response = await agent.process_request(
        input_text="Test streaming",
        user_id="test_user",
        session_id="test_session",
        chat_history=[]
    )
    assert response is not None

@pytest.mark.asyncio
async def test_multi_step_reasoning(agent_options):
    """Test that agent can use previous context for better responses"""
    agent = RecursiveThinkerAgent(agent_options)
    
    # First interaction - establish context
    response1 = await agent.process_request(
        input_text="My favorite color is blue and I live in Paris",
        user_id="test_user",
        session_id="test_session",
        chat_history=[]
    )
    
    # Second interaction - should use color context
    response2 = await agent.process_request(
        input_text="What's my favorite color?",
        user_id="test_user",
        session_id="test_session",
        chat_history=[response1]
    )
    assert "blue" in response2.content[0]["text"].lower()
    
    # Third interaction - should use location context
    response3 = await agent.process_request(
        input_text="Which city do I live in?",
        user_id="test_user",
        session_id="test_session",
        chat_history=[response1, response2]
    )
    assert "paris" in response3.content[0]["text"].lower()
    
    # Fourth interaction - should combine both pieces of context
    response4 = await agent.process_request(
        input_text="Tell me about my favorite color and city",
        user_id="test_user",
        session_id="test_session",
        chat_history=[response1, response2, response3]
    )
    assert "blue" in response4.content[0]["text"].lower()
    assert "paris" in response4.content[0]["text"].lower()

@pytest.mark.asyncio
async def test_memory_integration(
    agent: RecursiveThinkerAgent,
    mock_response: ConversationMessage
) -> None:
    """Test memory integration with proper typing"""
    response = await agent.process_request(
        input_text="Remember this: The sky is blue",
        user_id="test_user",
        session_id="test_session",
        chat_history=[]
    )
    assert isinstance(response, ConversationMessage)
    assert response.role == ParticipantRole.ASSISTANT

@pytest.mark.asyncio
async def test_error_handling(agent):
    with pytest.raises(Exception):
        await agent.process_request(
            input_text="",  # Empty input should raise error
            user_id="test_user",
            session_id="test_session",
            chat_history=[]
        )

@pytest.mark.asyncio
async def test_system_prompt_structure(agent, mock_llm):
    """Test that the system prompt contains all required recursive thinking components"""
    await agent.process_request(
        input_text="Test question",
        user_id="test_user",
        session_id="test_session",
        chat_history=[]
    )
    
    call_args = mock_llm.generate.call_args[1]
    system_prompt = call_args['system_prompt']
    
    # Check for key components in system prompt
    assert "Breaking them down into clear subcomponents" in system_prompt
    assert "Analyzing each component from multiple angles" in system_prompt
    assert "Considering potential implications and trade-offs" in system_prompt
    assert "Synthesizing insights into a cohesive solution" in system_prompt
    assert "Recursively improving the solution" in system_prompt

@pytest.mark.asyncio
async def test_llm_error_handling(agent, mock_llm):
    """Test handling of LLM provider errors"""
    # Make LLM raise an error
    mock_llm.generate.side_effect = LLMProviderError("Test error", "anthropic")
    
    result = await agent.process_request(
        input_text="test request",
        user_id="test_user",
        session_id="test_session",
        chat_history=[]
    )
    
    assert isinstance(result, ConversationMessage)
    assert "Error processing request" in result.content[0]["text"]
    assert "Test error" in result.content[0]["text"]

@pytest.mark.asyncio
async def test_config_error_handling(agent, mock_llm):
    """Test handling of LLM configuration errors"""
    # Make LLM raise a config error
    mock_llm.generate.side_effect = LLMConfigError("Missing API key")
    
    result = await agent.process_request(
        input_text="test request",
        user_id="test_user",
        session_id="test_session",
        chat_history=[]
    )
    
    assert isinstance(result, ConversationMessage)
    assert "Error processing request" in result.content[0]["text"]
    assert "Missing API key" in result.content[0]["text"]

@pytest.mark.asyncio
async def test_confidence_threshold(agent, mock_llm):
    """Test that agent respects confidence threshold"""
    # Mock low confidence response
    mock_llm.generate = AsyncMock(return_value={
        "content": "I'm not very confident about this...",
        "confidence": 0.5
    })
    
    result = await agent.process_request(
        input_text="Very complex question",
        user_id="test_user",
        session_id="test_session",
        chat_history=[]
    )
    
    assert "I need more information" in result.content[0]["text"]
    assert result.metadata.get("confidence", 1.0) < agent.options.min_confidence_threshold

@pytest.mark.asyncio
async def test_recursion_depth_limit(agent, mock_llm):
    """Test that agent respects max recursion depth"""
    session_id = "test_recursion_session"
    
    # Force recursive thinking by making responses incomplete
    mock_llm.generate = AsyncMock(side_effect=[
        "Need more analysis...",
        "Still thinking...",
        "Almost there...",
        "Final answer"
    ])
    
    result = await agent.process_request(
        input_text="Complex recursive problem",
        user_id="test_user",
        session_id=session_id,
        chat_history=[]
    )
    
    # Verify we didn't exceed max recursion depth
    assert mock_llm.generate.call_count <= agent.options.max_recursion_depth
    assert "Final answer" in result.content[0]["text"]

@pytest.mark.asyncio
async def test_context_window_management(agent, mock_llm, chat_history):
    """Test that agent properly manages context window size"""
    long_history = chat_history * 10  # Create a very long history
    
    await agent.process_request(
        input_text="Test question with long history",
        user_id="test_user",
        session_id="test_session",
        chat_history=long_history
    )
    
    # Verify that the prompt sent to LLM isn't too long
    call_args = mock_llm.generate.call_args[1]
    system_prompt = call_args['system_prompt']
    assert len(system_prompt) < 8000  # Assuming reasonable token limit

@pytest.mark.asyncio
async def test_metadata_preservation(agent, mock_llm):
    """Test that agent preserves and updates metadata correctly"""
    result = await agent.process_request(
        input_text="Test metadata",
        user_id="test_user",
        session_id="test_session",
        chat_history=[],
    )
    
    assert result.metadata.get("source") == "unit_test"
    assert result.metadata.get("priority") == "high"
    assert "timestamp" in result.metadata
    assert "processing_time" in result.metadata

@pytest.mark.asyncio
async def test_store_memory(agent):
    """Test storing memory entries"""
    session_id = "test_session"
    content = "test content"
    role = ParticipantRole.USER
    
    await agent._store_memory(session_id, content, role)
    
    assert session_id in agent.conversation_memories
    memory = agent.conversation_memories[session_id]
    assert isinstance(memory, ConversationMemory)
    assert len(memory.messages) == 1
    
    entry = memory.messages[0]
    assert isinstance(entry, MemoryEntry)
    assert entry.content == content
    assert entry.role == role
    assert entry.category == DataCategory.AGENT

@pytest.mark.asyncio
async def test_get_conversation_context(agent):
    """Test retrieving conversation context"""
    session_id = "test_session"
    
    # Store some test memories
    await agent._store_memory(session_id, "User message 1", ParticipantRole.USER)
    await agent._store_memory(session_id, "Assistant message 1", ParticipantRole.ASSISTANT)
    await agent._store_memory(session_id, "User message 2", ParticipantRole.USER)
    
    context = await agent._get_conversation_context(session_id)
    
    assert isinstance(context, str)
    assert "User: User message" in context
    assert "Assistant: Assistant message" in context

@pytest.mark.asyncio
async def test_process_request(agent):
    """Test processing a user request"""
    input_text = "Test input"
    user_id = "test_user"
    session_id = "test_session"
    
    response = await agent.process_request(
        input_text=input_text,
        user_id=user_id,
        session_id=session_id,
        chat_history=[]
    )
    
    assert isinstance(response, ConversationMessage)
    assert response.role == ParticipantRole.ASSISTANT.value
    assert isinstance(response.content, list)
    assert response.content[0]["text"] == "Test response"
    
    # Verify memory was stored
    assert session_id in agent.conversation_memories
    memory = agent.conversation_memories[session_id]
    assert len(memory.messages) == 2  # User input and assistant response

@pytest.mark.asyncio
async def test_memory_context_in_prompt(agent, mock_llm):
    """Test that memory context is included in the prompt"""
    await agent.process_request(
        input_text="Tell me about France",
        user_id="test_user",
        session_id="test_session",
        chat_history=[]
    )
    
    # Verify memory context in prompt
    call_args = mock_llm.generate.call_args[1]
    system_prompt = call_args['system_prompt']
    assert "Relevant context from memory" in system_prompt
    assert "French cities" in system_prompt

@pytest.mark.asyncio
async def test_streaming_with_memory(agent, mock_llm):
    """Test streaming responses with memory integration"""
    agent.streaming = True
    
    async def mock_stream():
        yield "Part 1"
        yield "Part 2"
    
    mock_llm.generate = AsyncMock(return_value=mock_stream())
    
    response = await agent.process_request(
        input_text="Complex question",
        user_id="test_user",
        session_id="test_session",
        chat_history=[]
    )
    
    parts = []
    async for part in response:
        parts.append(part)
    
    assert parts == ["Part 1", "Part 2"]
    # Verify memory was still stored after streaming
    agent.memory_system.store.assert_called_once()

@pytest.mark.asyncio
async def test_basic_memory(agent):
    """Test basic memory functionality"""
    response1 = await agent.process_request(
        input_text="My name is Alice",
        user_id="test_user",
        session_id="test_session",
        chat_history=[]
    )
    assert isinstance(response1, ConversationMessage)
    assert "Test response" in response1.content[0]["text"]
    
    # Verify memory was stored
    assert "test_session" in agent.conversation_memories

@pytest.mark.asyncio
async def test_conversation_context(agent):
    # First message
    await agent.process_request(
        input_text="My name is Alice",
        user_id="test_user",
        session_id="test_session",
        chat_history=[]
    )
    
    # Second message should have context
    response = await agent.process_request(
        input_text="What's my name?",
        user_id="test_user",
        session_id="test_session",
        chat_history=[]
    )
    assert isinstance(response, ConversationMessage)
    assert len(agent.conversation_memories["test_session"]) == 4  # Two pairs of messages

@pytest.mark.asyncio
async def test_error_handling(agent):
    # Force an error by making LLM generate fail
    agent._llm._generate = AsyncMock(side_effect=Exception("Test error"))
    
    response = await agent.process_request(
        input_text="This should fail",
        user_id="test_user",
        session_id="test_session",
        chat_history=[]
    )
    assert isinstance(response, ConversationMessage)
    assert "Error" in response.content[0]["text"]
