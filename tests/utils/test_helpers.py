from typing import Any, List, Dict, Union, Optional
from datetime import datetime
from MAX.types import (
    ConversationMessage,
    TimestampedMessage,
    ParticipantRole,
    MemoryEntry,
    ConversationMemory,
    AgentProviderType  # Now imported from types/__init__.py
)
from unittest.mock import AsyncMock, Mock
from abc import ABC, abstractmethod
from pydantic import BaseModel
from MAX.agents import RecursiveThinkerAgent, RecursiveThinkerOptions
from MAX.agents.teams.team_registry import TeamRegistry

# Import the functions to be tested
from MAX.utils import is_tool_input, conversation_to_dict

def test_is_tool_input():
    # Test valid tool input
    valid_input = {"selected_agent": "agent1", "confidence": 0.8}
    assert is_tool_input(valid_input) == True

    # Test invalid inputs
    invalid_inputs = [
        {"selected_agent": "agent1"},  # Missing 'confidence'
        {"confidence": 0.8},  # Missing 'selected_agent'
        "not a dict",  # Not a dictionary
        {},  # Empty dictionary
        {"key1": "value1", "key2": "value2"}  # Dictionary without required keys
    ]
    for invalid_input in invalid_inputs:
        assert is_tool_input(invalid_input) == False

def test_conversation_to_dict():
    # Test with a single ConversationMessage
    conv_msg = ConversationMessage(role=ParticipantRole.USER.value, content="Hello")
    result = conversation_to_dict(conv_msg)
    assert result == {"role": "user", "content": "Hello"}

    # Test with a single TimestampedMessage
    timestamp = datetime.now()
    timestamped_msg = TimestampedMessage(role=ParticipantRole.ASSISTANT.value, content="Hi there", timestamp=timestamp)
    result = conversation_to_dict(timestamped_msg)
    assert result == {"role": "assistant", "content": "Hi there", "timestamp": timestamp}

    # Test with a list of messages
    messages = [
        ConversationMessage(role=ParticipantRole.USER.value, content="How are you?"),
        TimestampedMessage(role=ParticipantRole.ASSISTANT.value, content="I'm fine, thanks!", timestamp=timestamp)
    ]
    result = conversation_to_dict(messages)
    assert result == [
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm fine, thanks!", "timestamp": timestamp}
    ]

# First, let's see what ChatStorage actually requires
class ChatStorage(ABC):
    @abstractmethod
    async def fetch_chat_with_timestamps(self, user_id: str, session_id: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def save_chat_message(self, user_id: str, session_id: str, message: Dict[str, Any]) -> bool:
        pass

# Now let's implement our mock properly
class MockChatStorage:
    """Mock storage for testing"""
    def __init__(self) -> None:
        self.messages: Dict[tuple, list] = {}
        self.timestamped_messages: Dict[tuple, list] = {}

    async def fetch_chat_with_timestamps(self, user_id: str, session_id: str) -> List[Dict[str, Any]]:
        return self.timestamped_messages.get((user_id, session_id), [])

    async def save_chat_message(self, user_id: str, session_id: str, message: Dict[str, Any]) -> bool:
        key = (user_id, session_id)
        if key not in self.timestamped_messages:
            self.timestamped_messages[key] = []
        if 'timestamp' not in message:
            message['timestamp'] = datetime.utcnow()
        self.timestamped_messages[key].append(message)
        return True

    async def fetch_chat(self, user_id: str, session_id: str) -> List[Dict[str, Any]]:
        return self.messages.get((user_id, session_id), [])

class MockBaseAgent:
    """Base mock agent for testing"""
    def __init__(self, name: str = "MOCK_AGENT", description: str = "Mock agent for testing"):
        self.name = name
        self.description = description
        self.provider_type = AgentProviderType.CUSTOM  # Updated to use enum
        self.capabilities = ["text_generation"]
        self.tool_config = None
        self.streaming = False

    async def generate_response(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {
            "content": f"Mock response from {self.name}", 
            "confidence": 0.9,
            "metadata": {"timestamp": datetime.utcnow()}
        }

    async def process_request(self, *args: Any, **kwargs: Any) -> ConversationMessage:
        return ConversationMessage(
            role=ParticipantRole.ASSISTANT.value,
            content=[{"text": f"Mock response from {self.name}"}],
            metadata={"timestamp": datetime.utcnow()}
        )

def create_mock_team_registry() -> TeamRegistry:
    """Create a mock team registry for testing"""
    registry = TeamRegistry()
    mock_agent = MockBaseAgent()
    registry.register_agent("mock_agent", mock_agent)
    return registry

def create_recursive_thinker_agent(
    storage: Optional[Any] = None,
    **kwargs: Any
) -> RecursiveThinkerAgent:
    """Create a pre-configured recursive thinker agent for testing"""
    options = RecursiveThinkerOptions(
        name="Recursive Thinker",
        description="A test recursive thinker agent",
        max_recursion_depth=3,
        min_confidence_threshold=0.7,
        streaming=False,
        **kwargs
    )
    
    agent = RecursiveThinkerAgent(options)
    if storage:
        agent.storage = storage
    return agent

def is_tool_input(data: Any) -> bool:
    """Test if the input matches tool input format"""
    if not isinstance(data, dict):
        return False
    return all(key in data for key in ['selected_agent', 'confidence'])

def conversation_to_dict(
    messages: Union[ConversationMessage, TimestampedMessage, List[Union[ConversationMessage, TimestampedMessage]]]
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Convert conversation messages to dictionary format"""
    if isinstance(messages, list):
        return [conversation_to_dict(msg) for msg in messages]
    
    result = {
        "role": messages.role,
        "content": messages.content
    }
    
    if isinstance(messages, TimestampedMessage):
        result["timestamp"] = messages.timestamp
        
    return result

class MockLLM:
    """Mock LLM for testing"""
    async def generate(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        stream: bool = False
    ) -> str:
        return "Test response"

def create_test_agent() -> RecursiveThinkerAgent:
    """Create a test agent with mock LLM"""
    options = RecursiveThinkerOptions(
        name="TestAgent",
        description="Test agent",
        model_id="test-model",
        streaming=False
    )
    agent = RecursiveThinkerAgent(options)
    agent._get_llm = AsyncMock(return_value=MockLLM())
    return agent

def create_test_memory(
    session_id: str,
    messages: Optional[List[str]] = None
) -> ConversationMemory:
    """Create test conversation memory"""
    if messages is None:
        messages = ["Test message 1", "Test message 2"]
    
    memory_entries = [
        MemoryEntry(
            content=msg,
            role=ParticipantRole.USER,
            timestamp=datetime.now(),
            category="user"
        )
        for msg in messages
    ]
    
    return ConversationMemory(
        conversation_id=session_id,
        messages=memory_entries,
        metadata={},
        created_at=datetime.now()
    )
