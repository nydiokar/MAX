import pytest
from MAX.agents.default_agent import DefaultAgent
from MAX.types import ConversationMessage

@pytest.mark.asyncio
async def test_default_agent_basic_response():
    # Initialize the agent
    agent = DefaultAgent()
    
    # Test a simple query
    response = await agent.process_request(
        input_text="Hello! Can you help me with a general question?",
        user_id="test_user",
        session_id="test_session",
        chat_history=None
    )
    
    # Print the actual response
    print("\n=== Basic Response Test ===")
    print(f"Input: Hello! Can you help me with a general question?")
    print(f"Output: {response.content[0]['text']}")
    print("========================\n")
    
    # Verify response
    assert isinstance(response, ConversationMessage)
    assert response.role == "assistant"
    assert len(response.content) > 0
    assert isinstance(response.content[0]["text"], str)
    assert response.metadata["user_id"] == "test_user"
    assert response.metadata["session_id"] == "test_session"
    assert "timestamp" in response.metadata

@pytest.mark.asyncio
async def test_default_agent_with_context():
    # Initialize the agent
    agent = DefaultAgent()
    
    # Create some chat history
    chat_history = [
        ConversationMessage(role="user", content=[{"text": "What's the weather like?"}]),
        ConversationMessage(role="assistant", content=[{"text": "I don't have access to real-time weather data."}]),
        ConversationMessage(role="user", content=[{"text": "Can you help me with something else then?"}])
    ]
    
    # Test with context
    response = await agent.process_request(
        input_text="Yes, I need help with Python programming",
        user_id="test_user",
        session_id="test_session",
        chat_history=chat_history
    )
    
    # Print the actual response
    print("\n=== Context Response Test ===")
    print(f"Input: Yes, I need help with Python programming")
    print(f"Output: {response.content[0]['text']}")
    print("==========================\n")
    
    # Verify response
    assert isinstance(response, ConversationMessage)
    assert response.role == "assistant"
    assert len(response.content) > 0
    assert isinstance(response.content[0]["text"], str)
    assert response.metadata["user_id"] == "test_user"
    assert response.metadata["session_id"] == "test_session"
    assert "timestamp" in response.metadata

@pytest.mark.asyncio
async def test_default_agent_complex_query():
    # Initialize the agent
    agent = DefaultAgent()
    
    # Test a more complex query
    response = await agent.process_request(
        input_text="Can you explain how to implement a binary search tree in Python?",
        user_id="test_user",
        session_id="test_session",
        chat_history=None
    )
    
    # Print the actual response
    print("\n=== Complex Query Test ===")
    print(f"Input: Can you explain how to implement a binary search tree in Python?")
    print(f"Output: {response.content[0]['text']}")
    print("========================\n")
    
    # Verify response
    assert isinstance(response, ConversationMessage)
    assert response.role == "assistant"
    assert len(response.content) > 0
    assert isinstance(response.content[0]["text"], str)
    assert len(response.content[0]["text"]) > 50  # Should be a substantial response