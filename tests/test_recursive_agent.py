import pytest
import aiohttp
import asyncio
from datetime import datetime
from MAX.types import TimestampedMessage
from MAX.agents import RecursiveThinkerAgent, RecursiveThinkerOptions
from MAX.types import ConversationMessage, ParticipantRole
from MAX.storage import InMemoryChatStorage
from typing import Dict, List, Optional

async def is_ollama_available():
    """Check if Ollama server is running and has models loaded"""
    try:
        timeout = aiohttp.ClientTimeout(total=5)  # Set overall timeout to 5 seconds
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Check server is up
            async with session.get("http://localhost:11434/api/tags") as resp:
                if resp.status != 200:
                    print("\nOllama server not responding")
                    return False
                tags = await resp.json()
                print(f"\nAvailable models: {tags}")
                
            # Check model can generate with timeout
            test_payload = {
                "model": "llama3.1:8b-instruct-q8_0",
                "prompt": "Say hi",
                "stream": False
            }
            async with session.post("http://localhost:11434/api/generate", json=test_payload) as resp:
                if resp.status != 200:
                    print("\nOllama cannot generate responses")
                    return False
                await resp.json()  # Actually wait for response
                print("\nOllama generation test successful")
                return True
                
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        print(f"\nError checking Ollama: {str(e)}")
        return False

class TestChatStorage(InMemoryChatStorage):
    """Simple storage implementation for testing"""
    
    def __init__(self):
        super().__init__()
        self.state: Dict = {}
        self.conversations: Dict[str, List[TimestampedMessage]] = {}
        
    async def initialize(self) -> bool:
        return True
        
    async def cleanup(self) -> bool:
        self.conversations.clear()
        return True
        
    async def check_health(self) -> bool:
        return True
        
    async def get_system_state(self) -> Dict:
        return self.state
        
    async def save_system_state(self, state: Dict) -> bool:
        self.state = state
        return True
        
    async def save_task_state(self, task_id: str, state: Dict) -> bool:
        self.state[task_id] = state
        return True

    async def save_chat_message(
        self,
        user_id: str,
        session_id: str,
        agent_id: str,
        new_message: ConversationMessage,
        max_history_size: Optional[int] = None,
    ) -> bool:
        key = f"{user_id}#{session_id}#{agent_id}"
        if key not in self.conversations:
            self.conversations[key] = []
            
        # Create TimestampedMessage properly
        if isinstance(new_message.content, list) and new_message.content:
            text = next((item.get("text", "") for item in new_message.content if isinstance(item, dict) and "text" in item), "")
        else:
            text = str(new_message.content) if new_message.content else ""
                
        timestamp = (new_message.metadata.get('timestamp') 
                    if new_message.metadata 
                    else datetime.now().timestamp())
            
        # Create ConversationMessage first
        message = ConversationMessage(
            role=ParticipantRole.USER,
            content=[{"text": text}],
            message_type="text"
        )
        
        # Create TimestampedMessage with proper structure
        timestamped_message = TimestampedMessage(
            message=message,
            timestamp=datetime.fromtimestamp(timestamp) if isinstance(timestamp, float) else timestamp
        )
        
        self.conversations[key].append(timestamped_message)
        
        if max_history_size and len(self.conversations[key]) > max_history_size:
            self.conversations[key] = self.conversations[key][-max_history_size:]
        
        return True
        
    async def fetch_chat(
        self,
        user_id: str,
        session_id: str,
        agent_id: str,
        max_history_size: Optional[int] = None,
    ) -> List[ConversationMessage]:
        key = f"{user_id}#{session_id}#{agent_id}"
        messages = self.conversations.get(key, [])
        
        if max_history_size:
            messages = messages[-max_history_size:]
            
        # Convert TimestampedMessage to ConversationMessage
        converted_messages = []
        for msg in messages:
            # Extract text from the message structure
            if isinstance(msg, TimestampedMessage):
                # Get the message content from the wrapped ConversationMessage
                text = msg.message.content[0]["text"] if msg.message.content else ""
                converted_messages.append(
                    ConversationMessage(
                        role=msg.message.role,
                        content=[{"text": text}],
                        message_type="text",
                        metadata={"timestamp": msg.timestamp}
                    )
                )
        return converted_messages
        
    async def search_similar_messages(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 5
    ) -> List[ConversationMessage]:
        return []

@pytest.fixture
def storage():
    return TestChatStorage()

@pytest.fixture
def agent_options(storage):
    return RecursiveThinkerOptions(
        name="test_agent",
        description="Test agent",
        model_type="llama3.1:8b-instruct-q8_0",  # Use model we know exists
        storage=storage,
        streaming=False
    )

@pytest.fixture
async def agent(agent_options):
    agent = RecursiveThinkerAgent(agent_options)
    await agent.initialize()
    return agent  # Simple return since no cleanup needed


@pytest.mark.asyncio
async def test_basic_chat(agent):
    """Test that agent can understand and respond to messages"""
    print("\nTesting basic chat functionality...")
    
    try:
        # Test greeting and factual question
        response = await agent.process_request(
            input_text="Hi! Can you tell me what's 2+2?",
            user_id="test_user",
            session_id="test_session",
            chat_history=[]
        )
        
        print(f"\nAgent response: {response.content[0]['text']}\n")
        
        # First check if we got an error response
        if response.metadata.get('error'):
            pytest.fail(f"Agent returned error: {response.content[0]['text']}")
            
        assert isinstance(response, ConversationMessage)
        assert response.role == ParticipantRole.ASSISTANT
        assert "4" in response.content[0]["text"]
        
    except Exception as e:
        pytest.fail(f"Test failed with exception: {str(e)}")

@pytest.mark.asyncio
async def test_memory(agent, storage):
    """Test memory system functionality"""
    # Test storage functionality without LLM
    test_messages = []
    for i in range(3):
        msg = ConversationMessage(
            role=ParticipantRole.USER,
            content=[{"text": f"Test message {i}"}],
            message_type="text",
            metadata={
                "timestamp": datetime.now().timestamp(),
                "user_id": "test_user",
                "session_id": "test_session"
            }
        )
        await storage.save_chat_message(
            user_id="test_user",
            session_id="test_session",
            agent_id=agent.id,
            new_message=msg
        )
        test_messages.append(msg)
    
    # Verify storage retrieval
    stored_messages = await storage.fetch_chat(
        user_id="test_user",
        session_id="test_session",
        agent_id=agent.id
    )
    assert len(stored_messages) >= 3
    
    # Verify message structure
    for msg in stored_messages:
        assert isinstance(msg, ConversationMessage)
        assert "timestamp" in msg.metadata

if __name__ == "__main__":
    pytest.main([__file__])
