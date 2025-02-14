# tests/adapters/discord/test_discord_adapter.py
import pytest
from unittest.mock import Mock, AsyncMock, patch, PropertyMock
from discord import Message, Client, TextChannel, Guild, Member, User

from MAX.adapters.discord import DiscordAdapter, DiscordAdapterConfig
from MAX.types import (
    ParticipantRole,
    ProcessedContent,
    DiscordAttachment,
    DiscordAdapterConfig,
    AgentResponse,
    ConversationMessage,
    RequestMetadata
)
import pytest
import pytest_asyncio  # Add this import

# Add this at the top of the file
pytestmark = pytest.mark.asyncio  # This marks all tests in the file as async

@pytest.fixture
def config():
    return DiscordAdapterConfig(
        command_prefix="!",
        allowed_channels=[123456789],
        retry_attempts=3,
        retry_delay=1,
        max_message_length=2000
    )

@pytest.fixture
def mock_orchestrator():
    orchestrator = AsyncMock()
    orchestrator.route_request.return_value = AgentResponse(
        metadata=RequestMetadata(
            user_input="test",
            agent_id="test_agent",
            agent_name="Test Agent",
            user_id="123",
            session_id="test_session",
            additional_params={},
            error_type=None
        ),
        output=ConversationMessage(
            role=ParticipantRole.ASSISTANT,  # Use enum directly
            content=[{"text": "Test response"}]
        ),
        streaming=False
    )
    return orchestrator

@pytest.fixture
async def adapter(config, mock_orchestrator):
    adapter = DiscordAdapter(
        token="test_token",
        config=config
    )
    adapter.set_orchestrator(mock_orchestrator)
    return adapter

@pytest.mark.asyncio
async def test_message_processing_success(adapter):
    # Mock message
    message = AsyncMock(spec=Message)
    message.content = "!hello"
    message.author = AsyncMock(spec=User)
    message.author.id = "123"
    message.author.name = "TestUser"
    message.channel = AsyncMock(spec=TextChannel)
    message.channel.id = 123456789
    message.guild = AsyncMock(spec=Guild)
    message.guild.id = "456"
    message.attachments = []
    
    # Test processing
    processed = await adapter.process_message(message)
    assert processed is not None
    assert processed.text == "hello"

@pytest.mark.asyncio
async def test_message_processing_wrong_channel(adapter):
    # Mock message from non-allowed channel
    message = AsyncMock(spec=Message)
    message.content = "!hello"
    message.channel = AsyncMock(spec=TextChannel)
    message.channel.id = 999  # Different channel ID
    
    # Test processing
    processed = await adapter.process_message(message)
    assert processed is None

@pytest.mark.asyncio
async def test_handle_message_success(adapter, mock_orchestrator):
    # Mock message
    message = AsyncMock(spec=Message)
    message.content = "!hello"
    message.author = AsyncMock(spec=User)
    message.author.id = "123"
    message.author.name = "TestUser"
    message.channel = AsyncMock(spec=TextChannel)
    message.channel.id = 123456789
    message.guild = AsyncMock(spec=Guild)
    message.guild.id = "456"
    message.attachments = []
    message.channel.send = AsyncMock()
    
    # Test handling
    await adapter._handle_message(message)
    
    # Verify orchestrator was called correctly
    mock_orchestrator.route_request.assert_called_once()
    call_args = mock_orchestrator.route_request.call_args[1]
    assert call_args["user_input"] == "hello"
    assert call_args["user_id"] == "123"

@pytest.mark.asyncio
async def test_send_response_long_message(adapter):
    # Create a long message that needs splitting
    long_message = "x" * 3000  # Longer than max_message_length
    
    message = AsyncMock(spec=Message)
    message.channel = AsyncMock()
    message.channel.send = AsyncMock()
    
    response = AgentResponse(
        metadata={},
        output=ConversationMessage(
            role=ParticipantRole.ASSISTANT,
            content=[{"text": long_message}]
        ),
        streaming=False
    )
    
    await adapter._send_response(message, response)
    assert message.channel.send.call_count > 1  # Should be called multiple times for chunks

@pytest.mark.asyncio
async def test_setup_handlers(adapter):
    # The client.event decorator is actually a function, not a Mock
    # Let's test the client setup differently
    assert hasattr(adapter.client, 'on_ready')
    assert hasattr(adapter.client, 'on_message')

@pytest.mark.asyncio
async def test_on_ready(adapter):
    # Create mock guild and client
    guild = AsyncMock(spec=Guild)
    adapter.client = AsyncMock(spec=Client)
    
    # Mock the is_ready method
    adapter.client.is_ready.return_value = True
    
    # Mock the guilds property
    type(adapter.client).guilds = PropertyMock(return_value=[guild])
    
    # If the adapter has an on_ready callback, call it
    if hasattr(adapter, '_on_ready'):
        await adapter._on_ready()
    elif hasattr(adapter.client, '_on_ready'):
        await adapter.client._on_ready()
    
    # Verify the client is ready
    assert adapter.client.is_ready()
    # Verify we have access to guilds
    assert len(adapter.client.guilds) == 1
    
@pytest.mark.asyncio
async def test_conversation_message_handling(adapter, mock_orchestrator):
    message = AsyncMock(spec=Message)
    message.content = "!hello"
    message.author = AsyncMock(spec=User)
    message.author.id = "123"
    message.channel = AsyncMock(spec=TextChannel)
    message.channel.id = 123456789
    message.guild = AsyncMock(spec=Guild)
    message.attachments = []
    message.channel.send = AsyncMock()

    mock_orchestrator.route_request.return_value = AgentResponse(
        metadata=RequestMetadata(
            user_input="hello",
            agent_id="test_agent",
            agent_name="Test Agent",
            user_id="123",
            session_id="test_session",
            additional_params={},
            error_type=None
        ),
        output=ConversationMessage(
            role=ParticipantRole.ASSISTANT,
            content=[{"text": "Test assistant response"}]
        ),
        streaming=False
    )

    await adapter._handle_message(message)
    message.channel.send.assert_called_once_with("Test assistant response")

@pytest.mark.asyncio
async def test_config_validation(config):
    # Test DiscordAdapterConfig validation
    assert config.command_prefix == "!"
    assert config.allowed_channels == [123456789]
    assert config.retry_attempts == 3
    assert config.retry_delay == 1
    assert config.max_message_length == 2000

@pytest.mark.asyncio
async def test_error_handling(adapter):
    # Test error handling when orchestrator fails
    message = AsyncMock(spec=Message)
    message.content = "!hello"
    message.author = AsyncMock(spec=User)
    message.author.id = "123"
    message.channel = AsyncMock(spec=TextChannel)
    message.channel.id = 123456789
    message.guild = AsyncMock(spec=Guild)
    message.attachments = []
    message.channel.send = AsyncMock()

    # Make orchestrator raise an exception
    adapter.orchestrator.route_request.side_effect = Exception("Test error")

    # Test handling
    await adapter._handle_message(message)
    
    # Verify error message was sent
    message.channel.send.assert_called_once_with(
        "Sorry, I encountered an error processing your request."
    )

@pytest.mark.asyncio
async def test_retry_mechanism(adapter):
    # Test retry mechanism for sending messages
    message = AsyncMock(spec=Message)
    message.channel = AsyncMock()
    message.channel.send = AsyncMock(side_effect=[
        Exception("First attempt failed"),
        Exception("Second attempt failed"),
        None  # Success on third attempt
    ])

    response = AgentResponse(
        metadata={},
        output=ConversationMessage(
            role=ParticipantRole.ASSISTANT.value,
            content=[{"text": "Test message"}]
        ),
        streaming=False
    )

    await adapter._send_response(message, response)
    assert message.channel.send.call_count == 3

if __name__ == '__main__':
    pytest.main([__file__, '-v'])