from typing import Optional, Dict, Any, List
from discord import Client, Message, Intents
import asyncio

from ...types import ConversationMessage, PlatformAdapter, AgentResponse
from MAX.utils.logger import Logger
from MAX.types import DiscordAdapterConfig, ProcessedContent
from MAX.adapters.discord.attachment_manager import AttachmentManager
from MAX.adapters.discord.channel_manager import ChannelManager


class DiscordAdapter(PlatformAdapter):
    def __init__(
        self,
        token: str,
        config: DiscordAdapterConfig,
        logger: Optional[Logger] = None,
    ):

        self.token = token
        self.config = config
        self.logger = logger or Logger()

        # Initialize Discord client with intents
        intents = Intents.default()
        intents.message_content = True
        intents.guild_messages = True
        intents.dm_messages = True
        self.client = Client(intents=intents)

        # Initialize managers as protected members
        self._attachment_manager = AttachmentManager(self)
        self._channel_manager = ChannelManager(config, self.logger)

        # Reference to orchestrator (set later)
        self.orchestrator = None

        # Setup handlers
        self._setup_handlers()

    async def process_message(self, message: Message) -> ProcessedContent:
        """Single entry point for message processing"""
        try:
            # Channel validation
            if not self._channel_manager.is_channel_allowed(
                message.channel.id
            ):
                return None

            # Process content and attachments
            return await self._attachment_manager.process_message_content(
                message.content[len(self.config.command_prefix):],
                [a.__dict__ for a in message.attachments],
            )
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            return None

    async def _handle_message(self, message: Message):
        """Simplified message handling"""
        try:
            processed_content = await self.process_message(message)
            if not processed_content:
                return

            response: AgentResponse = await self.orchestrator.route_request(
                user_input=processed_content.text,
                user_id=str(message.author.id),
                session_id=f"discord_{message.channel.id}_{message.author.id}",
                platform="discord",
                additional_params={
                    "channel_id": str(message.channel.id),
                    "guild_id": (
                        str(message.guild.id) if message.guild else None
                    ),
                    "attachments": processed_content.attachments,
                    "username": message.author.name,
                },
            )

            await self._send_response(message, response)

        except Exception as e:
            self.logger.error(f"Error handling message: {str(e)}")
            await message.channel.send(
                "Sorry, I encountered an error processing your request."
            )

    async def _send_response(
        self, original_message: Message, response: AgentResponse
    ):
        """Send response with retry logic"""
        if not isinstance(response.output, ConversationMessage):
            await original_message.channel.send(str(response.output))
            return

        content = response.output.content[0]["text"]
        chunks = self._split_content(content)

        for chunk in chunks:
            for attempt in range(self.config.retry_attempts):
                try:
                    await original_message.channel.send(chunk)
                    break
                except Exception as e:
                    if attempt == self.config.retry_attempts - 1:
                        self.logger.error(
                            f"Failed to send message after {self.config.retry_attempts} attempts: {str(e)}"
                        )
                        await original_message.channel.send(
                            "Sorry, I encountered an error sending the response."
                        )
                    else:
                        await asyncio.sleep(self.config.retry_delay)

    def _split_content(self, content: str) -> List[str]:
        """Split content into Discord-friendly chunks"""
        if not content:
            return []

        chunks = []
        current_chunk = ""

        for line in content.split("\n"):
            if (
                len(current_chunk) + len(line) + 1
                > self.config.max_message_length
            ):
                chunks.append(current_chunk)
                current_chunk = line
            else:
                current_chunk += "\n" + line if current_chunk else line

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def set_orchestrator(self, orchestrator):
        """Set orchestrator reference"""
        self.orchestrator = orchestrator

    async def start(self):
        """Start the Discord client"""
        await self.client.start(self.token)

    def _setup_handlers(self):
        @self.client.event
        async def on_ready():
            self.logger.info(f"Logged in as {self.client.user}")
            for guild in self.client.guilds:
                await self._channel_manager.initialize_guild(guild)

        @self.client.event
        async def on_guild_join(guild):
            await self._channel_manager.initialize_guild(guild)

        @self.client.event
        async def on_message(message: Message):
            if message.author == self.client.user:
                return

            if not message.content.startswith(self.config.command_prefix):
                return
            await self._handle_message(message)
