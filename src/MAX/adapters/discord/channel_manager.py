from typing import Dict, List, Optional
from discord import Guild, TextChannel, Message

from MAX.types import DiscordAdapterConfig
from MAX.utils.logger import Logger

class ChannelManager:
    def __init__(self, config: DiscordAdapterConfig, logger: Optional[Logger] = None):
        self.config = config
        self.logger = logger or Logger()
        self.recent_messages: Dict[str, List[Message]] = {}

    def is_channel_allowed(self, channel_id: int) -> bool:
        """Check if interaction with channel is allowed"""
        if not self.config.allowed_channels:
            return True
        return channel_id in self.config.allowed_channels

    async def initialize_guild(self, guild: Guild):
        """Initialize new guild"""
        try:
            for channel in guild.text_channels:
                if self.is_channel_allowed(channel.id):
                    await self.cache_channel_messages(channel)
        except Exception as e:
            self.logger.error(f"Error initializing guild {guild.id}: {str(e)}")

    async def cache_channel_messages(self, channel: TextChannel):
        """Cache recent messages for context"""
        try:
            messages = await channel.history(
                limit=self.config.cache_message_count
            ).flatten()
            self.recent_messages[str(channel.id)] = messages
            self.logger.info(f"Cached {len(messages)} messages for channel {channel.id}")
        except Exception as e:
            self.logger.error(f"Error caching messages for channel {channel.id}: {str(e)}")