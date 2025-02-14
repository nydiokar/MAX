from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum
from collections import defaultdict
import asyncio
from uuid import uuid4
from datetime import datetime


class DataProvenance(Enum):
    TWITTER_VERIFIED_ACCOUNT = "twitter_verified"
    TWITTER_KEYWORD_MENTION = "twitter_keyword"


class TwitterAccountTier(Enum):
    CRITICAL = "critical"
    IMPORTANT = "important"


@dataclass
class TwitterMonitorConfig:
    tracked_accounts: Dict[str, TwitterAccountTier]
    keyword_patterns: list
    min_follower_count: int
    min_account_age_days: int
    engagement_thresholds: Dict[str, int]


class FetcherAgent:
    def __init__(self, twitter_client, config: TwitterMonitorConfig):
        self.twitter_client = twitter_client
        self.config = config
        self.monitored_accounts = defaultdict(set)
        self._init_monitoring()

    def _init_monitoring(self):
        for account, tier in self.config.tracked_accounts.items():
            self.monitored_accounts[tier].add(account)
        self.keyword_patterns = self.config.keyword_patterns

    async def monitor_sources(self):
        await asyncio.gather(
            self._monitor_critical_accounts(), self._monitor_keywords()
        )

    async def _monitor_critical_accounts(self):
        critical_accounts = self.monitored_accounts[
            TwitterAccountTier.CRITICAL
        ]
        async for tweet in self.twitter_client.stream_users(critical_accounts):
            await self._handle_tweet(
                tweet, DataProvenance.TWITTER_VERIFIED_ACCOUNT
            )

    async def _monitor_keywords(self):
        async for tweet in self.twitter_client.filter_stream(
            track=self.keyword_patterns
        ):
            if await self._validate_tweet_source(tweet):
                await self._handle_tweet(
                    tweet, DataProvenance.TWITTER_KEYWORD_MENTION
                )

    async def _validate_tweet_source(self, tweet: Dict[str, Any]) -> bool:
        return (
            tweet["user"]["followers_count"] >= self.config.min_follower_count
            and (datetime.now() - tweet["user"]["created_at"]).days
            >= self.config.min_account_age_days
            and tweet["public_metrics"]["like_count"]
            >= self.config.engagement_thresholds["likes"]
        )

    async def _handle_tweet(
        self, tweet: Dict[str, Any], provenance: DataProvenance
    ):
        # Placeholder for further collaboration with expert agents
        print(f"Fetched tweet: {tweet['text']} with provenance {provenance}")
