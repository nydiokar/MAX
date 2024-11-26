# To be revised, currently 2 agents in this file: fetcher and analyst - 
# -------------------- DO NOT USE --------------------

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Set
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from uuid import uuid4
from pydantic import BaseModel, validator
from collections import defaultdict

# Data Source Tracking (for provenance)
class DataProvenance(Enum):
    TWITTER_VERIFIED_ACCOUNT = "twitter_verified"
    TWITTER_KEYWORD_MENTION = "twitter_keyword"
    USER_CONTRIBUTION = "user_contribution"
    TELEGRAM_MONITOR = "telegram"
    DISCORD_MONITOR = "discord"

# Twitter Monitoring Configuration
class TwitterAccountTier(Enum):
    CRITICAL = "critical"  # Major influencers, must monitor all tweets
    IMPORTANT = "important"  # Monitor token-related tweets
    REGULAR = "regular"  # Monitor only when discussing tracked tokens

class TwitterMonitorConfig(BaseModel):
    tracked_accounts: Dict[str, TwitterAccountTier]
    keyword_patterns: List[str]
    min_follower_count: int
    min_account_age_days: int
    engagement_thresholds: Dict[str, int]

# Database Models
class TokenMention(BaseModel):
    id: str
    token_name: str
    source_type: DataProvenance
    source_id: str  # Twitter ID, user ID, etc.
    content: str
    timestamp: datetime
    engagement_metrics: Dict[str, int]
    sentiment_score: float
    verified: bool
    related_urls: List[str]
    
    class Config:
        orm_mode = True

class TokenInfo(BaseModel):
    token_name: str
    symbol: str
    first_seen: datetime
    mentions: List[TokenMention]
    verified_mentions_count: int
    total_mentions_count: int
    sentiment_scores: Dict[DataProvenance, float]
    price_data: Optional[Dict[str, float]]
    notable_accounts: Dict[str, int]  # account -> mention count
    confidence_score: float

@dataclass
class MemeTokenAnalystOptions(AgentOptions):
    db_client: Any
    twitter_api_client: Any
    user_registry: Dict[str, Any]
    twitter_config: TwitterMonitorConfig
    reporting_intervals: List[int] = field(default_factory=lambda: [3, 24])
    min_confidence_threshold: float = 0.7
    retriever: Optional[Any] = None

class MemeTokenAnalystAgent(Agent):
    def __init__(self, options: MemeTokenAnalystOptions):
        super().__init__(options)
        self.db = options.db_client
        self.twitter = options.twitter_api_client
        self.user_registry = options.user_registry
        self.twitter_config = options.twitter_config
        self.reporting_intervals = options.reporting_intervals
        self.min_confidence = options.min_confidence_threshold
        
        # Initialize monitoring strategies
        self._init_twitter_monitoring()

    def _init_twitter_monitoring(self):
        """Initialize both account-based and keyword-based monitoring"""
        self.monitored_accounts = defaultdict(set)
        for account, tier in self.twitter_config.tracked_accounts.items():
            self.monitored_accounts[tier].add(account)
        
        # Initialize keyword monitoring
        self.keyword_patterns = self.twitter_config.keyword_patterns

    async def monitor_twitter_sources(self):
        """Concurrent monitoring of both accounts and keywords"""
        await asyncio.gather(
            self._monitor_critical_accounts(),
            self._monitor_important_accounts(),
            self._monitor_keywords()
        )

    async def _monitor_critical_accounts(self):
        """Monitor all tweets from critical accounts"""
        critical_accounts = self.monitored_accounts[TwitterAccountTier.CRITICAL]
        async for tweet in self.twitter.stream_users(critical_accounts):
            await self._process_tweet(tweet, DataProvenance.TWITTER_VERIFIED_ACCOUNT)

    async def _monitor_keywords(self):
        """Monitor keyword mentions with filtering"""
        async for tweet in self.twitter.filter_stream(track=self.keyword_patterns):
            if await self._validate_tweet_source(tweet):
                await self._process_tweet(tweet, DataProvenance.TWITTER_KEYWORD_MENTION)

    async def _validate_tweet_source(self, tweet: Dict[str, Any]) -> bool:
        """Validate tweet source based on configured criteria"""
        return (
            tweet.user.followers_count >= self.twitter_config.min_follower_count and
            (datetime.now() - tweet.user.created_at).days >= self.twitter_config.min_account_age_days and
            tweet.engagement_metrics.get('likes', 0) >= self.twitter_config.engagement_thresholds['likes']
        )

    async def _process_tweet(self, tweet: Dict[str, Any], provenance: DataProvenance):
        """Process and store tweet information"""
        mention = TokenMention(
            id=str(uuid4()),
            token_name=await self._extract_token_name(tweet),
            source_type=provenance,
            source_id=tweet.id,
            content=tweet.text,
            timestamp=tweet.created_at,
            engagement_metrics={
                'likes': tweet.public_metrics.get('like_count', 0),
                'retweets': tweet.public_metrics.get('retweet_count', 0),
                'replies': tweet.public_metrics.get('reply_count', 0)
            },
            sentiment_score=await self._analyze_sentiment(tweet.text),
            verified=tweet.user.verified,
            related_urls=self._extract_urls(tweet)
        )
        
        # Store in database
        await self.db.store_mention(mention)
        
        # Update token info
        await self._update_token_info(mention)

    async def _update_token_info(self, mention: TokenMention):
        """Update or create token information in database"""
        token_info = await self.db.get_token_info(mention.token_name)
        
        if not token_info:
            token_info = TokenInfo(
                token_name=mention.token_name,
                symbol=await self._fetch_token_symbol(mention.token_name),
                first_seen=mention.timestamp,
                mentions=[mention],
                verified_mentions_count=1 if mention.verified else 0,
                total_mentions_count=1,
                sentiment_scores={mention.source_type: mention.sentiment_score},
                notable_accounts={mention.source_id: 1} if mention.verified else {},
                confidence_score=await self._calculate_confidence_score(mention)
            )
        else:
            # Update existing token info
            token_info.mentions.append(mention)
            token_info.total_mentions_count += 1
            if mention.verified:
                token_info.verified_mentions_count += 1
                token_info.notable_accounts[mention.source_id] = \
                    token_info.notable_accounts.get(mention.source_id, 0) + 1
            
            # Update sentiment scores
            current_score = token_info.sentiment_scores.get(mention.source_type, 0)
            count = sum(1 for m in token_info.mentions if m.source_type == mention.source_type)
            token_info.sentiment_scores[mention.source_type] = \
                (current_score * (count - 1) + mention.sentiment_score) / count
            
            token_info.confidence_score = await self._calculate_confidence_score(token_info)
        
        await self.db.update_token_info(token_info)

    async def _calculate_confidence_score(self, data: Union[TokenMention, TokenInfo]) -> float:
        """Calculate confidence score based on source and verification"""
        if isinstance(data, TokenMention):
            base_score = 0.5
            modifiers = {
                DataProvenance.TWITTER_VERIFIED_ACCOUNT: 0.3,
                DataProvenance.TWITTER_KEYWORD_MENTION: 0.1,
                DataProvenance.USER_CONTRIBUTION: 0.2
            }
            
            score = base_score + modifiers.get(data.source_type, 0)
            if data.verified:
                score += 0.2
                
            return min(score, 1.0)
        else:
            # For TokenInfo, consider overall metrics
            verified_ratio = data.verified_mentions_count / max(data.total_mentions_count, 1)
            notable_accounts_weight = len(data.notable_accounts) * 0.1
            return min(0.3 + verified_ratio * 0.4 + notable_accounts_weight, 1.0)