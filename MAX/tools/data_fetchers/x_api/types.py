from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TweetMetrics:
    retweet_count: int = 0
    reply_count: int = 0
    like_count: int = 0
    quote_count: int = 0
    impression_count: int = 0


@dataclass(frozen=True)
class Tweet:
    id: str
    text: str
    created_at: datetime
    author_id: str
    metrics: TweetMetrics
    referenced_tweets: List[Dict] = field(default_factory=list)
    context_annotations: List[Dict] = field(default_factory=list)
    entities: Dict = field(default_factory=dict)


@dataclass(frozen=True)
class UserMetrics:
    followers_count: int = 0
    following_count: int = 0
    tweet_count: int = 0
    listed_count: int = 0


@dataclass(frozen=True)
class User:
    id: str
    username: str
    name: str
    metrics: UserMetrics
    verified: bool = False
    protected: bool = False
    description: Optional[str] = None
    location: Optional[str] = None
