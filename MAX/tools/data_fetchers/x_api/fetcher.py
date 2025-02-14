from datetime import datetime
from typing import Dict, Optional, Union, List
from urllib.parse import urlencode
from prometheus_client import CollectorRegistry, REGISTRY

from aiohttp import ClientResponse
import backoff

from ..base.fetcher import AbstractFetcher, FetcherConfig
from ..base.types import FetchResult, FetchStatus
from .exceptions import (
    XAPIError,
    XAPIAuthError,
    XAPIResourceNotFound,
    XAPIInvalidRequest,
)
from .types import Tweet, User, TweetMetrics, UserMetrics
from MAX.config.settings import settings
from MAX.utils import Logger

logger = Logger.getLogger(__name__)


class XApiFetcher(AbstractFetcher[Union[Tweet, User]]):
    """X API v2 fetcher implementation."""

    _instance_count = 0  # Class variable to track instances

    def __init__(self):
        # Clear existing metrics for this instance
        try:
            # Only unregister if this isn't the first instance
            if XApiFetcher._instance_count > 0:
                for collector in list(REGISTRY._collector_to_names.keys()):
                    if collector._name.startswith("x_api_"):
                        REGISTRY.unregister(collector)
        except Exception as e:
            logger.warn(f"Error clearing metrics: {e}")

        XApiFetcher._instance_count += 1

        config = FetcherConfig(
            base_url="https://api.twitter.com/2/",
            requests_per_second=0.1,  # Reduced to 6 requests per minute
            cache_ttl=300,
            timeout=30.0,
            max_retries=3,
            retry_delay=15.0,
            circuit_breaker_threshold=5,
            circuit_breaker_timeout=300,
            enable_metrics=XApiFetcher._instance_count
            == 1,  # Only enable metrics for first instance
        )

        # Debug logging with timestamp
        logger.info(
            f"[{datetime.now().isoformat()}] Initializing X API Fetcher (Instance #{XApiFetcher._instance_count})"
        )
        logger.info(
            f"[{datetime.now().isoformat()}] Bearer Token available: {bool(settings.X_BEARER_TOKEN)}"
        )
        logger.info(
            f"[{datetime.now().isoformat()}] API Key available: {bool(settings.X_API_KEY)}"
        )
        logger.info(
            f"[{datetime.now().isoformat()}] API Secret available: {bool(settings.X_API_SECRET)}"
        )

        # Initialize with base config
        super().__init__(config, name="x_api")

        # Validate required settings
        if not all(
            [
                settings.X_BEARER_TOKEN,
                settings.X_API_KEY,
                settings.X_API_SECRET,
            ]
        ):
            raise XAPIAuthError("Missing required X API credentials")

    def __del__(self):
        """Cleanup when instance is destroyed."""
        XApiFetcher._instance_count -= 1

    @property
    def default_headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {settings.X_BEARER_TOKEN}",
            "Content-Type": "application/json",
            "User-Agent": "MAX+/1.0",
        }
        logger.debug(f"Using headers: {headers}")
        return headers

    async def process_response(
        self, response: ClientResponse
    ) -> Union[Tweet, User]:
        """Process X API response with error handling."""
        logger.info(
            f"[{datetime.now().isoformat()}] Processing response: {response.status}"
        )
        if response.status == 401:
            raise XAPIAuthError("Invalid authentication credentials")
        elif response.status == 404:
            raise XAPIResourceNotFound("Requested resource not found")

        data = await response.json()

        if "errors" in data:
            errors = data["errors"]
            raise XAPIInvalidRequest(
                f"X API returned {len(errors)} errors", errors=errors
            )

        if "data" not in data:
            logger.warn("Empty response from X API")
            return None

        item = data["data"]
        if "text" in item:  # Tweet
            return Tweet(
                id=item["id"],
                text=item["text"],
                created_at=datetime.fromisoformat(
                    item["created_at"].replace("Z", "+00:00")
                ),
                author_id=item["author_id"],
                metrics=TweetMetrics(**item.get("public_metrics", {})),
                referenced_tweets=item.get("referenced_tweets", []),
                context_annotations=item.get("context_annotations", []),
                entities=item.get("entities", {}),
            )
        else:  # User
            return User(
                id=item["id"],
                username=item["username"],
                name=item["name"],
                metrics=UserMetrics(**item.get("public_metrics", {})),
                verified=item.get("verified", False),
                protected=item.get("protected", False),
                description=item.get("description"),
                location=item.get("location"),
            )

    @backoff.on_exception(
        backoff.expo,
        (XAPIError, XAPIAuthError),
        max_tries=3,
        max_time=300,  # Maximum total time to retry
    )
    async def get_tweet(self, tweet_id: str) -> FetchResult[Tweet]:
        """Fetch a single tweet by ID."""
        logger.info(f"Fetching tweet: {tweet_id}")
        params = {
            "tweet.fields": "created_at,public_metrics,referenced_tweets,context_annotations,entities",
            "expansions": "author_id",
            "user.fields": "verified,protected,public_metrics",
        }

        return await self.fetch(f"tweets/{tweet_id}", params=params)

    async def search_tweets(
        self,
        query: str,
        max_results: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        **kwargs,
    ) -> FetchResult[List[Tweet]]:
        """Search recent tweets."""
        params = {
            "query": query,
            "max_results": min(max_results, 100),
            "tweet.fields": "created_at,public_metrics,referenced_tweets,context_annotations",
            "expansions": "author_id",
            "user.fields": "verified,protected,public_metrics",
        }

        if start_time:
            params["start_time"] = start_time.isoformat() + "Z"
        if end_time:
            params["end_time"] = end_time.isoformat() + "Z"

        params.update(kwargs)

        return await self.fetch("tweets/search/recent", params=params)
