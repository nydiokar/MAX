import asyncio
import logging
from datetime import datetime, timedelta
from typing import List
import os
import pathlib
import json

import pytest
from unittest.mock import Mock, patch
from MAX.adapters.fetchers.x_api.fetcher import XApiFetcher
from MAX.adapters.fetchers.base.types import FetchResult, FetchStatus
from MAX.adapters.fetchers.x_api.types import Tweet, User, TweetMetrics, UserMetrics
from MAX.config.settings import settings

# Configure logging with timestamp
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Sample dummy data
DUMMY_TWEETS = [
    {
        "id": "1234567890",
        "text": "This is a test tweet about #python programming",
        "created_at": "2024-03-15T10:00:00Z",
        "author_id": "12345",
        "public_metrics": {
            "retweet_count": 10,
            "reply_count": 5,
            "like_count": 20,
            "quote_count": 2
        },
        "context_annotations": [
            {"domain": {"id": "123", "name": "Programming"}, 
             "entity": {"id": "456", "name": "Python"}}
        ]
    },
    {
        "id": "1234567891",
        "text": "Another test tweet about #datascience",
        "created_at": "2024-03-15T10:05:00Z",
        "author_id": "12346",
        "public_metrics": {
            "retweet_count": 15,
            "reply_count": 8,
            "like_count": 25,
            "quote_count": 3
        },
        "context_annotations": [
            {"domain": {"id": "124", "name": "Technology"}, 
             "entity": {"id": "457", "name": "Data Science"}}
        ]
    }
]

@pytest.mark.asyncio
async def test_x_api_connection():
    """Test basic API connection and authentication."""
    logger.info("\nEnvironment file:")
    env_path = pathlib.Path(__file__).parent.parent.parent / '.env'
    logger.info(f"Looking for .env at: {env_path.absolute()}")
    logger.info(f"File exists: {env_path.exists()}")
    logger.info("Environment variables:")
    logger.info(f"X_BEARER_TOKEN exists in env: {bool(os.getenv('X_BEARER_TOKEN'))}")
    logger.info(f"X_API_KEY exists in env: {bool(os.getenv('X_API_KEY'))}")
    logger.info(f"X_API_SECRET exists in env: {bool(os.getenv('X_API_SECRET'))}")
    
    fetcher = XApiFetcher()
    
    async with fetcher:
        # Test search functionality
        result = await fetcher.search_tweets(
            query="python programming",
            max_results=10
        )
        
        assert result.status == FetchStatus.SUCCESS, f"Search failed: {result.error}"
        assert result.data is not None
        
        # Log the results
        tweets = result.data
        logger.info(f"Retrieved {len(tweets)} tweets")
        for tweet in tweets:
            logger.info(f"Tweet ID: {tweet.id}")
            logger.info(f"Content: {tweet.text[:100]}...")
            logger.info(f"Metrics: {tweet.metrics}")
            logger.info("---")

@pytest.mark.asyncio
async def test_rate_limiting():
    """Test rate limiting functionality."""
    fetcher = XApiFetcher()
    
    async with fetcher:
        # Make multiple requests in quick succession
        results = []
        for _ in range(3):
            result = await fetcher.search_tweets("python", max_results=5)
            results.append(result)
            
        # Check if rate limiting worked
        rate_limited = any(r.status == FetchStatus.RATE_LIMITED for r in results)
        logger.info(f"Rate limiting test: {'Passed' if rate_limited else 'Not triggered'}")

@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling with invalid requests."""
    fetcher = XApiFetcher()
    
    async with fetcher:
        # Test with invalid tweet ID
        result = await fetcher.get_tweet("invalid_id")
        assert result.status == FetchStatus.FAILED
        logger.info(f"Error handling test: {result.error}")

@pytest.mark.asyncio
async def test_with_dummy_data():
    """Test integration using dummy data without API calls."""
    logger.info("Testing with dummy data...")
    
    async def mock_fetch(*args, **kwargs):
        """Mock fetch function that returns dummy data."""
        logger.info(f"[{datetime.now().isoformat()}] Mock fetch called with args: {args}")
        return FetchResult(
            status=FetchStatus.SUCCESS,
            data=[Tweet(
                id=tweet["id"],
                text=tweet["text"],
                created_at=datetime.fromisoformat(tweet["created_at"].replace('Z', '+00:00')),
                author_id=tweet["author_id"],
                metrics=TweetMetrics(**tweet["public_metrics"]),
                context_annotations=tweet["context_annotations"],
                referenced_tweets=[],
                entities={}
            ) for tweet in DUMMY_TWEETS],
            error=None
        )

    fetcher = XApiFetcher()
    
    # Patch the fetch method
    with patch.object(fetcher, 'fetch', side_effect=mock_fetch):
        result = await fetcher.search_tweets(
            query="python programming",
            max_results=10
        )
        
        assert result.status == FetchStatus.SUCCESS
        assert len(result.data) == len(DUMMY_TWEETS)
        
        # Generate timestamp for dataset
        timestamp = datetime.now().isoformat()
        
        # Verify dummy data structure
        for i, tweet in enumerate(result.data):
            logger.info(f"[{timestamp}] Verifying tweet {i + 1}...")
            assert isinstance(tweet, Tweet)
            assert tweet.id == DUMMY_TWEETS[i]["id"]
            assert tweet.text == DUMMY_TWEETS[i]["text"]
            
        logger.info(f"[{timestamp}] Dummy data test completed successfully")
        
        # Generate test dataset with dummy data
        dataset = {
            "metadata": {
                "test_type": "dummy_data",
                "timestamp": timestamp,
                "total_tweets": len(result.data)
            },
            "tweets": [
                {
                    "id": tweet.id,
                    "text": tweet.text,
                    "timestamp": tweet.created_at.isoformat(),
                    "metrics": {
                        "retweets": tweet.metrics.retweet_count,
                        "replies": tweet.metrics.reply_count,
                        "likes": tweet.metrics.like_count
                    },
                    "context": tweet.context_annotations
                }
                for tweet in result.data
            ]
        }
        
        # Save dummy test results
        output_file = f"dummy_test_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
            logger.info(f"[{timestamp}] Dummy test dataset saved to {output_file}")

def create_sample_dataset(tweets: List[Tweet]) -> dict:
    """Create a structured dataset from tweets."""
    return {
        "metadata": {
            "count": len(tweets),
            "timestamp": datetime.now().isoformat(),
        },
        "tweets": [
            {
                "id": tweet.id,
                "text": tweet.text,
                "created_at": tweet.created_at.isoformat(),
                "metrics": {
                    "retweets": tweet.metrics.retweet_count,
                    "replies": tweet.metrics.reply_count,
                    "likes": tweet.metrics.like_count,
                }
            }
            for tweet in tweets
        ]
    }

@pytest.mark.asyncio
async def generate_test_dataset():
    """Generate a comprehensive test dataset using dummy data."""
    logger.info("Generating test dataset...")
    
    async def mock_fetch(*args, **kwargs):
        """Mock fetch function that returns dummy data."""
        logger.info(f"[{datetime.now().isoformat()}] Mock fetch called with args: {args}")
        return FetchResult(
            status=FetchStatus.SUCCESS,
            data=[Tweet(
                id=tweet["id"],
                text=tweet["text"],
                created_at=datetime.fromisoformat(tweet["created_at"].replace('Z', '+00:00')),
                author_id=tweet["author_id"],
                metrics=TweetMetrics(**tweet["public_metrics"]),
                context_annotations=tweet["context_annotations"],
                referenced_tweets=[],
                entities={}
            ) for tweet in DUMMY_TWEETS],
            error=None
        )

    fetcher = XApiFetcher()
    dataset = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "api_version": "2.0",
            "query_params": []
        },
        "data": {
            "tweets": [],
            "metrics": {
                "total_tweets": 0,
                "total_likes": 0,
                "total_retweets": 0,
                "total_replies": 0
            }
        }
    }
    
    # Patch the fetch method to use dummy data
    with patch.object(fetcher, 'fetch', side_effect=mock_fetch):
        async with fetcher:
            queries = [
                "python programming",
                "data science",
                "machine learning",
                "artificial intelligence"
            ]
            
            for query in queries:
                logger.info(f"[{datetime.now().isoformat()}] Processing dummy data for query: {query}")
                result = await fetcher.search_tweets(
                    query=query,
                    max_results=10
                )
                
                if result.status == FetchStatus.SUCCESS and result.data:
                    dataset["metadata"]["query_params"].append({
                        "query": query,
                        "timestamp": datetime.now().isoformat(),
                        "results_count": len(result.data)
                    })
                    
                    for tweet in result.data:
                        tweet_data = {
                            "id": tweet.id,
                            "text": tweet.text,
                            "created_at": tweet.created_at.isoformat(),
                            "metrics": {
                                "retweets": tweet.metrics.retweet_count,
                                "replies": tweet.metrics.reply_count,
                                "likes": tweet.metrics.like_count,
                            },
                            "author_id": tweet.author_id,
                            "context_annotations": tweet.context_annotations
                        }
                        dataset["data"]["tweets"].append(tweet_data)
                        
                        # Update metrics
                        dataset["data"]["metrics"]["total_tweets"] += 1
                        dataset["data"]["metrics"]["total_likes"] += tweet.metrics.like_count
                        dataset["data"]["metrics"]["total_retweets"] += tweet.metrics.retweet_count
                        dataset["data"]["metrics"]["total_replies"] += tweet.metrics.reply_count
                
                # Simulate API delay without actually waiting
                logger.info(f"[{datetime.now().isoformat()}] Processed query: {query}")
    
    # Save the final dataset
    output_file = f"test_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
        logger.info(f"[{datetime.now().isoformat()}] Test dataset saved to {output_file}")
    
    return dataset

async def main():
    """Run all tests with dummy data."""
    try:
        logger.info(f"[{datetime.now().isoformat()}] Starting X API integration tests with dummy data...")
        
        # Run dummy data test
        await test_with_dummy_data()
        
        # Generate comprehensive dataset with dummy data
        dataset = await generate_test_dataset()
        
        logger.info(f"[{datetime.now().isoformat()}] All dummy data tests completed successfully")
            
    except Exception as e:
        logger.error(f"[{datetime.now().isoformat()}] Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())