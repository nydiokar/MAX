import asyncio
import logging
from aiohttp import web
from typing import Dict, Any
from datetime import datetime
import signal
import sys
from MAX.adapters.fetchers.test_fetcher import AbstractFetcher, FetchResult, FetchStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fetcher_simulation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GracefulExit(SystemExit):
    pass

def signal_handler(signum, frame):
    raise GracefulExit()

class TestFetcher(AbstractFetcher[Dict[str, Any]]):
    """Test implementation of AbstractFetcher."""
    
    @property
    def default_headers(self) -> Dict[str, str]:
        return {
            "User-Agent": "TestFetcher/1.0",
            "Accept": "application/json"
        }

    async def process_response(self, response) -> Dict[str, Any]:
        return await response.json()

    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> FetchResult[Dict[str, Any]]:
        logger.error(f"Error in fetch operation: {str(error)}, Context: {context}")
        return FetchResult(
            status=FetchStatus.FAILED,
            error=str(error)
        )

async def create_test_server():
    """Create a test server with various endpoints."""
    
    request_count = 0
    
    async def handler(request):
        nonlocal request_count
        request_count += 1
        
        # Simulate rate limiting
        if request_count % 5 == 0:
            return web.Response(
                status=429,
                headers={'Retry-After': '1'}
            )
        
        # Simulate random errors
        if request_count % 7 == 0:
            return web.Response(status=500)
            
        # Successful response
        return web.json_response({
            'request_id': request_count,
            'timestamp': datetime.now().isoformat(),
            'message': 'Test response'
        })

    app = web.Application()
    app.router.add_get('/test', handler)
    app.router.add_post('/test', handler)
    
    return app

async def cleanup(runner: web.AppRunner, fetcher: TestFetcher):
    """Cleanup function to ensure proper resource handling."""
    logger.info("Cleaning up resources...")
    try:
        await fetcher.disconnect()
        await runner.cleanup()
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

async def run_simulation():
    """Run the fetcher simulation."""
    runner = None
    fetcher = None
    
    try:
        logger.info("Starting fetcher simulation")
        
        # Start test server
        app = await create_test_server()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', 8080)
        await site.start()
        logger.info("Test server started")

        # Create fetcher
        fetcher = TestFetcher(
            base_url='http://localhost:8080',
            requests_per_second=5.0,
            max_retries=3,
            retry_delay=0.5
        )
        await fetcher.connect()
        
        # Test single requests
        logger.info("\nTesting single requests:")
        for i in range(10):
            result = await fetcher.fetch('test')
            logger.info(f"Request {i + 1} Result: {result}")
            await asyncio.sleep(0.2)  # Small delay between requests
        
        # Test bulk requests
        logger.info("\nTesting bulk requests:")
        bulk_results = await fetcher.bulk_fetch([
            {'endpoint': 'test'} for _ in range(5)
        ])
        logger.info(f"Bulk results: {bulk_results}")
        
        # Print statistics
        logger.info("\nFinal Statistics:")
        logger.info(fetcher.stats)
        
    except Exception as e:
        logger.error(f"Error during simulation: {e}")
        raise
    finally:
        if runner or fetcher:
            await cleanup(runner, fetcher)
        logger.info("Simulation completed")

def main():
    """Main function with proper signal handling."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        asyncio.run(run_simulation())
    except GracefulExit:
        logger.info("Received shutdown signal, exiting gracefully...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    
if __name__ == "__main__":
    main()