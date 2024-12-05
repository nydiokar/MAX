# server.py
import os
import json
import logging
from typing import Sequence
from datetime import datetime
import httpx
from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("brave-search")

WEB_SEARCH_TOOL = Tool(
    name="brave_web_search",
    description=(
        "Performs a web search using the Brave Search API, ideal for general queries, news, articles, and online content. " +
        "Use this for broad information gathering, recent events, or when you need diverse web sources. " +
        "Supports pagination, content filtering, and freshness controls. " +
        "Maximum 20 results per request, with offset for pagination. "
    ),
    inputSchema={
        "type": "object", 
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (max 400 chars)"
            },
            "count": {
                "type": "number", 
                "description": "Results (1-20)",
                "default": 10
            },
            "offset": {
                "type": "number",
                "description": "Pagination offset",
                "default": 0
            }
        },
        "required": ["query"]
    }
)

LOCAL_SEARCH_TOOL = Tool(
    name="brave_local_search",
    description=(
        "Searches for local businesses and places. Returns names, "
        "addresses, ratings, hours, and more."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string", 
                "description": "Local search query"
            },
            "count": {
                "type": "number",
                "default": 5,
                "description": "Number of results"
            }
        },
        "required": ["query"]
    }
)

class BraveSearchServer(Server):
    def __init__(self):
        super().__init__("brave-search")
        self.api_key = os.getenv("BRAVE_API_KEY")
        if not self.api_key:
            raise ValueError("BRAVE_API_KEY required")
            
        self.http_client = httpx.AsyncClient(
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key
            }
        )

    @Server.list_tools()
    async def list_tools(self) -> list[Tool]:
        return [WEB_SEARCH_TOOL, LOCAL_SEARCH_TOOL]

    @Server.call_tool()  
    async def call_tool(
        self, 
        name: str, 
        arguments: dict
    ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        try:
            if name == "brave_web_search":
                results = await self._web_search(arguments)
            elif name == "brave_local_search":
                results = await self._local_search(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")

            return [TextContent(type="text", text=results)]

        except Exception as e:
            logger.error(f"Tool error: {str(e)}")
            return [TextContent(
                type="text",
                text=f"Error: {str(e)}"
            )]

    async def _web_search(self, args: dict) -> str:
        response = await self.http_client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={
                "q": args["query"],
                "count": min(args.get("count", 10), 20),
                "offset": args.get("offset", 0)
            }
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for result in data.get("web", {}).get("results", []):
            results.append(
                f"Title: {result.get('title', '')}\n"
                f"Description: {result.get('description', '')}\n"
                f"URL: {result.get('url', '')}"
            )
        return "\n\n".join(results)

    async def _local_search(self, args: dict) -> str:
        # Local search implementation following TypeScript version
        # Omitted for brevity but includes POI lookup etc.
        pass

async def main():
    server = BraveSearchServer()
    async with stdio_server() as (read, write):
        await server.run(
            read,
            write,
            InitializationOptions(
                server_name="brave-search",
                server_version="0.1.0",
                capabilities=server.get_capabilities()
            )
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())