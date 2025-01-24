from dataclasses import dataclass
from typing import Any, Optional, Dict, List, Union
from MAX.retrievers import Retriever
from MAX.storage import ChatStorage, ChromaDBChatStorage
from MAX.utils import Logger
from MAX.storage import ChromaDB


@dataclass
class KnowledgeBasesRetrieverOptions:
    """Options for Knowledge Bases Retriever."""

    storage_client: ChatStorage
    collection_name: str
    max_results: int = 5
    similarity_threshold: float = 0.7


class KnowledgeBasesRetriever(Retriever):
    def __init__(self, options: KnowledgeBasesRetrieverOptions):
        super().__init__(options)
        self.options = options
        self.storage = options.storage_client
        self.logger = Logger()

    async def retrieve(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve information based on input text using the configured storage.
        """
        try:
            if isinstance(self.storage, ChromaDBChatStorage):
                return await self._retrieve_from_chroma(text, **kwargs)
            elif isinstance(self.storage, ChromaDB):
                return await self._retrieve_from_mongo(text, **kwargs)
            else:
                # Fallback to basic retrieval for other storage types
                return await self._retrieve_basic(text, **kwargs)
        except Exception as e:
            self.logger.error(f"Error in retrieve: {str(e)}")
            return []

    async def _retrieve_from_chroma(
        self, text: str, **kwargs
    ) -> List[Dict[str, Any]]:
        """Specialized retrieval for ChromaDB"""
        try:
            results = self.storage.collection.query(
                query_texts=[text],
                n_results=self.options.max_results,
                where=kwargs.get("filters", {}),
            )

            return [
                {"content": {"text": doc}, "metadata": metadata}
                for doc, metadata in zip(
                    results["documents"][0], results["metadatas"][0]
                )
            ]
        except Exception as e:
            self.logger.error(f"ChromaDB retrieval error: {str(e)}")
            return []

    async def _retrieve_from_mongo(
        self, text: str, **kwargs
    ) -> List[Dict[str, Any]]:
        """Specialized retrieval for MongoDB"""
        try:
            # Use MongoDB text search capabilities
            cursor = (
                self.storage.collection.find(
                    {"$text": {"$search": text}, **kwargs.get("filters", {})},
                    {"score": {"$meta": "textScore"}},
                )
                .sort([("score", {"$meta": "textScore"})])
                .limit(self.options.max_results)
            )

            results = []
            async for doc in cursor:
                results.append(
                    {
                        "content": {"text": doc.get("content", "")},
                        "metadata": {
                            "score": doc.get("score", 0),
                            "timestamp": doc.get("timestamp", None),
                            **doc.get("metadata", {}),
                        },
                    }
                )
            return results
        except Exception as e:
            self.logger.error(f"MongoDB retrieval error: {str(e)}")
            return []

    async def _retrieve_basic(
        self, text: str, **kwargs
    ) -> List[Dict[str, Any]]:
        """Basic retrieval for other storage types"""
        try:
            # Fetch recent messages and perform basic text matching
            messages = await self.storage.fetch_all_chats(
                user_id=kwargs.get("user_id", ""),
                session_id=kwargs.get("session_id", ""),
            )

            # Basic text matching (you might want to implement more sophisticated matching)
            results = []
            for msg in messages:
                if text.lower() in msg.content[0]["text"].lower():
                    results.append(
                        {
                            "content": {"text": msg.content[0]["text"]},
                            "metadata": msg.content[0].get("metadata", {}),
                        }
                    )
                    if len(results) >= self.options.max_results:
                        break
            return results
        except Exception as e:
            self.logger.error(f"Basic retrieval error: {str(e)}")
            return []

    async def retrieve_and_combine_results(self, text: str, **kwargs) -> str:
        """Combine retrieved results into a single text"""
        results = await self.retrieve(text, **kwargs)
        return "\n".join(
            result["content"]["text"]
            for result in results
            if result
            and result.get("content")
            and isinstance(result["content"].get("text"), str)
        )

    async def retrieve_and_generate(self, text: str, **kwargs) -> str:
        """Generate new content based on retrieved results"""
        results = await self.retrieve(text, **kwargs)
        if not results:
            return "No relevant information found."

        # Combine retrieved information and generate a summary
        combined_info = "\n".join(
            f"- {result['content']['text']} (relevance: {result['metadata'].get('score', 'N/A')})"
            for result in results
        )

        return f"Based on the retrieved information:\n{combined_info}"
