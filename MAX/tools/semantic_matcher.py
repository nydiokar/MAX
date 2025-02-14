import spacy
from typing import List, Tuple, Dict, Any
from pydantic import BaseModel, ConfigDict


class SemanticMatcherConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    threshold: float = 0.5
    max_matches: int = 5
    settings: Dict[str, Any] = {}


class ToolSemanticMatcher:
    def __init__(self, config: SemanticMatcherConfig = None):
        self.config = config or SemanticMatcherConfig()
        """
        Initialize with a spaCy model. Options include:
        - 'en_core_web_sm' (small)
        - 'en_core_web_md' (medium)
        - 'en_core_web_lg' (large)
        """
        try:
            self.nlp = spacy.load("en_core_web_md")
        except OSError:
            # Fallback to simpler comparison if model not available
            self.nlp = None
            print(
                "Warning: spaCy model not found. Using basic string matching instead."
            )

    def compute_similarity(
        self, query: str, candidates: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Compute semantic similarity between query and candidate texts
        Returns list of (text, similarity_score) tuples sorted by similarity
        """
        if self.nlp is None:
            # Fallback to basic string matching
            results = []
            query_lower = query.lower()
            for text in candidates:
                # Simple word overlap similarity
                text_lower = text.lower()
                words_query = set(query_lower.split())
                words_text = set(text_lower.split())
                similarity = len(words_query.intersection(words_text)) / max(
                    len(words_query), len(words_text)
                )
                results.append((text, float(similarity)))
            return sorted(results, key=lambda x: x[1], reverse=True)

        # Use spaCy if available
        query_doc = self.nlp(query)
        results = []
        for text in candidates:
            candidate_doc = self.nlp(text)
            similarity = query_doc.similarity(candidate_doc)
            results.append((text, float(similarity)))

        return sorted(results, key=lambda x: x[1], reverse=True)
