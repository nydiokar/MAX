from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import numpy as np

class ToolSemanticMatcher:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        
    def compute_similarity(self, query: str, candidates: List[str]) -> List[Tuple[str, float]]:
        """
        Compute semantic similarity between query and candidate texts
        Returns list of (text, similarity_score) tuples
        """
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        candidate_embeddings = self.model.encode(candidates, convert_to_tensor=True)
        
        # Compute cosine similarities
        similarities = np.inner(query_embedding, candidate_embeddings)
        
        # Create (text, score) pairs
        results = [(text, float(score)) for text, score in zip(candidates, similarities)]
        return sorted(results, key=lambda x: x[1], reverse=True)
