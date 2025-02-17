"""test_kpu.py - Complete test suite for KPU implementation"""

import pytest
import asyncio
from typing import List, Dict, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from MAX.KPU.kpu import KPU
from MAX.retrievers import Retriever

class MockRetriever(Retriever):
    """Mock implementation of the Retriever for testing KPU."""
    
    def __init__(self, return_data: Optional[Dict[str, Any]] = None):
        # Initialize with mock configuration
        super().__init__({"type": "mock", "name": "mock_retriever"})
        self.return_data = return_data or {
            'context': '',
            'facts': []
        }
    
    async def retrieve(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Mock implementation of retrieve method."""
        facts = self.return_data.get('facts', [])
        if kwargs.get('filters'):
            # Apply any filters if provided
            filter_type = kwargs['filters'].get('type')
            if filter_type:
                facts = [f for f in facts if f.get('metadata', {}).get('type') == filter_type]
        return facts
    
    async def retrieve_and_combine_results(self, query: str, **kwargs) -> str:
        """Mock implementation of retrieve_and_combine_results method."""
        return self.return_data.get('context', '')
    
    async def retrieve_and_generate(self, query: str, **kwargs) -> str:
        """Mock implementation of retrieve_and_generate method."""
        return f"Generated content for query: {query}"

@pytest.fixture
def mock_knowledge_base():
    """Fixture providing mock knowledge base data."""
    return {
        'context': 'Python is a high-level programming language.',
        'facts': [
            {
                'content': {'text': 'Python is a programming language'},
                'metadata': {'type': 'fact', 'confidence': 0.9, 'source': 'test'}
            },
            {
                'content': {'text': 'Python is good for beginners'},
                'metadata': {'type': 'fact', 'confidence': 0.85, 'source': 'test'}
            },
            {
                'content': {'text': 'Python setup requires installation and configuration'},
                'metadata': {'type': 'task_pattern', 'confidence': 0.9, 'agent': 'setup_agent'}
            }
        ]
    }

@pytest.fixture
def base_kpu(mock_knowledge_base):
    """Fixture providing a basic KPU instance with mock data."""
    return KPU(retriever=MockRetriever(mock_knowledge_base))

@pytest.mark.asyncio
async def test_basic_request_processing(base_kpu):
    """Test basic request processing functionality."""
    result = await base_kpu.process_request(
        user_input="What is Python?",
        selected_agent="test_agent",
        conversation_history=[]
    )
    
    # Verify response structure
    assert isinstance(result, dict)
    assert all(key in result for key in [
        "enriched_context",
        "suggested_decomposition",
        "collaborators",
        "verification_results",
        "feedback_required",
        "conversation_history"
    ])
    
    # Verify content
    assert "Python" in result["enriched_context"]
    assert isinstance(result["verification_results"], dict)
    assert isinstance(result["collaborators"], list)

@pytest.mark.asyncio
async def test_process_request_without_retriever():
    """Test that process_request works correctly without a retriever."""
    kpu = KPU(retriever=None)
    
    result = await kpu.process_request(
        user_input="Tell me about Python",
        selected_agent="test_agent",
        conversation_history=[{"role": "user", "content": "Previous message"}]
    )
    
    # Verify default/empty values when no retriever is present
    assert result["enriched_context"] == ""
    assert isinstance(result["suggested_decomposition"], list)
    assert isinstance(result["collaborators"], list)
    assert isinstance(result["verification_results"], dict)
    assert isinstance(result["feedback_required"], list)
    assert len(result["conversation_history"]) == 1

@pytest.mark.asyncio
async def test_context_enrichment(mock_knowledge_base):
    """Test context enrichment functionality."""
    kpu = KPU(retriever=MockRetriever(mock_knowledge_base))
    
    result = await kpu.process_request(
        user_input="Tell me about Python programming",
        selected_agent="test_agent",
        conversation_history=[]
    )
    
    # Verify context was enriched from knowledge base
    assert result["enriched_context"] == mock_knowledge_base["context"]
    assert "Python" in result["enriched_context"]
    assert "programming" in result["enriched_context"]

@pytest.mark.asyncio
async def test_knowledge_verification(base_kpu):
    """Test knowledge verification functionality."""
    result = await base_kpu.process_request(
        user_input="Python is a programming language for beginners",
        selected_agent="test_agent",
        conversation_history=[]
    )
    
    verification = result["verification_results"]
    assert "verified_facts" in verification
    assert "uncertain_claims" in verification
    assert "contradictions" in verification

@pytest.mark.asyncio
async def test_conversation_history_handling(base_kpu):
    """Test handling of conversation history."""
    history = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."}
    ]
    
    result = await base_kpu.process_request(
        user_input="Is it good for beginners?",
        selected_agent="test_agent",
        conversation_history=history
    )
    
    # Verify history is maintained
    assert result["conversation_history"] == history
    assert isinstance(result["enriched_context"], str)

@pytest.mark.asyncio
async def test_collaborator_identification(base_kpu):
    """Test identification of potential collaborators."""
    result = await base_kpu.process_request(
        user_input="Need to collaborate with the data team on this",
        selected_agent="test_agent",
        conversation_history=[]
    )
    
    assert isinstance(result["collaborators"], list)
    collaborators = result["collaborators"]
    if collaborators:  # If collaborators were identified
        assert all(isinstance(c, dict) for c in collaborators)
        assert all("type" in c for c in collaborators)
        assert all("confidence" in c for c in collaborators)

@pytest.mark.asyncio
async def test_feedback_point_identification(base_kpu):
    """Test identification of feedback points."""
    result = await base_kpu.process_request(
        user_input="Please confirm if this is the correct approach",
        selected_agent="test_agent",
        conversation_history=[]
    )
    
    assert isinstance(result["feedback_required"], list)
    feedback_points = result["feedback_required"]
    if feedback_points:  # If feedback points were identified
        assert all(isinstance(f, dict) for f in feedback_points)
        assert all("type" in f for f in feedback_points)
        assert all("priority" in f for f in feedback_points)

@pytest.mark.asyncio
async def test_task_decomposition(base_kpu):
    """Test task decomposition functionality."""
    result = await base_kpu.process_request(
        user_input="Help me set up Python",
        selected_agent="setup_agent",
        conversation_history=[]
    )
    
    assert isinstance(result["suggested_decomposition"], list)

@pytest.mark.asyncio
async def test_empty_input_handling():
    """Test handling of empty input."""
    # Create a clean KPU instance with empty mock data
    empty_mock_data = {
        'context': '',
        'facts': []
    }
    kpu = KPU(retriever=MockRetriever(empty_mock_data))
    
    result = await kpu.process_request(
        user_input="",
        selected_agent="test_agent",
        conversation_history=[]
    )
    
    # Should handle empty input gracefully
    assert isinstance(result, dict)
    assert result["enriched_context"] == ""
    assert isinstance(result["verification_results"], dict)

@pytest.mark.asyncio
async def test_semantic_similarity():
    """Test semantic similarity computation."""
    kpu = KPU()
    
    # Test similar texts
    similarity = kpu._compute_semantic_similarity(
        "Python is a programming language",
        "Python is a coding language"
    )
    assert 0 <= similarity <= 1
    assert similarity > 0.5  # Similar texts should have high similarity

    # Test different texts
    similarity = kpu._compute_semantic_similarity(
        "Python is a programming language",
        "The weather is nice today"
    )
    assert 0 <= similarity <= 1
    assert similarity < 0.5  # Different texts should have low similarity

if __name__ == "__main__":
    asyncio.run(pytest.main([__file__]))