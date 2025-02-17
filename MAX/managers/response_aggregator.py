from typing import Dict, List, Any, Optional, Union
from enum import Enum
import json
from datetime import datetime
from dataclasses import dataclass
from MAX.utils.logger import Logger
from MAX.types import ConversationMessage, ParticipantRole

class AggregationStrategy(Enum):
    SEQUENTIAL = "sequential"  # Combine responses in sequence
    PARALLEL = "parallel"     # Merge parallel responses
    WEIGHTED = "weighted"     # Use weighted scoring
    VOTING = "voting"        # Use consensus/voting
    HYBRID = "hybrid"        # Combine multiple strategies

class ResponseType(Enum):
    TEXT = "text"
    STRUCTURED = "structured"
    CODE = "code"
    DATA = "data"
    ERROR = "error"

@dataclass
class AgentResponse:
    """Individual agent response with metadata."""
    agent_id: str
    content: Union[str, Dict[str, Any]]
    response_type: ResponseType
    timestamp: datetime
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class AggregatedResponse:
    """Combined response from multiple agents."""
    responses: List[AgentResponse]
    merged_content: Union[str, Dict[str, Any]]
    strategy_used: AggregationStrategy
    confidence_score: float
    metadata: Dict[str, Any]
    created_at: datetime

class ResponseAggregator:
    """Manages collection and aggregation of responses from multiple agents."""
    
    def __init__(self):
        self.response_buffer: Dict[str, List[AgentResponse]] = {}  # task_id -> responses
        self.aggregation_cache: Dict[str, AggregatedResponse] = {}  # task_id -> result
        self.validation_rules: Dict[ResponseType, List[callable]] = self._setup_validation_rules()
        
    def _setup_validation_rules(self) -> Dict[ResponseType, List[callable]]:
        """Setup validation rules for different response types."""
        def validate_text(content: str) -> bool:
            return isinstance(content, str) and bool(content.strip())
            
        def validate_structured(content: Dict) -> bool:
            return isinstance(content, dict) and bool(content)
            
        def validate_code(content: str) -> bool:
            return isinstance(content, str) and "```" in content
            
        def validate_data(content: Any) -> bool:
            try:
                if isinstance(content, str):
                    json.loads(content)
                return True
            except (json.JSONDecodeError, TypeError):
                return False
                
        return {
            ResponseType.TEXT: [validate_text],
            ResponseType.STRUCTURED: [validate_structured],
            ResponseType.CODE: [validate_code],
            ResponseType.DATA: [validate_data],
            ResponseType.ERROR: [validate_text]
        }
        
    async def add_response(
        self,
        task_id: str,
        response: AgentResponse
    ) -> None:
        """Add a new agent response to the buffer."""
        if not self._validate_response(response):
            Logger.warning(f"Invalid response from agent {response.agent_id}")
            return
            
        if task_id not in self.response_buffer:
            self.response_buffer[task_id] = []
            
        self.response_buffer[task_id].append(response)
        
    def _validate_response(self, response: AgentResponse) -> bool:
        """Validate response against type-specific rules."""
        if response.response_type not in self.validation_rules:
            return False
            
        rules = self.validation_rules[response.response_type]
        return all(rule(response.content) for rule in rules)
        
    async def aggregate_responses(
        self,
        task_id: str,
        strategy: AggregationStrategy = AggregationStrategy.SEQUENTIAL,
        weights: Optional[Dict[str, float]] = None
    ) -> AggregatedResponse:
        """Aggregate responses using specified strategy."""
        if task_id not in self.response_buffer:
            raise ValueError(f"No responses found for task {task_id}")
            
        responses = self.response_buffer[task_id]
        if not responses:
            raise ValueError("Response buffer is empty")
            
        # Select aggregation method based on strategy
        if strategy == AggregationStrategy.SEQUENTIAL:
            merged = await self._aggregate_sequential(responses)
        elif strategy == AggregationStrategy.PARALLEL:
            merged = await self._aggregate_parallel(responses)
        elif strategy == AggregationStrategy.WEIGHTED:
            merged = await self._aggregate_weighted(responses, weights or {})
        elif strategy == AggregationStrategy.VOTING:
            merged = await self._aggregate_voting(responses)
        elif strategy == AggregationStrategy.HYBRID:
            merged = await self._aggregate_hybrid(responses, weights)
        else:
            raise ValueError(f"Unsupported aggregation strategy: {strategy}")
            
        # Create aggregated response
        aggregated = AggregatedResponse(
            responses=responses,
            merged_content=merged["content"],
            strategy_used=strategy,
            confidence_score=merged["confidence"],
            metadata={
                "num_responses": len(responses),
                "response_types": [r.response_type.value for r in responses],
                "agent_ids": [r.agent_id for r in responses],
                "aggregation_details": merged.get("details", {})
            },
            created_at=datetime.utcnow()
        )
        
        # Cache the result
        self.aggregation_cache[task_id] = aggregated
        return aggregated
        
    async def _aggregate_sequential(
        self,
        responses: List[AgentResponse]
    ) -> Dict[str, Any]:
        """Combine responses in sequence."""
        merged_content = []
        total_confidence = 0.0
        
        for resp in sorted(responses, key=lambda x: x.timestamp):
            if isinstance(resp.content, str):
                merged_content.append(resp.content)
            else:
                merged_content.append(json.dumps(resp.content))
            total_confidence += resp.confidence
            
        return {
            "content": "\n".join(merged_content),
            "confidence": total_confidence / len(responses),
            "details": {"merge_type": "sequential"}
        }
        
    async def _aggregate_parallel(
        self,
        responses: List[AgentResponse]
    ) -> Dict[str, Any]:
        """Merge parallel responses by type."""
        merged = {}
        confidences = []
        
        for resp in responses:
            if resp.response_type == ResponseType.STRUCTURED:
                # Merge dictionaries
                if isinstance(resp.content, dict):
                    merged.update(resp.content)
            elif resp.response_type == ResponseType.DATA:
                # Combine data responses
                key = f"data_{len(merged)}"
                merged[key] = resp.content
            else:
                # Append text/code responses
                key = f"response_{len(merged)}"
                merged[key] = resp.content
                
            confidences.append(resp.confidence)
            
        return {
            "content": merged,
            "confidence": sum(confidences) / len(confidences),
            "details": {"merge_type": "parallel"}
        }
        
    async def _aggregate_weighted(
        self,
        responses: List[AgentResponse],
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Aggregate responses using weighted scoring."""
        weighted_responses = []
        total_weight = 0
        weighted_confidence = 0
        
        for resp in responses:
            weight = weights.get(resp.agent_id, 1.0)
            total_weight += weight
            weighted_confidence += resp.confidence * weight
            
            if isinstance(resp.content, str):
                weighted_responses.append(
                    {"content": resp.content, "weight": weight}
                )
            else:
                weighted_responses.append(
                    {"content": json.dumps(resp.content), "weight": weight}
                )
                
        # Combine weighted responses
        merged_content = "\n".join(
            f"{r['content']}" for r in sorted(
                weighted_responses,
                key=lambda x: x["weight"],
                reverse=True
            )
        )
        
        return {
            "content": merged_content,
            "confidence": weighted_confidence / total_weight,
            "details": {
                "merge_type": "weighted",
                "weights_used": weights
            }
        }
        
    async def _aggregate_voting(
        self,
        responses: List[AgentResponse]
    ) -> Dict[str, Any]:
        """Aggregate responses using voting/consensus."""
        # Count occurrences of each unique response
        vote_counts = {}
        for resp in responses:
            key = str(resp.content)
            if key not in vote_counts:
                vote_counts[key] = {
                    "count": 0,
                    "confidence_sum": 0,
                    "content": resp.content
                }
            vote_counts[key]["count"] += 1
            vote_counts[key]["confidence_sum"] += resp.confidence
            
        # Find response with most votes
        winner = max(
            vote_counts.values(),
            key=lambda x: (x["count"], x["confidence_sum"])
        )
        
        consensus_ratio = winner["count"] / len(responses)
        confidence = winner["confidence_sum"] / winner["count"]
        
        return {
            "content": winner["content"],
            "confidence": confidence * consensus_ratio,
            "details": {
                "merge_type": "voting",
                "vote_distribution": {
                    k: v["count"] for k, v in vote_counts.items()
                }
            }
        }
        
    async def _aggregate_hybrid(
        self,
        responses: List[AgentResponse],
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Combine multiple aggregation strategies based on response types."""
        # Group responses by type
        grouped = {}
        for resp in responses:
            if resp.response_type not in grouped:
                grouped[resp.response_type] = []
            grouped[resp.response_type].append(resp)
            
        merged_results = {}
        confidence_scores = []
        
        # Apply appropriate strategy for each type
        for resp_type, resps in grouped.items():
            if resp_type == ResponseType.STRUCTURED:
                result = await self._aggregate_parallel(resps)
            elif resp_type == ResponseType.TEXT:
                result = await self._aggregate_weighted(resps, weights or {})
            else:
                result = await self._aggregate_sequential(resps)
                
            merged_results[resp_type.value] = result["content"]
            confidence_scores.append(result["confidence"])
            
        return {
            "content": merged_results,
            "confidence": sum(confidence_scores) / len(confidence_scores),
            "details": {"merge_type": "hybrid"}
        }
        
    def get_formatted_response(
        self,
        task_id: str,
        format_type: str = "default"
    ) -> ConversationMessage:
        """Get formatted final response."""
        if task_id not in self.aggregation_cache:
            raise ValueError(f"No aggregated response found for task {task_id}")
            
        aggregated = self.aggregation_cache[task_id]
        
        if format_type == "default":
            if isinstance(aggregated.merged_content, str):
                formatted_content = aggregated.merged_content
            else:
                formatted_content = json.dumps(
                    aggregated.merged_content,
                    indent=2
                )
        else:
            # Add custom formatters as needed
            formatted_content = str(aggregated.merged_content)
            
        return ConversationMessage(
            role=ParticipantRole.ASSISTANT.value,
            content=[{
                'text': formatted_content,
                'metadata': {
                    'confidence': aggregated.confidence_score,
                    'strategy': aggregated.strategy_used.value,
                    'num_sources': len(aggregated.responses)
                }
            }]
        )
        
    def cleanup_task(self, task_id: str) -> None:
        """Clean up resources for a task."""
        self.response_buffer.pop(task_id, None)
        self.aggregation_cache.pop(task_id, None)
