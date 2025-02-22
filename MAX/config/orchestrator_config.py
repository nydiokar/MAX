from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator"""
    max_turns: int = 10
    timeout_seconds: int = 30
    fallback_agent: str = "default"
    enable_memory: bool = True
    memory_window: int = 5
    
    # Additional orchestrator-specific settings
    parallel_execution: bool = False
    max_retries: int = 3
    error_threshold: float = 0.8
    logging_level: str = "INFO"
    
    # Resource management
    max_concurrent_agents: int = 5
    agent_timeout: int = 15
    
    # Memory settings
    memory_decay_rate: float = 0.1
    min_confidence_threshold: float = 0.7
    context_window_size: int = 1000 