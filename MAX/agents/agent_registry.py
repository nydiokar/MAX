from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from MAX.agents import Agent
from MAX.utils.logger import Logger

logger = Logger.get_logger()

@dataclass
class AgentMetadata:
    """Metadata for an agent"""
    name: str
    description: str
    capabilities: List[str] = field(default_factory=list)
    specializations: List[str] = field(default_factory=list)
    max_concurrent_tasks: int = 5
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    health_status: str = "healthy"
    maintenance_mode: bool = False
    current_tasks: List[str] = field(default_factory=list)
    success_rate: float = 1.0
    error_rate: float = 0.0
    avg_response_time: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

class AgentRegistry:
    """Manages agent registration, metadata, and availability"""
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.metadata: Dict[str, AgentMetadata] = {}
        self.logger = logger

    def register_agent(
        self,
        agent: Agent,
        capabilities: Optional[List[str]] = None,
        specializations: Optional[List[str]] = None,
        max_concurrent_tasks: int = 5
    ) -> bool:
        """Register a new agent with metadata"""
        try:
            if agent.id in self.agents:
                self.logger.warning(f"Agent {agent.id} is already registered")
                return False

            # Create metadata
            metadata = AgentMetadata(
                name=agent.name,
                description=agent.description,
                capabilities=capabilities or [],
                specializations=specializations or [],
                max_concurrent_tasks=max_concurrent_tasks
            )

            # Store agent and metadata
            self.agents[agent.id] = agent
            self.metadata[agent.id] = metadata
            self.logger.info(f"Successfully registered agent {agent.id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to register agent {agent.id}: {str(e)}")
            return False

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent"""
        try:
            if agent_id not in self.agents:
                self.logger.warning(f"Agent {agent_id} not found")
                return False

            del self.agents[agent_id]
            del self.metadata[agent_id]
            self.logger.info(f"Successfully unregistered agent {agent_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to unregister agent {agent_id}: {str(e)}")
            return False

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID"""
        return self.agents.get(agent_id)

    def get_metadata(self, agent_id: str) -> Optional[AgentMetadata]:
        """Get agent metadata by ID"""
        return self.metadata.get(agent_id)

    def update_agent_status(
        self,
        agent_id: str,
        health_status: Optional[str] = None,
        maintenance_mode: Optional[bool] = None
    ) -> bool:
        """Update agent status"""
        try:
            if agent_id not in self.metadata:
                self.logger.warning(f"Agent {agent_id} not found")
                return False

            metadata = self.metadata[agent_id]
            if health_status:
                metadata.health_status = health_status
            if maintenance_mode is not None:
                metadata.maintenance_mode = maintenance_mode
            metadata.last_active = datetime.now()

            self.logger.info(f"Updated status for agent {agent_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to update agent {agent_id} status: {str(e)}")
            return False

    def track_request(
        self,
        agent_id: str,
        success: bool,
        response_time: float
    ) -> None:
        """Track agent request performance"""
        try:
            if agent_id not in self.metadata:
                return

            metadata = self.metadata[agent_id]
            metadata.total_requests += 1
            metadata.avg_response_time = (
                (metadata.avg_response_time * (metadata.total_requests - 1) + response_time)
                / metadata.total_requests
            )

            if success:
                metadata.successful_requests += 1
                metadata.success_rate = metadata.successful_requests / metadata.total_requests
            else:
                metadata.failed_requests += 1
                metadata.error_rate = metadata.failed_requests / metadata.total_requests

        except Exception as e:
            self.logger.error(f"Failed to track request for agent {agent_id}: {str(e)}")

    def get_available_agents(self) -> List[str]:
        """Get list of available agent IDs"""
        return [
            agent_id for agent_id, metadata in self.metadata.items()
            if metadata.health_status == "healthy" and not metadata.maintenance_mode
        ]

    def is_agent_available(self, agent_id: str) -> bool:
        """Check if an agent is available"""
        metadata = self.metadata.get(agent_id)
        if not metadata:
            return False

        return (
            metadata.health_status == "healthy"
            and not metadata.maintenance_mode
            and len(metadata.current_tasks) < metadata.max_concurrent_tasks
        )

    def add_task(self, agent_id: str, task_id: str) -> bool:
        """Add a task to an agent's current tasks"""
        try:
            if agent_id not in self.metadata:
                return False

            metadata = self.metadata[agent_id]
            if len(metadata.current_tasks) >= metadata.max_concurrent_tasks:
                return False

            metadata.current_tasks.append(task_id)
            return True

        except Exception as e:
            self.logger.error(f"Failed to add task to agent {agent_id}: {str(e)}")
            return False

    def remove_task(self, agent_id: str, task_id: str) -> bool:
        """Remove a task from an agent's current tasks"""
        try:
            if agent_id not in self.metadata:
                return False

            metadata = self.metadata[agent_id]
            if task_id in metadata.current_tasks:
                metadata.current_tasks.remove(task_id)
                return True
            return False

        except Exception as e:
            self.logger.error(f"Failed to remove task from agent {agent_id}: {str(e)}")
            return False

    def find_agents_by_capability(self, capability: str) -> List[str]:
        """Find agents that have a specific capability"""
        return [
            agent_id for agent_id, metadata in self.metadata.items()
            if capability in metadata.capabilities
        ]

    def find_agents_by_specialization(self, specialization: str) -> List[str]:
        """Find agents that have a specific specialization"""
        return [
            agent_id for agent_id, metadata in self.metadata.items()
            if specialization in metadata.specializations
        ]

    def check_agent_health(self, agent_id: str) -> Dict[str, Any]:
        """Get detailed health metrics for an agent"""
        metadata = self.metadata.get(agent_id)
        if not metadata:
            return {}
        
        return {
            "health_status": metadata.health_status,
            "maintenance_mode": metadata.maintenance_mode,
            "current_tasks": len(metadata.current_tasks),
            "max_tasks": metadata.max_concurrent_tasks,
            "success_rate": metadata.success_rate,
            "error_rate": metadata.error_rate,
            "avg_response_time": metadata.avg_response_time,
            "uptime": (datetime.now() - metadata.created_at).total_seconds(),
            "last_active": metadata.last_active
        } 