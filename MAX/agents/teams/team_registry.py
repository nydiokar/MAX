from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from MAX.agents import Agent
from MAX.types.collaboration_types import CollaborationRole
from MAX.types.workflow_types import WorkflowStage

class TeamType(Enum):
    WORKFLOW = "workflow"         # Memory → Reasoning → Execution teams
    SPECIALIST = "specialist"     # Teams of domain experts
    COLLABORATIVE = "collaborative"  # General purpose collaborative teams
    RESEARCH = "research"         # Research and analysis focused teams
    CREATIVE = "creative"         # Creative task focused teams
    ANALYSIS = "analysis"
    IMPLEMENTATION = "implementation"
    REVIEW = "review"

@dataclass
class TeamConfiguration:
    """Configuration for a specific team instance."""
    team_type: TeamType
    max_members: int
    required_roles: List[str]
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class TeamSpec:
    """Specification for a team of agents."""
    team_id: str
    team_type: TeamType
    members: Dict[str, 'Agent']  # Use string type annotation
    created_at: datetime
    configuration: TeamConfiguration

class TeamRegistry:
    """Registry for managing team specifications and configurations."""
    
    def __init__(self):
        self.team_specs: Dict[str, TeamSpec] = {}
        self.active_teams: Dict[str, TeamConfiguration] = {}
        self.team_configurations: Dict[TeamType, TeamConfiguration] = {}
        
        # Register default team specifications
        self._register_default_teams()
        
    def _register_default_teams(self) -> None:
        """Register default team specifications."""
        # Memory → Reasoning → Execution workflow team
        self.register_team_spec(TeamSpec(
            team_id="core_workflow",
            team_type=TeamType.WORKFLOW,
            members={
                "memory_agent": CollaborationRole.CONTRIBUTOR,
                "reasoning_agent": CollaborationRole.COORDINATOR,
                "execution_agent": CollaborationRole.CONTRIBUTOR,
                "introspection_agent": CollaborationRole.VALIDATOR
            },
            created_at=datetime.now(),
            configuration=TeamConfiguration(
                team_type=TeamType.WORKFLOW,
                max_members=5,
                required_roles=["memory_agent", "reasoning_agent", "execution_agent", "introspection_agent"]
            )
        ))
        
        # Specialist research team
        self.register_team_spec(TeamSpec(
            team_id="research_team",
            team_type=TeamType.RESEARCH,
            members={
                "researcher": CollaborationRole.CONTRIBUTOR,
                "analyst": CollaborationRole.COORDINATOR,
                "validator": CollaborationRole.VALIDATOR
            },
            created_at=datetime.now(),
            configuration=TeamConfiguration(
                team_type=TeamType.RESEARCH,
                max_members=3,
                required_roles=["researcher", "analyst", "validator"]
            )
        ))
        
        # Add more default teams as needed...

    def register_team_spec(self, spec: TeamSpec) -> None:
        """Register a new team specification."""
        if spec.team_id in self.team_specs:
            raise ValueError(f"Team spec '{spec.team_id}' already exists")
        
        self.team_specs[spec.team_id] = spec

    def create_team(
        self,
        spec_name: str,
        supervisor: Agent,
        members: Dict[str, Agent]
    ) -> TeamConfiguration:
        """Create a new team instance from a specification."""
        spec = self.team_specs.get(spec_name)
        if not spec:
            raise ValueError(f"Team spec '{spec_name}' not found")
            
        # Validate team composition
        if len(members) < spec.configuration.max_members:
            raise ValueError(
                f"Team size must be between 2 and {spec.configuration.max_members}"
            )
            
        # Validate roles
        required_roles = set(spec.configuration.required_roles)
        provided_roles = {role for role in members.values()}
        if not required_roles.issubset(provided_roles):
            missing = required_roles - provided_roles
            raise ValueError(f"Missing required roles: {missing}")
            
        # Create team configuration
        team_config = TeamConfiguration(
            team_type=spec.team_type,
            max_members=len(members),
            required_roles=list(required_roles)
        )
        
        # Store active team
        team_id = f"{spec_name}_{len(self.active_teams)}"
        self.active_teams[team_id] = team_config
        
        return team_config

    def get_team_by_type(
        self,
        team_type: TeamType,
        workflow_stage: Optional[WorkflowStage] = None
    ) -> List[TeamConfiguration]:
        """Get all active teams of a specific type."""
        return [
            team for team in self.active_teams.values()
            if team.team_type == team_type
            and (not workflow_stage 
                 or (team.team_type in [TeamType.WORKFLOW, TeamType.RESEARCH] 
                     and workflow_stage in [WorkflowStage.MEMORY, WorkflowStage.REASONING]))
        ]

    def get_available_agents(
        self,
        role: CollaborationRole,
        specialties: Optional[List[str]] = None
    ) -> List[Agent]:
        """Get list of available agents for a specific role and specialties."""
        available_agents = []
        
        for team in self.active_teams.values():
            for agent in team.members.values():
                # Check if agent fits role and specialties
                if (hasattr(agent, 'collaboration_config') 
                    and role in agent.collaboration_config.supported_roles):
                    if not specialties or all(s in agent.collaboration_config.specialties 
                                            for s in specialties):
                        available_agents.append(agent)
                        
        return available_agents

    def update_team_metrics(
        self,
        team_id: str,
        metrics: Dict[str, float]
    ) -> None:
        """Update performance metrics for a team."""
        if team_id not in self.active_teams:
            raise ValueError(f"Team '{team_id}' not found")
            
        team = self.active_teams[team_id]
        if not team.metadata:
            team.metadata = {}
            
        team.metadata.update(metrics)

    def disband_team(self, team_id: str) -> None:
        """Remove a team from active teams."""
        if team_id not in self.active_teams:
            raise ValueError(f"Team '{team_id}' not found")
            
        self.active_teams.pop(team_id)
