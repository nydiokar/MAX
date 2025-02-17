from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
from datetime import datetime
import uuid
from MAX.types.collaboration_types import SubTask, CollaborationRole
from MAX.agents import Agent
from MAX.utils.logger import Logger

@dataclass
class TaskDivisionPlan:
    """Plan for dividing a task among team members."""
    parent_task_id: str
    subtasks: List[SubTask]
    dependencies: Dict[str, Set[str]]  # subtask_id -> set of dependent subtask_ids
    estimated_duration: Dict[str, float]  # subtask_id -> estimated hours
    assignment_map: Dict[str, str]  # subtask_id -> agent_id

class TaskDivisionManager:
    """Manages task division and allocation among team members."""

    def __init__(self):
        self.active_tasks: Dict[str, TaskDivisionPlan] = {}
        self.agent_workloads: Dict[str, List[str]] = {}  # agent_id -> list of assigned subtask_ids

    async def create_task_division(
        self,
        parent_task_id: str,
        task_description: str,
        available_agents: List[Agent],
        task_type: str,
        complexity_level: int = 1
    ) -> TaskDivisionPlan:
        """Create a plan to divide a task among available agents."""
        # Create unique IDs for all subtasks
        subtask_ids = [str(uuid.uuid4()) for _ in range(self._estimate_subtask_count(complexity_level))]
        
        # Create subtasks with dependencies
        subtasks = []
        dependencies = {}
        current_time = datetime.utcnow()

        # Break down task based on type and complexity
        if task_type == "WORKFLOW":
            # Create sequential workflow subtasks
            subtasks = [
                SubTask(
                    id=subtask_ids[0],
                    parent_task_id=parent_task_id,
                    assigned_agent="",  # Will be assigned later
                    description=f"Memory/Context Phase: Analyze and gather context for {task_description}",
                    dependencies=set(),
                    status="pending",
                    created_at=current_time
                ),
                SubTask(
                    id=subtask_ids[1],
                    parent_task_id=parent_task_id,
                    assigned_agent="",
                    description=f"Reasoning Phase: Plan and make decisions for {task_description}",
                    dependencies={subtask_ids[0]},
                    status="pending",
                    created_at=current_time
                ),
                SubTask(
                    id=subtask_ids[2],
                    parent_task_id=parent_task_id,
                    assigned_agent="",
                    description=f"Execution Phase: Implement solutions for {task_description}",
                    dependencies={subtask_ids[1]},
                    status="pending",
                    created_at=current_time
                )
            ]
            # Add dependencies
            dependencies = {
                subtask_ids[1]: {subtask_ids[0]},
                subtask_ids[2]: {subtask_ids[1]}
            }

        elif task_type == "PARALLEL":
            # Create independent subtasks that can run in parallel
            chunk_size = len(task_description) // complexity_level
            for i in range(complexity_level):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < complexity_level - 1 else len(task_description)
                subtask = SubTask(
                    id=subtask_ids[i],
                    parent_task_id=parent_task_id,
                    assigned_agent="",
                    description=f"Process part {i+1}: {task_description[start_idx:end_idx]}",
                    dependencies=set(),
                    status="pending",
                    created_at=current_time
                )
                subtasks.append(subtask)

        else:  # Default to sequential breakdown
            for i in range(complexity_level):
                subtask = SubTask(
                    id=subtask_ids[i],
                    parent_task_id=parent_task_id,
                    assigned_agent="",
                    description=f"Step {i+1}: {task_description}",
                    dependencies={subtask_ids[i-1]} if i > 0 else set(),
                    status="pending",
                    created_at=current_time
                )
                subtasks.append(subtask)
                if i > 0:
                    dependencies[subtask_ids[i]] = {subtask_ids[i-1]}

        # Estimate duration for each subtask
        estimated_duration = {
            subtask.id: self._estimate_subtask_duration(subtask, complexity_level)
            for subtask in subtasks
        }

        # Create initial assignment map
        assignment_map = await self._assign_subtasks_to_agents(
            subtasks, available_agents, dependencies
        )

        # Create and store task division plan
        plan = TaskDivisionPlan(
            parent_task_id=parent_task_id,
            subtasks=subtasks,
            dependencies=dependencies,
            estimated_duration=estimated_duration,
            assignment_map=assignment_map
        )
        
        self.active_tasks[parent_task_id] = plan
        return plan

    def _estimate_subtask_count(self, complexity_level: int) -> int:
        """Estimate number of subtasks based on complexity."""
        return max(3, complexity_level * 2)  # Minimum 3 subtasks, scales with complexity

    def _estimate_subtask_duration(self, subtask: SubTask, complexity_level: int) -> float:
        """Estimate duration for a subtask in hours."""
        base_duration = 0.5  # Base duration in hours
        return base_duration * complexity_level

    async def _assign_subtasks_to_agents(
        self,
        subtasks: List[SubTask],
        available_agents: List[Agent],
        dependencies: Dict[str, Set[str]]
    ) -> Dict[str, str]:
        """Assign subtasks to available agents based on capabilities and workload."""
        assignment_map = {}
        
        # Sort agents by current workload
        agent_workloads = {
            agent.id: len(self.agent_workloads.get(agent.id, []))
            for agent in available_agents
        }
        
        # Sort subtasks by dependencies (topological sort)
        sorted_subtasks = self._topological_sort(subtasks, dependencies)
        
        for subtask in sorted_subtasks:
            # Find best agent for this subtask
            best_agent = None
            min_workload = float('inf')
            
            for agent in available_agents:
                if not hasattr(agent, 'capabilities'):
                    continue
                    
                # Check if agent can handle this subtask type
                if self._can_handle_subtask(agent, subtask):
                    current_workload = agent_workloads[agent.id]
                    if current_workload < min_workload:
                        min_workload = current_workload
                        best_agent = agent
            
            if best_agent:
                assignment_map[subtask.id] = best_agent.id
                agent_workloads[best_agent.id] += 1
                
                # Update agent's workload tracking
                if best_agent.id not in self.agent_workloads:
                    self.agent_workloads[best_agent.id] = []
                self.agent_workloads[best_agent.id].append(subtask.id)
            else:
                Logger.warning(f"No suitable agent found for subtask {subtask.id}")
        
        return assignment_map

    def _topological_sort(
        self,
        subtasks: List[SubTask],
        dependencies: Dict[str, Set[str]]
    ) -> List[SubTask]:
        """Sort subtasks based on dependencies."""
        # Create adjacency list
        adj_list = {subtask.id: set() for subtask in subtasks}
        for subtask_id, deps in dependencies.items():
            adj_list[subtask_id].update(deps)
            
        # Find tasks with no dependencies
        no_deps = [
            subtask for subtask in subtasks
            if not dependencies.get(subtask.id, set())
        ]
        
        sorted_tasks = []
        while no_deps:
            current = no_deps.pop(0)
            sorted_tasks.append(current)
            
            # Remove current task from others' dependencies
            for subtask in subtasks:
                if current.id in dependencies.get(subtask.id, set()):
                    dependencies[subtask.id].remove(current.id)
                    if not dependencies[subtask.id]:
                        no_deps.append(subtask)
                        
        return sorted_tasks

    def _can_handle_subtask(self, agent: Agent, subtask: SubTask) -> bool:
        """Check if an agent can handle a specific subtask."""
        if not hasattr(agent, 'capabilities'):
            return False
            
        # Extract required capabilities from subtask description
        required_capabilities = set()
        desc_lower = subtask.description.lower()
        
        if "memory" in desc_lower or "context" in desc_lower:
            required_capabilities.update(["memory_access", "context_analysis"])
        if "reason" in desc_lower or "plan" in desc_lower:
            required_capabilities.update(["reasoning", "planning"])
        if "execute" in desc_lower or "implement" in desc_lower:
            required_capabilities.update(["execution", "implementation"])
            
        return bool(required_capabilities & set(agent.capabilities))

    def get_subtask_status(self, parent_task_id: str) -> Dict[str, Any]:
        """Get status of all subtasks for a parent task."""
        if parent_task_id not in self.active_tasks:
            return {}
            
        plan = self.active_tasks[parent_task_id]
        return {
            "total_subtasks": len(plan.subtasks),
            "completed": sum(1 for st in plan.subtasks if st.status == "completed"),
            "in_progress": sum(1 for st in plan.subtasks if st.status == "in_progress"),
            "pending": sum(1 for st in plan.subtasks if st.status == "pending"),
            "failed": sum(1 for st in plan.subtasks if st.status == "failed"),
            "assignments": plan.assignment_map,
            "estimated_completion": self._estimate_completion_time(plan)
        }

    def _estimate_completion_time(self, plan: TaskDivisionPlan) -> float:
        """Estimate total completion time based on dependencies and durations."""
        if not plan.subtasks:
            return 0.0
            
        # Calculate completion time for each path through dependency graph
        completion_times = {}
        
        def calculate_path_time(subtask_id: str, visited: Set[str]) -> float:
            if subtask_id in completion_times:
                return completion_times[subtask_id]
                
            if subtask_id in visited:
                return 0.0  # Handle cycles
                
            visited.add(subtask_id)
            
            # Get dependencies completion time
            deps_time = 0.0
            if subtask_id in plan.dependencies:
                for dep_id in plan.dependencies[subtask_id]:
                    deps_time = max(deps_time, calculate_path_time(dep_id, visited))
                    
            # Add own duration
            total_time = deps_time + plan.estimated_duration.get(subtask_id, 0.0)
            completion_times[subtask_id] = total_time
            
            return total_time
            
        # Calculate for all terminal tasks (those that no other tasks depend on)
        terminal_tasks = {st.id for st in plan.subtasks} - {
            dep_id
            for deps in plan.dependencies.values()
            for dep_id in deps
        }
        
        max_completion_time = 0.0
        for task_id in terminal_tasks:
            completion_time = calculate_path_time(task_id, set())
            max_completion_time = max(max_completion_time, completion_time)
            
        return max_completion_time
