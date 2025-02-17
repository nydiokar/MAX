from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import time
from MAX.types.workflow_types import (
    WorkflowStage,
    WorkflowStatus,
    WorkflowState,
    WorkflowTransition,
    WorkflowContext,
    IntrospectionScope
)
from MAX.utils.logger import Logger

class WorkflowManager:
    """Manages the Memory → Reasoning → Execution workflow transitions."""
    
    def __init__(self):
        self.active_workflows: Dict[str, WorkflowContext] = {}
        
        # Define stage validations
        self.stage_validations = {
            WorkflowStage.MEMORY: self._validate_memory_stage,
            WorkflowStage.REASONING: self._validate_reasoning_stage,
            WorkflowStage.EXECUTION: self._validate_execution_stage,
            WorkflowStage.INTROSPECTION: self._validate_introspection_stage,
        }

        # Introspection scopes for different stages
        self.stage_introspection = {
            WorkflowStage.MEMORY: [IntrospectionScope.AGENT_PERFORMANCE, IntrospectionScope.DECISION_QUALITY],
            WorkflowStage.REASONING: [IntrospectionScope.WORKFLOW_EFFECTIVENESS, IntrospectionScope.DECISION_QUALITY],
            WorkflowStage.EXECUTION: [IntrospectionScope.AGENT_PERFORMANCE, IntrospectionScope.ERROR_ANALYSIS],
        }
        
        # Define stage transitions
        self.valid_transitions = {
            WorkflowStage.MEMORY: [WorkflowStage.REASONING],
            WorkflowStage.REASONING: [WorkflowStage.EXECUTION, WorkflowStage.MEMORY],  # Can go back to memory if needed
            WorkflowStage.EXECUTION: [WorkflowStage.MEMORY],  # Can restart workflow if needed
        }

    async def create_workflow(
        self, 
        session_id: str,
        user_id: str,
        initial_input: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WorkflowContext:
        """Create a new workflow context."""
        initial_state = WorkflowState(
            stage=WorkflowStage.MEMORY,
            status=WorkflowStatus.PENDING,
            memory_context={
                "timestamp": datetime.utcnow().isoformat(),
                "initial_input": initial_input
            }
        )
        
        workflow = WorkflowContext(
            session_id=session_id,
            user_id=user_id,
            initial_input=initial_input,
            current_state=initial_state,
            history=[],
            metadata=metadata or {}
        )
        
        self.active_workflows[session_id] = workflow
        return workflow

    async def transition_stage(
        self,
        session_id: str,
        to_stage: WorkflowStage,
        context_update: Optional[Dict[str, Any]] = None
    ) -> WorkflowTransition:
        """Transition workflow to a new stage with validation."""
        workflow = self.active_workflows.get(session_id)
        if not workflow:
            raise ValueError(f"No active workflow found for session {session_id}")

        current_stage = workflow.current_state.stage
        
        # Validate transition is allowed
        if to_stage not in self.valid_transitions[current_stage]:
            return WorkflowTransition(
                from_stage=current_stage,
                to_stage=to_stage,
                validation_result=False,
                validation_message=f"Invalid transition from {current_stage} to {to_stage}"
            )
            
        # Validate new stage prerequisites
        validator = self.stage_validations[to_stage]
        is_valid, message = await validator(workflow, context_update)
        
        transition = WorkflowTransition(
            from_stage=current_stage,
            to_stage=to_stage,
            validation_result=is_valid,
            validation_message=message
        )
        
        if is_valid:
            # Update workflow state
            workflow.current_state.stage = to_stage
            workflow.current_state.status = WorkflowStatus.IN_PROGRESS
            
            # Update context if provided
            if context_update:
                if to_stage == WorkflowStage.MEMORY:
                    workflow.current_state.memory_context.update(context_update)
                elif to_stage == WorkflowStage.REASONING:
                    workflow.current_state.reasoning_result = context_update
                elif to_stage == WorkflowStage.EXECUTION:
                    workflow.current_state.execution_result = context_update
                    
            # Add transition to history
            workflow.history.append(transition)
            
            # Run introspection after stage transition
            await self._run_introspection(workflow, to_stage)
            
        return transition

    async def _run_introspection(
        self,
        workflow: WorkflowContext,
        current_stage: WorkflowStage
    ) -> None:
        """Run introspection analysis for the current stage."""
        if current_stage not in self.stage_introspection:
            return

        scopes = self.stage_introspection[current_stage]
        introspection_data = {}

        for scope in scopes:
            analysis = await self._analyze_scope(workflow, scope)
            introspection_data[scope.value] = analysis

        # Update workflow state with introspection results
        if not workflow.current_state.introspection_data:
            workflow.current_state.introspection_data = {}
        
        workflow.current_state.introspection_data[current_stage.value] = introspection_data

    async def _analyze_scope(
        self,
        workflow: WorkflowContext,
        scope: IntrospectionScope
    ) -> Dict[str, Any]:
        """Analyze a specific introspection scope."""
        if scope == IntrospectionScope.AGENT_PERFORMANCE:
            return self._analyze_agent_performance(workflow)
        elif scope == IntrospectionScope.WORKFLOW_EFFECTIVENESS:
            return self._analyze_workflow_effectiveness(workflow)
        elif scope == IntrospectionScope.DECISION_QUALITY:
            return self._analyze_decision_quality(workflow)
        elif scope == IntrospectionScope.ERROR_ANALYSIS:
            return self._analyze_errors(workflow)
        return {}

    def _analyze_agent_performance(self, workflow: WorkflowContext) -> Dict[str, Any]:
        """Analyze agent performance metrics."""
        if not workflow.current_state.current_agent:
            return {}

        return {
            "agent": workflow.current_state.current_agent,
            "response_time": time.time() - workflow.current_state.memory_context.get("timestamp", time.time()),
            "success_rate": self._calculate_success_rate(workflow)
        }

    def _analyze_workflow_effectiveness(self, workflow: WorkflowContext) -> Dict[str, Any]:
        """Analyze workflow effectiveness."""
        transitions = len(workflow.history) if workflow.history else 0
        return {
            "transitions_count": transitions,
            "stage_duration": time.time() - workflow.current_state.memory_context.get("timestamp", time.time()),
            "workflow_efficiency": self._calculate_workflow_efficiency(workflow)
        }

    def _analyze_decision_quality(self, workflow: WorkflowContext) -> Dict[str, Any]:
        """Analyze decision quality metrics."""
        return {
            "confidence_score": workflow.current_state.reasoning_result.get("confidence", 0.0),
            "alternative_options": len(workflow.current_state.reasoning_result.get("alternatives", [])),
            "decision_factors": workflow.current_state.reasoning_result.get("decision_factors", [])
        }

    def _analyze_errors(self, workflow: WorkflowContext) -> Dict[str, Any]:
        """Analyze any errors or issues."""
        return {
            "error_count": len([t for t in workflow.history if not t.validation_result]),
            "last_error": workflow.current_state.error_details,
            "error_patterns": self._identify_error_patterns(workflow)
        }

    def _calculate_success_rate(self, workflow: WorkflowContext) -> float:
        """Calculate success rate based on workflow history."""
        if not workflow.history:
            return 1.0
        successful = len([t for t in workflow.history if t.validation_result])
        return successful / len(workflow.history)

    def _calculate_workflow_efficiency(self, workflow: WorkflowContext) -> float:
        """Calculate workflow efficiency score."""
        if not workflow.history:
            return 1.0
        transitions = len(workflow.history)
        expected_transitions = len(WorkflowStage) - 1  # Excluding INTROSPECTION
        return min(1.0, expected_transitions / transitions)

    def _identify_error_patterns(self, workflow: WorkflowContext) -> List[str]:
        """Identify common error patterns in the workflow."""
        patterns = []
        if workflow.history:
            failed_transitions = [t for t in workflow.history if not t.validation_result]
            if failed_transitions:
                patterns.extend([t.validation_message for t in failed_transitions if t.validation_message])
        return patterns

    async def _validate_memory_stage(
        self, 
        workflow: WorkflowContext,
        context_update: Optional[Dict[str, Any]]
    ) -> tuple[bool, Optional[str]]:
        """Validate memory stage requirements."""
        if not workflow.current_state.memory_context:
            return False, "No memory context available"
        return True, None

    async def _validate_reasoning_stage(
        self,
        workflow: WorkflowContext,
        context_update: Optional[Dict[str, Any]]
    ) -> tuple[bool, Optional[str]]:
        """Validate reasoning stage requirements."""
        if not workflow.current_state.memory_context:
            return False, "Memory context required for reasoning"
        return True, None

    async def _validate_execution_stage(
        self,
        workflow: WorkflowContext,
        context_update: Optional[Dict[str, Any]]
    ) -> Tuple[bool, Optional[str]]:
        """Validate execution stage requirements."""
        if not workflow.current_state.reasoning_result:
            return False, "Reasoning result required for execution"
        return True, None

    async def _validate_introspection_stage(
        self,
        workflow: WorkflowContext,
        context_update: Optional[Dict[str, Any]]
    ) -> Tuple[bool, Optional[str]]:
        """Validate introspection stage requirements."""
        # Introspection can run at any time but needs some workflow history
        if not workflow.history:
            return False, "No workflow history available for introspection"
        
        # Check if we have enough context for meaningful introspection
        if not workflow.current_state.memory_context and \
           not workflow.current_state.reasoning_result and \
           not workflow.current_state.execution_result:
            return False, "Insufficient context for meaningful introspection"
            
        return True, None

    def get_workflow_state(self, session_id: str) -> Optional[WorkflowState]:
        """Get current state of a workflow."""
        workflow = self.active_workflows.get(session_id)
        return workflow.current_state if workflow else None

    def mark_workflow_completed(self, session_id: str) -> None:
        """Mark a workflow as completed."""
        if workflow := self.active_workflows.get(session_id):
            workflow.current_state.status = WorkflowStatus.COMPLETED
            Logger.info(f"Workflow {session_id} completed successfully")

    def mark_workflow_failed(self, session_id: str, error_details: str) -> None:
        """Mark a workflow as failed."""
        if workflow := self.active_workflows.get(session_id):
            workflow.current_state.status = WorkflowStatus.FAILED
            workflow.current_state.error_details = error_details
            Logger.error(f"Workflow {session_id} failed: {error_details}")

    def cleanup_workflow(self, session_id: str) -> None:
        """Remove workflow from active workflows."""
        self.active_workflows.pop(session_id, None)
