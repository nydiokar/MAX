import asyncio
from typing import Dict, Any, AsyncIterable, Optional, Union, List
from dataclasses import dataclass, fields, asdict, replace
import time
from MAX.types.workflow_types import WorkflowStage
from MAX.utils.logger import Logger
from MAX.types.base_types import (
    ConversationMessage,
    ParticipantRole,
    AgentMessage,
    MessageType,
)
from MAX.classifiers import (
    Classifier,
    ClassifierResult,
    AnthropicClassifier,
    AnthropicClassifierOptions,
)
from MAX.agents import (
    Agent,
    AgentResponse,
    AgentProcessingResult,
    TaskExpertAgent,
    TaskExpertOptions,
    SupervisorAgent,
    SupervisorAgentOptions,
)
from MAX.storage import ChatStorage, InMemoryChatStorage
from MAX.config.database_config import DatabaseConfig
from MAX.managers.system_state_manager import StateManager
from MAX.config.orchestrator_config import OrchestratorConfig

@dataclass
class MultiAgentOrchestrator:
    def __init__(
        self,
        options: Optional[OrchestratorConfig] = None,
        storage: Optional[ChatStorage] = None,
        classifier: Optional[Classifier] = None,
        logger: Optional[Logger] = None,
        use_local_classifier: bool = False,
    ):
        # Initialize database configuration
        self.db_config = DatabaseConfig()

        # Initialize managers and supervisor
        self.state_manager = StateManager(self.db_config)
        self.supervisor = SupervisorAgent(
            options=SupervisorAgentOptions(
                lead_agent=TaskExpertAgent(
                    options=TaskExpertOptions(
                        name="SUPERVISOR",
                        streaming=True,
                        description="Team coordinator and workflow manager"
                    )
                ),
                storage=storage or InMemoryChatStorage()
            )
        )

        DEFAULT_CONFIG = OrchestratorConfig()
        ################# CREATE CLASSIFIER AND CHECK THE CODE BELOW IT - TaskExpertOptions seems suspicious  ##################

        # if use_local_classifier:
        #     from MAX.adapters.llm import OllamaProvider
        #     self.classifier = create_llm_provider(
        #        "ollama",
        #        options=OllamaProviderOptions(
        #           name="intent_classifier",
        #          model_id="llama3.1:8b-instruct-q8_0",
        #         description="Local intent classifier"
        #    )
        # )
        # else:
        #    self.classifier = classifier or AnthropicClassifier(
        #       options=AnthropicClassifierOptions()
        #  )

        if options is None:
            options = {}
        if isinstance(options, dict):
            valid_keys = {f.name for f in fields(OrchestratorConfig)}
            options = {k: v for k, v in options.items() if k in valid_keys}
            options = OrchestratorConfig(**options)
        elif not isinstance(options, OrchestratorConfig):
            raise ValueError(
                "options must be a dictionary or an OrchestratorConfig instance"
            )

        self.config = replace(DEFAULT_CONFIG, **asdict(options))
        self.storage = storage or InMemoryChatStorage()
        self.logger = Logger(self.config, logger)
        self.agents: Dict[str, Agent] = {}
        self.classifier: Classifier = classifier or AnthropicClassifier(
            options=AnthropicClassifierOptions()
        )
        self.execution_times: Dict[str, float] = {}
        self.default_agent: Agent = TaskExpertAgent(
            options=TaskExpertOptions(
                name="DEFAULT",
                streaming=True,
                description="A knowledgeable generalist capable of addressing a wide range of topics.",
            )
        )

        # Initialize system state at startup
        asyncio.create_task(self._initialize_system())

        # Add message queue
        self.message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self.pending_responses: Dict[str, asyncio.Future] = {}
        self.message_processor_task = asyncio.create_task(
            self._process_message_queue()
        )

    async def _initialize_system(self):
        """Initialize system state and restore previous state if available"""
        try:
            await self.state_manager.restore_state_from_storage()
            self.logger.info("System state restored successfully")
        except Exception as e:
            self.logger.error(f"Failed to restore system state: {str(e)}")

    def add_agent(self, agent: Agent):
        if agent.id in self.agents:
            raise ValueError(f"An agent with ID '{agent.id}' already exists.")
        self.agents[agent.id] = agent
        self.classifier.set_agents(self.agents)

    def get_default_agent(self) -> Agent:
        return self.default_agent

    def set_default_agent(self, agent: Agent):
        self.default_agent = agent

    def set_classifier(self, intent_classifier: Classifier):
        self.classifier = intent_classifier

    def get_all_agents(self) -> Dict[str, Dict[str, str]]:
        return {
            key: {"name": agent.name, "description": agent.description}
            for key, agent in self.agents.items()
        }

    async def dispatch_to_agent(
        self, params: Dict[str, Any]
    ) -> Union[ConversationMessage, AsyncIterable[Any]]:
        user_input = params["user_input"]
        user_id = params["user_id"]
        session_id = params["session_id"]
        classifier_result: ClassifierResult = params["classifier_result"]
        additional_params = params.get("additional_params", {})

        if not classifier_result.selected_agent:
            return "I'm sorry, but I need more information to understand your request. Could you please be more specific?"

        selected_agent = classifier_result.selected_agent
        agent_chat_history = await self.storage.fetch_chat(
            user_id, session_id, selected_agent.id
        )

        self.logger.print_chat_history(agent_chat_history, selected_agent.id)

        response = await self.measure_execution_time(
            f"Agent {selected_agent.name} | Processing request",
            lambda: selected_agent.process_request(
                user_input,
                user_id,
                session_id,
                agent_chat_history,
                additional_params,
            ),
        )

        return response

    async def route_request(
        self,
        user_input: str,
        user_id: str,
        session_id: str,
        additional_params: Dict[str, str] = {},
    ) -> AgentResponse:
        self.execution_times.clear()

        try:
            # Track conversation state
            await self.state_manager.track_conversation_state(
                user_id=user_id,
                session_id=session_id,
                message=ConversationMessage(
                    role=ParticipantRole.USER.value, content=user_input
                ),
                metadata=additional_params,
            )

            # Memory Stage: Fetch context
            chat_history = await self.storage.fetch_all_chats(user_id, session_id) or []
            
            # Activate memory team through supervisor
            await self.supervisor.activate_team(
                team_type="MEMORY",
                task_description=user_input,
                workflow_stage=WorkflowStage.MEMORY.value
            )
            
            # Send memory stage data to team
            await self.supervisor.send_team_message(
                content=str({
                    "user_input": user_input,
                    "chat_history": [msg.content[0].get('text', '') for msg in chat_history],
                    "stage": WorkflowStage.MEMORY.value
                })
            )

            # Reasoning Stage: Classify intent and select agent
            classifier_result: ClassifierResult = (
                await self.measure_execution_time(
                    "Classifying user intent",
                    lambda: self.classifier.classify(user_input, chat_history),
                )
            )

            if self.config.LOG_CLASSIFIER_OUTPUT:
                self.print_intent(user_input, classifier_result)

            # Activate reasoning team through supervisor
            await self.supervisor.activate_team(
                team_type="REASONING",
                task_description=user_input,
                workflow_stage=WorkflowStage.REASONING.value
            )

            # Send reasoning stage data to team
            await self.supervisor.send_team_message(
                content=str({
                    "intents": classifier_result.intents,
                    "selected_agent": classifier_result.selected_agent.name if classifier_result.selected_agent else None,
                    "confidence": classifier_result.confidence,
                    "stage": WorkflowStage.REASONING.value
                })
            )

            # Handle agent selection failures
            if not classifier_result.selected_agent:
                return await self._handle_agent_selection_failure(
                    classifier_result, 
                    user_input,
                    user_id,
                    session_id,
                    additional_params
                )
                
            # Validate agent before execution
            if not await self._validate_agent(classifier_result.selected_agent):
                await self.supervisor.update_task_status(
                    subtask_id=session_id,  # Using session_id as task identifier
                    status="failed",
                    result={"reason": f"Agent {classifier_result.selected_agent.name} failed validation"}
                )
                self.logger.error(f"Selected agent {classifier_result.selected_agent.name} failed validation")
                return AgentResponse(
                    metadata=self.create_metadata(
                        classifier_result,
                        user_input,
                        user_id,
                        session_id,
                        additional_params,
                    ),
                    output=ConversationMessage(
                        role=ParticipantRole.ASSISTANT.value,
                        content="The selected agent is currently unavailable. Please try again later.",
                    ),
                    streaming=False,
                )

            try:
                agent_response = await self.dispatch_to_agent(
                    {
                        "user_input": user_input,
                        "user_id": user_id,
                        "session_id": session_id,
                        "classifier_result": classifier_result,
                        "additional_params": additional_params,
                    }
                )

                metadata = self.create_metadata(
                    classifier_result,
                    user_input,
                    user_id,
                    session_id,
                    additional_params,
                )

                # Save question
                await self.save_message(
                    ConversationMessage(
                        role=ParticipantRole.USER.value, content=user_input
                    ),
                    user_id,
                    session_id,
                    classifier_result.selected_agent,
                )

                # Track response state if it's a conversation message
                if isinstance(agent_response, ConversationMessage):
                    await self.state_manager.track_conversation_state(
                        user_id=user_id,
                        session_id=session_id,
                        message=agent_response,
                    )
                    # Save the response
                    await self.save_message(
                        agent_response,
                        user_id,
                        session_id,
                        classifier_result.selected_agent,
                    )

                # Activate execution team and send results
                await self.supervisor.activate_team(
                    team_type="EXECUTION",
                    task_description=user_input,
                    workflow_stage=WorkflowStage.EXECUTION.value
                )
                
                # Send execution results to team
                await self.supervisor.send_team_message(
                    content=str({
                        "result": agent_response.content[0].get('text', '') if isinstance(agent_response, ConversationMessage) else str(agent_response),
                        "agent": classifier_result.selected_agent.name,
                        "stage": WorkflowStage.EXECUTION.value
                    })
                )

                # Mark task as completed
                await self.supervisor.update_task_status(
                    subtask_id=session_id,
                    status="completed",
                    result={"response": agent_response}
                )

                return AgentResponse(
                    metadata=metadata,
                    output=agent_response,
                    streaming=classifier_result.selected_agent.is_streaming_enabled(),
                )

            except Exception as error:
                self.logger.error(
                    f"Error during agent dispatch or processing: {str(error)}"
                )
                return AgentResponse(
                    metadata=self.create_metadata(
                        classifier_result,
                        user_input,
                        user_id,
                        session_id,
                        additional_params,
                    ),
                    output=(
                        self.config.GENERAL_ROUTING_ERROR_MSG_MESSAGE
                        if self.config.GENERAL_ROUTING_ERROR_MSG_MESSAGE
                        else str(error)
                    ),
                    streaming=False,
                )

        except Exception as error:
            self.logger.error(
                f"Error during intent classification: {str(error)}"
            )
            return AgentResponse(
                metadata=self.create_metadata(
                    None, user_input, user_id, session_id, additional_params
                ),
                output=(
                    self.config.CLASSIFICATION_ERROR_MESSAGE
                    if self.config.CLASSIFICATION_ERROR_MESSAGE
                    else str(error)
                ),
                streaming=False,
            )

        finally:
            self.logger.print_execution_times(self.execution_times)

    def print_intent(
        self, user_input: str, intent_classifier_result: ClassifierResult
    ) -> None:
        """Print the classified intent."""
        self.logger.log_header("Classified Intent")
        self.logger.info(f"> Text: {user_input}")
        selected_agent_string = (
            intent_classifier_result.selected_agent.name
            if intent_classifier_result.selected_agent
            else "No agent selected"
        )
        self.logger.info(f"> Selected Agent: {selected_agent_string}")
        self.logger.info(
            f"> Confidence: {intent_classifier_result.confidence:.2f}"
        )
        self.logger.info("")

    async def measure_execution_time(self, timer_name: str, fn):
        if not self.config.LOG_EXECUTION_TIMES:
            return await fn()

        start_time = time.time()
        self.execution_times[timer_name] = start_time

        try:
            result = await fn()
            end_time = time.time()
            duration = end_time - start_time
            self.execution_times[timer_name] = duration
            return result
        except Exception as error:
            end_time = time.time()
            duration = end_time - start_time
            self.execution_times[timer_name] = duration
            raise error

    def create_metadata(
        self,
        intent_classifier_result: Optional[ClassifierResult],
        user_input: str,
        user_id: str,
        session_id: str,
        additional_params: Dict[str, str],
    ) -> AgentProcessingResult:
        base_metadata = AgentProcessingResult(
            user_input=user_input,
            agent_id="no_agent_selected",
            agent_name="No Agent",
            user_id=user_id,
            session_id=session_id,
            additional_params=additional_params,
        )

        if (
            not intent_classifier_result
            or not intent_classifier_result.selected_agent
        ):
            base_metadata.additional_params["error_type"] = (
                "classification_failed"
            )
            if intent_classifier_result and intent_classifier_result.fallback_reason:
                base_metadata.additional_params["failure_reason"] = intent_classifier_result.fallback_reason
            if intent_classifier_result and intent_classifier_result.intents:
                base_metadata.additional_params["detected_intents"] = ",".join(intent_classifier_result.intents)
        else:
            base_metadata.agent_id = intent_classifier_result.selected_agent.id
            base_metadata.agent_name = intent_classifier_result.selected_agent.name
            if intent_classifier_result.intents:
                base_metadata.additional_params["detected_intents"] = ",".join(intent_classifier_result.intents)

        return base_metadata

    async def _handle_agent_selection_failure(
        self,
        classifier_result: ClassifierResult,
        user_input: str,
        user_id: str,
        session_id: str,
        additional_params: Dict[str, Any]
    ) -> AgentResponse:
        """Handle agent selection failures with workflow awareness."""
        # Try to get best agent match based on intents
        if classifier_result.intents:
            best_agent = self._find_best_agent_for_intents(classifier_result.intents)
            if best_agent:
                classifier_result.selected_agent = best_agent
                classifier_result.confidence = 0.7  # Conservative confidence for fallback
                self.logger.info(f"Found alternate agent {best_agent.name} based on intents")

                # Update team with fallback selection
                await self.supervisor.send_team_message(
                    content=str({
                        "fallback_agent": best_agent.name,
                        "fallback_reason": "Intent-based fallback selection",
                        "confidence": 0.7,
                        "stage": WorkflowStage.REASONING.value
                    })
                )
                return await self.dispatch_to_agent({
                    "user_input": user_input,
                    "user_id": user_id,
                    "session_id": session_id,
                    "classifier_result": classifier_result,
                    "additional_params": additional_params,
                })

        # Use default agent as last resort if configured
        if self.config.USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED:
            classifier_result = self.get_fallback_result()
            self.logger.info(f"Using default agent. Detected intents: {classifier_result.intents}")
            
            # Update team with default agent selection
            await self.supervisor.send_team_message(
                content=str({
                    "fallback_agent": "DEFAULT",
                    "fallback_reason": "Using default agent",
                    "confidence": 0.5,
                    "stage": WorkflowStage.REASONING.value
                })
            )
            return await self.dispatch_to_agent({
                "user_input": user_input,
                "user_id": user_id,
                "session_id": session_id,
                "classifier_result": classifier_result,
                "additional_params": additional_params,
            })

        # No suitable agent found
        await self.supervisor.update_task_status(
            subtask_id=session_id,
            status="failed",
            result={"reason": "No suitable agent found and no default agent configured"}
        )

        return AgentResponse(
            metadata=self.create_metadata(
                classifier_result,
                user_input,
                user_id,
                session_id,
                additional_params,
            ),
            output=ConversationMessage(
                role=ParticipantRole.ASSISTANT.value,
                content=(
                    f"{self.config.NO_SELECTED_AGENT_MESSAGE}\n"
                    f"Detected intents: {classifier_result.intents or []}"
                ),
            ),
            streaming=False,
        )

    def get_fallback_result(self) -> ClassifierResult:
        return ClassifierResult(
            selected_agent=self.get_default_agent(),
            confidence=0,
            intents=["fallback_required"],
            fallback_reason="Using default agent as fallback"
        )

    async def save_message(
        self,
        message: ConversationMessage,
        user_id: str,
        session_id: str,
        agent: Agent,
    ):
        if agent and agent.save_chat:
            return await self.storage.save_chat_message(
                user_id=user_id,
                session_id=session_id,
                agent_id=agent.id,
                new_message=message,
                max_history_size=self.config.MAX_MESSAGE_PAIRS_PER_AGENT,
            )

    # Add new message queue processing methods
    async def _process_message_queue(self):
        while True:
            try:
                message = await self.message_queue.get()
                if message.target_agent not in self.agents:
                    self.logger.error(
                        f"Unknown target agent: {message.target_agent}"
                    )
                    continue

                target_agent = self.agents[message.target_agent]
                response = await self._dispatch_to_agent(target_agent, message)

                # Handle response
                if message.correlation_id in self.pending_responses:
                    future = self.pending_responses[message.correlation_id]
                    if not future.done():
                        future.set_result(response)

                self.message_queue.task_done()

            except Exception as e:
                self.logger.error(f"Error processing message: {str(e)}")
                continue

    def _find_best_agent_for_intents(self, intents: List[str]) -> Optional[Agent]:
        """
        Find best matching agent based on detected intents and query analysis.
        Used as an intelligent fallback when primary agent selection fails.
        """
        best_score = 0
        best_agent = None

        for agent in self.agents.values():
            score = 0
            
            # Check agent availability first
            if not self._is_agent_available(agent):
                continue

            # Check capabilities match
            if hasattr(agent, 'capabilities'):
                matching_caps = [cap for cap in agent.capabilities if any(i in cap for i in intents)]
                if matching_caps:
                    score += len(matching_caps) / len(agent.capabilities)

            # Check specializations match
            if hasattr(agent, 'specializations'):
                matching_specs = [spec for spec in agent.specializations if any(i in spec for i in intents)]
                if matching_specs:
                    score += len(matching_specs)

            # Consider agent's historical performance
            if hasattr(agent, 'success_rate'):
                score *= (1 + agent.success_rate)

            # Consider current workload
            if hasattr(agent, 'current_tasks'):
                workload_penalty = len(agent.current_tasks) * 0.1
                score = max(0, score - workload_penalty)

            # Consider agent's response time history
            if hasattr(agent, 'avg_response_time'):
                time_score = 1.0 / (1.0 + agent.avg_response_time)  # Normalize to 0-1
                score *= (1 + time_score)

            # Consider agent's error rate
            if hasattr(agent, 'error_rate'):
                error_penalty = agent.error_rate * 0.5
                score = max(0, score - error_penalty)

            if score > best_score:
                best_score = score
                best_agent = agent

        # Require higher minimum match quality
        return best_agent if best_score > 0.4 else None

    def _is_agent_available(self, agent: Agent) -> bool:
        """
        Check if an agent is currently available and healthy.
        """
        if not agent:
            return False

        try:
            # Check if agent is initialized
            if hasattr(agent, 'is_initialized') and not agent.is_initialized:
                return False

            # Check if agent is at max capacity
            if hasattr(agent, 'current_tasks'):
                if len(agent.current_tasks) >= agent.max_concurrent_tasks:
                    return False

            # Check agent's health status
            if hasattr(agent, 'health_status'):
                if agent.health_status != 'healthy':
                    return False

            # Check if agent is in maintenance mode
            if hasattr(agent, 'maintenance_mode'):
                if agent.maintenance_mode:
                    return False

            # Check if agent has required resources
            if hasattr(agent, 'check_resources'):
                if not agent.check_resources():
                    return False

            return True

        except Exception as e:
            Logger.error(f"Error checking agent availability: {str(e)}")
            return False

    async def _validate_agent(self, agent: Agent) -> bool:
        """
        Validate agent availability and readiness.
        """
        if not agent:
            return False

        try:
            # Basic interface check
            if not hasattr(agent, 'process_request') or not callable(getattr(agent, 'process_request')):
                self.logger.error(f"Agent {agent.name} lacks required interface")
                return False

            # Check initialization
            if hasattr(agent, 'is_initialized') and not agent.is_initialized:
                self.logger.error(f"Agent {agent.name} not properly initialized")
                return False

            # Check state if available
            if hasattr(agent, 'get_state'):
                state = await agent.get_state()
                if state.get('status') == 'busy':
                    self.logger.warning(f"Agent {agent.name} is busy")
                    return False
                if state.get('health') == 'unhealthy':
                    self.logger.error(f"Agent {agent.name} reported unhealthy state")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating agent {agent.name}: {str(e)}")
            return False

    async def shutdown(self):
        self.message_processor_task.cancel()
        try:
            await self.message_processor_task
        except asyncio.CancelledError:
            pass

        # Additional cleanup
        self.pending_responses.clear()
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
