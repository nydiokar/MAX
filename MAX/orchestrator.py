import asyncio
import time
from typing import Dict, Any, AsyncIterable, Optional, Union, List
from dataclasses import dataclass, fields, asdict, replace


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
    AgentOptions,
)
from MAX.storage import ChatStorage, InMemoryChatStorage
from MAX.config.database_config import DatabaseConfig
from MAX.managers.system_state_manager import StateManager
from MAX.config.orchestrator_config import OrchestratorConfig
from MAX.agents.agent_registry import AgentRegistry, AgentMetadata
from MAX.managers.memory_manager import MemoryManager
from MAX.agents.default_agent import DefaultAgent
from MAX.agents.recursive_thinker import RecursiveThinker


@dataclass
class MultiAgentOrchestrator:
    """
    Orchestrator for managing multiple agents, handling intent classification,
    routing requests, and coordinating a workflow that includes memory, reasoning,
    and execution stages.
    """

    def __init__(
        self,
        options: Optional[OrchestratorConfig] = None,
        storage: Optional[ChatStorage] = None,
        classifier: Optional[Classifier] = None,
        logger: Optional[Logger] = None,
        use_local_classifier: bool = False,
    ):
        """
        Initialize the Orchestrator with optional configuration, storage, classifier,
        and logger. If none are provided, default instances/configurations will be used.
        """
        # 1. Initialize essential components
        self.db_config = DatabaseConfig()
        self.state_manager = StateManager(self.db_config)
        self.storage = storage or InMemoryChatStorage()

        # 2. Agent Registry
        self.agent_registry = AgentRegistry()

        # 4. Configuration
        self.config = self._init_config(options)
        self.logger = Logger(self.config, logger)

        # 5. Classifier
        self.classifier: Classifier = classifier or AnthropicClassifier(
            options=AnthropicClassifierOptions()
        )
        # If you want to enable a local classifier, uncomment and adjust the following block:
        #
        # if use_local_classifier:
        #     from MAX.adapters.llm import OllamaProvider
        #     self.classifier = create_llm_provider(
        #         "ollama",
        #         options=OllamaProviderOptions(
        #             name="intent_classifier",
        #             model_id="llama3.1:8b-instruct-q8_0",
        #             description="Local intent classifier"
        #         )
        #     )

        # 6. Register Agents
        # Default Agent
        self.default_agent = DefaultAgent(
            options=AgentOptions(
                name="DEFAULT",
                description="A responsive general-purpose agent for handling default cases.",
                save_chat=True,
                resources=None
            )
        )
        
        # Recursive Thinker Agent
        self.recursive_thinker = RecursiveThinker(
            options=AgentOptions(
                name="RECURSIVE_THINKER",
                description="An agent capable of deep reasoning and recursive problem-solving.",
                save_chat=True,
                resources=None
            )
        )

        # Register both agents
        self.agent_registry.register_agent(
            self.default_agent,
            capabilities=["general", "fallback"],
            specializations=["general_knowledge"],
            max_concurrent_tasks=10
        )

        self.agent_registry.register_agent(
            self.recursive_thinker,
            capabilities=["reasoning", "problem_solving"],
            specializations=["recursive_thinking", "deep_reasoning"],
            max_concurrent_tasks=5
        )

        # Set the default agent as the primary fallback
        self.default_agent = self.default_agent

        # 7. Execution Times and Additional Structures
        self.execution_times: Dict[str, float] = {}
        self.agents: Dict[str, Agent] = {}
        self.message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self.pending_responses: Dict[str, asyncio.Future] = {}

        # 8. Initialize system state and message processor
        asyncio.create_task(self._initialize_system())
        self.message_processor_task = asyncio.create_task(self._process_message_queue())

        # Initialize memory manager
        self.memory_manager = None
        if self.config.MEMORY_ENABLED:
            memory_config = {
                "vector_store": {
                    "provider": "chroma",
                    "config": {
                        "collection_name": self.config.MEMORY_COLLECTION_NAME,
                        "path": self.config.MEMORY_STORAGE_PATH
                    }
                }
            }
            self.memory_manager = MemoryManager(config=memory_config)

    def _init_config(self, options: Optional[Any]) -> OrchestratorConfig:
        """
        Safely initialize or merge an OrchestratorConfig object.
        """
        DEFAULT_CONFIG = OrchestratorConfig()
        
        # First validate the type
        if not (options is None or isinstance(options, (dict, OrchestratorConfig))):
            raise ValueError("options must be a dictionary or an OrchestratorConfig instance")
    
        if options is None:
            return DEFAULT_CONFIG
        elif isinstance(options, dict):
            valid_keys = {f.name for f in fields(OrchestratorConfig)}
            filtered_opts = {k: v for k, v in options.items() if k in valid_keys}
            return replace(DEFAULT_CONFIG, **filtered_opts)
        elif isinstance(options, OrchestratorConfig):
            return replace(DEFAULT_CONFIG, **asdict(options))
        else:
            raise ValueError("Unexpected type for options") # type: ignore[unreachable]

    async def _initialize_system(self):
        """
        Initialize system state and restore previous state from storage if available.
        """
        try:
            await self.state_manager.restore_state_from_storage()
            self.logger.info("System state restored successfully")
        except Exception as e:
            self.logger.error(f"Failed to restore system state: {str(e)}")

    # ------------------- Public API for Agent Management ------------------- #

    def add_agent(self, agent: Agent, capabilities: Optional[List[str]] = None,
                  specializations: Optional[List[str]] = None) -> bool:
        """
        Register a new agent with the orchestrator's AgentRegistry.
        """
        if self.agent_registry.register_agent(agent, capabilities, specializations):
            self.classifier.set_agents(self.agent_registry.agents)
            return True
        return False

    def get_default_agent(self) -> Agent:
        """
        Retrieve the currently configured default (fallback) agent.
        """
        return self.default_agent

    def set_default_agent(self, agent: Agent) -> bool:
        """
        Set a new default agent to handle general or fallback requests.
        """
        if self.agent_registry.register_agent(agent, capabilities=["general", "fallback"]):
            self.default_agent = agent
            return True
        return False

    def set_classifier(self, intent_classifier: Classifier):
        """
        Manually override the orchestrator's classifier.
        """
        self.classifier = intent_classifier

    def get_all_agents(self) -> Dict[str, Dict[str, str]]:
        """
        Retrieve a dictionary of all registered agents, along with their metadata.
        """
        return {
            agent_id: {
                "name": metadata.name,
                "description": metadata.description,
                "capabilities": metadata.capabilities,
                "specializations": metadata.specializations,
                "health_status": metadata.health_status,
                "current_tasks": len(metadata.current_tasks),
            }
            for agent_id, metadata in self.agent_registry.metadata.items()
        }

    # ------------------- Main Request Handling ------------------- #

    async def route_request(
        self,
        user_input: str,
        user_id: str,
        session_id: str,
        additional_params: Dict[str, str] = None,
    ) -> AgentResponse:
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

            # --- Memory Stage ---
            chat_history = await self.storage.fetch_all_chats(user_id, session_id) or []
            if self.memory_manager:
                # Get relevant context from memory
                context = await self.memory_manager.get_relevant_context(
                    query=user_input,
                    agent_id=classifier_result.selected_agent.id,
                    session_id=session_id
                )
                
                # Store user message in memory
                await self.memory_manager.store_message(
                    message=ConversationMessage(
                        role=ParticipantRole.USER.value,
                        content=user_input
                    ),
                    agent_id=classifier_result.selected_agent.id,
                    session_id=session_id,
                    metadata=additional_params
                )

            # --- Reasoning Stage: Classify intent ---
            classifier_result: ClassifierResult = await self.measure_execution_time(
                "Classifying user intent",
                lambda: self.classifier.classify(user_input, chat_history),
            )

            if self.config.LOG_CLASSIFIER_OUTPUT:
                self.print_intent(user_input, classifier_result)

            # --- Execution Stage: Route to appropriate agent ---
            agent_response = await self.measure_execution_time(
                "Agent execution",
                lambda: self._execute_agent(
                    classifier_result.selected_agent,
                    user_input,
                    user_id,
                    session_id,
                    additional_params,
                ),
            )

            # Store agent's response if it's a conversation message
            if self.memory_manager and isinstance(agent_response, ConversationMessage):
                await self.memory_manager.store_message(
                    message=agent_response,
                    agent_id=classifier_result.selected_agent.id,
                    session_id=session_id
                )

            return AgentResponse(
                metadata=self.create_metadata(
                    classifier_result,
                    user_input,
                    user_id,
                    session_id,
                    additional_params,
                ),
                output=agent_response,
                streaming=classifier_result.selected_agent.is_streaming_enabled(),
            )

        except Exception as error:
            self.logger.error(f"Error in route_request: {str(error)}")
            raise

    async def dispatch_to_agent(
        self, params: Dict[str, Any]
    ) -> Union[ConversationMessage, AsyncIterable[Any], str]:
        """
        Dispatch the user's request to the selected agent with retry logic.
        """
        user_input = params["user_input"]
        user_id = params["user_id"]
        session_id = params["session_id"]
        classifier_result: ClassifierResult = params["classifier_result"]
        additional_params = params.get("additional_params", {})
        max_retries = 3
        retry_delay = 1.0
        last_error = None

        # Validate agent selection
        selected_agent = classifier_result.selected_agent
        if not selected_agent:
            return self._handle_no_agent_selected()

        # Validate agent is registered
        if selected_agent.id not in self.agent_registry.agents:
            self.logger.error(f"Agent {selected_agent.name} not registered")
            return "Selected agent is not available. Using default agent instead."

        # Get chat history
        agent_chat_history = await self.storage.fetch_chat(
            user_id, session_id, selected_agent.id
        )

        start_time = time.time()
        success = False

        try:
            # Check agent availability
            if not self.agent_registry.add_task(selected_agent.id, session_id):
                return "The selected agent is currently busy. Please try again later."

            # Attempt request with retries
            for attempt in range(max_retries):
                try:
                    timer_name = (
                        f"Agent {selected_agent.name} | "
                        f"Processing request (Attempt {attempt + 1}/{max_retries})"
                    )
                    
                    response = await self.measure_execution_time(
                        timer_name,
                        lambda: selected_agent.process_request(
                            user_input,
                            user_id,
                            session_id,
                            agent_chat_history,
                            additional_params,
                        ),
                    )
                    
                    if not self._validate_response(response):
                        raise ValueError(f"Invalid response format from agent {selected_agent.name}")
                    
                    success = True
                    return response

                except Exception as e:
                    last_error = e
                    self.logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed for agent "
                        f"{selected_agent.name}: {str(e)}"
                    )
                    
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    
                    # On final attempt, try fallback to default agent
                    if selected_agent.id != self.default_agent.id:
                        self.logger.info("Falling back to default agent")
                        return await self.default_agent.process_request(
                            user_input,
                            user_id,
                            session_id,
                            agent_chat_history,
                            additional_params,
                        )
                    raise

        except Exception as e:
            return self._handle_dispatch_error(e, last_error, selected_agent)

        finally:
            # Track metrics and cleanup
            response_time = time.time() - start_time
            self.agent_registry.track_request(selected_agent.id, success, response_time)
            self.agent_registry.remove_task(selected_agent.id, session_id)

    def _handle_no_agent_selected(self) -> str:
        """Handle case when no agent is selected."""
        return (
            "I'm sorry, but I need more information to understand your request. "
            "Could you please be more specific?"
        )

    def _validate_response(self, response: Any) -> bool:
        """
        Validate the response format from the agent.
        """
        if isinstance(response, ConversationMessage):
            return (
                hasattr(response, "role") and
                hasattr(response, "content") and
                isinstance(response.content, list) and
                len(response.content) > 0 and
                isinstance(response.content[0], dict) and
                "text" in response.content[0]
            )
        elif isinstance(response, str):
            return True
        elif isinstance(response, AsyncIterable):
            return True
        return False

    def _handle_dispatch_error(
        self,
        error: Exception,
        last_error: Optional[Exception],
        selected_agent: Agent
    ) -> str:
        """
        Handle different types of errors during agent dispatch.
        """
        error_message = str(error)
        
        if "timeout" in error_message.lower():
            return (
                "The request timed out. The agent is taking longer than expected to respond. "
                "Please try again."
            )
        elif "connection" in error_message.lower():
            return (
                "There was a connection error while communicating with the agent. "
                "Please try again in a moment."
            )
        elif "resource" in error_message.lower():
            return (
                "The agent is currently experiencing resource constraints. "
                "Please try again later."
            )
        else:
            self.logger.error(f"Unexpected error during agent dispatch: {error_message}")
            if last_error:
                self.logger.error(f"Last error before retry: {str(last_error)}")
            return (
                "I encountered an unexpected error while processing your request. "
                "Please try again or rephrase your question."
            )

    # ------------------- Helper and Utility Methods ------------------- #

    async def measure_execution_time(self, timer_name: str, fn):
        """
        Execute a coroutine, measuring and optionally logging the time it takes.
        """
        if not self.config.LOG_EXECUTION_TIMES:
            return await fn()

        start_time = time.time()
        self.execution_times[timer_name] = start_time

        try:
            result = await fn()
            duration = time.time() - start_time
            self.execution_times[timer_name] = duration
            return result
        except Exception as error:
            duration = time.time() - start_time
            self.execution_times[timer_name] = duration
            raise error

    async def handle_no_agent_selected(
        self,
        classifier_result: ClassifierResult,
        user_input: str,
        user_id: str,
        session_id: str,
        additional_params: Dict[str, str] = None,
    ) -> AgentResponse:
        """
        Handle failures when no agent is selected by the classifier. This method attempts
        to find a fallback agent or uses the default agent if configured.
        """
        if classifier_result.intents:
            best_agent = self._find_best_agent_for_intents(classifier_result.intents)
            if best_agent:
                classifier_result.selected_agent = best_agent
                classifier_result.confidence = 0.7
                self.logger.info(
                    f"Found alternate agent {best_agent.name} based on intents"
                )
                return await self.dispatch_to_agent({
                    "user_input": user_input,
                    "user_id": user_id,
                    "session_id": session_id,
                    "classifier_result": classifier_result,
                    "additional_params": additional_params,
                })

        if self.config.USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED:
            classifier_result = self.get_fallback_result()
            self.logger.info(
                f"Using default agent. Detected intents: {classifier_result.intents}"
            )
            return await self.dispatch_to_agent({
                "user_input": user_input,
                "user_id": user_id,
                "session_id": session_id,
                "classifier_result": classifier_result,
                "additional_params": additional_params,
            })

        # No suitable agent found, no default fallback configured
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

    def create_metadata(
        self,
        intent_classifier_result: Optional[ClassifierResult],
        user_input: str,
        user_id: str,
        session_id: str,
        additional_params: Dict[str, str],
    ) -> AgentProcessingResult:
        """
        Create a standard metadata object capturing the details of a request and
        how it was processed or failed.
        """
        base_metadata = AgentProcessingResult(
            user_input=user_input,
            agent_id="no_agent_selected",
            agent_name="No Agent",
            user_id=user_id,
            session_id=session_id,
            additional_params=additional_params,
        )

        if not intent_classifier_result or not intent_classifier_result.selected_agent:
            base_metadata.additional_params["error_type"] = "classification_failed"
            if intent_classifier_result and intent_classifier_result.fallback_reason:
                base_metadata.additional_params["failure_reason"] = (
                    intent_classifier_result.fallback_reason
                )
            if intent_classifier_result and intent_classifier_result.intents:
                base_metadata.additional_params["detected_intents"] = ",".join(
                    intent_classifier_result.intents
                )
        else:
            base_metadata.agent_id = intent_classifier_result.selected_agent.id
            base_metadata.agent_name = intent_classifier_result.selected_agent.name
            if intent_classifier_result.intents:
                base_metadata.additional_params["detected_intents"] = ",".join(
                    intent_classifier_result.intents
                )

        return base_metadata

    def get_fallback_result(self) -> ClassifierResult:
        """
        Produce a fallback classifier result pointing to the default agent.
        """
        return ClassifierResult(
            selected_agent=self.get_default_agent(),
            confidence=0,
            intents=["fallback_required"],
            fallback_reason="Using default agent as fallback",
        )

    def print_intent(
        self, user_input: str, intent_classifier_result: ClassifierResult
    ) -> None:
        """
        Print the classified intent information for debugging or logging purposes.
        """
        self.logger.log_header("Classified Intent")
        self.logger.info(f"> Text: {user_input}")
        selected_agent_string = (
            intent_classifier_result.selected_agent.name
            if intent_classifier_result.selected_agent
            else "No agent selected"
        )
        self.logger.info(f"> Selected Agent: {selected_agent_string}")
        self.logger.info(f"> Confidence: {intent_classifier_result.confidence:.2f}")
        self.logger.info("")

    async def save_message(
        self,
        message: ConversationMessage,
        user_id: str,
        session_id: str,
        agent: Agent,
    ):
        """
        Save a conversation message to storage if the agent is configured to save chat.
        """
        if agent and agent.save_chat:
            return await self.storage.save_chat_message(
                user_id=user_id,
                session_id=session_id,
                agent_id=agent.id,
                new_message=message,
                max_history_size=self.config.MAX_MESSAGE_PAIRS_PER_AGENT,
            )

    # ------------------- Private Agent Validation ------------------- #

    def _validate_agent_interface(self, agent: Optional[Agent]) -> bool:
        """
        Ensure the agent provides required methods and is initialized.
        """
        if agent is None:
            self.logger.error("Agent is None")
            return False
        
        if not hasattr(agent, "process_request"):
            self.logger.error(f"Agent {agent.name} lacks process_request method")
            return False
        
        process_request = getattr(agent, "process_request")
        if not callable(process_request):
            self.logger.error(f"Agent {agent.name} process_request is not callable")
            return False
        
        if hasattr(agent, "is_initialized") and not agent.is_initialized:
            self.logger.error(f"Agent {agent.name} not properly initialized")
            return False
        
        return True

    async def _validate_agent(self, agent: Optional[Agent]) -> bool:
        """
        Validate agent's interface, registration, and availability before dispatch.
        """
        if agent is None:
            self.logger.error("Agent is None")
            return False
        
        if not self._validate_agent_interface(agent):
            return False
        
        if agent.id not in self.agent_registry.agents:
            self.logger.error(f"Agent {agent.name} not registered")
            return False
        
        if not self.agent_registry.is_agent_available(agent.id):
            self.logger.error(f"Agent {agent.name} not available")
            return False
        
        return True

    def _is_agent_available(self, agent: Optional[Agent]) -> bool:
        """
        Basic availability check for an agent.
        """
        if agent is None:
            return False

        # Check initialization
        if hasattr(agent, "is_initialized") and not agent.is_initialized:
            return False

        # Check task capacity
        if hasattr(agent, "current_tasks") and len(agent.current_tasks) >= agent.max_concurrent_tasks:
            return False

        # Check health status
        if hasattr(agent, "health_status") and agent.health_status != "healthy":
            return False

        # Check maintenance mode
        if hasattr(agent, "maintenance_mode") and agent.maintenance_mode:
            return False

        # Check resources
        if hasattr(agent, "check_resources") and not agent.check_resources():
            return False

        return True

    # ------------------- Private Agent Selection Helpers ------------------- #

    def _find_best_agent_for_intents(self, intents: List[str]) -> Optional[Agent]:
        """
        Find the best matching agent for a given set of intents as a fallback
        when the primary classifier fails to select one.
        """
        best_score = 0
        best_agent = None

        for agent in self.agents.values():
            if not self._is_agent_available(agent):
                continue

            score = self._calculate_agent_score(agent, intents)
            if score > best_score:
                best_score = score
                best_agent = agent

        return best_agent if best_score > 0.4 else None

    def _calculate_agent_score(self, agent: Agent, intents: List[str]) -> float:
        """
        Calculate a 'match score' for each agent based on capabilities, specializations,
        and performance metrics.
        """
        score = 0.0

        # Match capabilities
        if hasattr(agent, "capabilities"):
            matching_caps = [cap for cap in agent.capabilities if any(i in cap for i in intents)]
            if matching_caps:
                score += len(matching_caps) / len(agent.capabilities)

        # Match specializations
        if hasattr(agent, "specializations"):
            matching_specs = [spec for spec in agent.specializations if any(i in spec for i in intents)]
            if matching_specs:
                score += len(matching_specs)

        # Consider success rate
        if hasattr(agent, "success_rate"):
            score *= (1 + agent.success_rate)

        # Penalty for current workload
        if hasattr(agent, "current_tasks"):
            workload_penalty = len(agent.current_tasks) * 0.1
            score = max(0, score - workload_penalty)

        # Reward shorter average response time
        if hasattr(agent, "avg_response_time"):
            time_score = 1.0 / (1.0 + agent.avg_response_time)
            score *= (1 + time_score)

        # Penalty for higher error rate
        if hasattr(agent, "error_rate"):
            error_penalty = agent.error_rate * 0.5
            score = max(0, score - error_penalty)

        return score

    # ------------------- Message Queue Processing ------------------- #

    async def _process_message_queue(self):
        """
        Continuously process messages from the queue, dispatching them to
        the appropriate agent.
        """
        while True:
            try:
                message = await self.message_queue.get()
                if message.target_agent not in self.agents:
                    self.logger.error(f"Unknown target agent: {message.target_agent}")
                    continue

                target_agent = self.agents[message.target_agent]
                response = await self._dispatch_to_agent(target_agent, message)

                # Fulfill any pending future awaiting this response
                if message.correlation_id in self.pending_responses:
                    future = self.pending_responses[message.correlation_id]
                    if not future.done():
                        future.set_result(response)

                self.message_queue.task_done()

            except Exception as e:
                self.logger.error(f"Error processing message: {str(e)}")
                continue

    async def _dispatch_to_agent(self, agent: Agent, message: AgentMessage):
        """
        A helper method for message-based dispatch to a target agent.
        """
        # Implement message handling or call agent.process_request if needed.
        return await agent.process_request(
            message.content.get("text", ""),
            message.user_id,
            message.session_id,
            [],
            message.content,
        )

    # ------------------- Shutdown ------------------- #

    async def shutdown(self):
        """
        Gracefully shut down any background tasks, clearing message queues
        and pending futures.
        """
        self.message_processor_task.cancel()
        try:
            await self.message_processor_task
        except asyncio.CancelledError:
            pass

        self.pending_responses.clear()
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
                