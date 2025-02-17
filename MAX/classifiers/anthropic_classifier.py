from typing import List, Optional, Dict, Any
from anthropic import Anthropic
from MAX.utils.helpers import is_tool_input
from MAX.utils.logger import Logger
from MAX.types import ConversationMessage
from MAX.classifiers import Classifier, ClassifierResult
from MAX.agents import Agent
import logging
import json

logging.getLogger("httpx").setLevel(logging.WARNING)

ANTHROPIC_MODEL_ID_CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20240620"

class AnthropicClassifierOptions:
    def __init__(
        self,
        api_key: str,
        model_id: Optional[str] = None,
        inference_config: Optional[Dict[str, Any]] = None,
        min_confidence_threshold: float = 0.7
    ):
        self.api_key = api_key
        self.model_id = model_id
        self.inference_config = inference_config or {}
        self.min_confidence_threshold = min_confidence_threshold

class AnthropicClassifier(Classifier):
    def __init__(self, options: AnthropicClassifierOptions):
        super().__init__()

        if not options.api_key:
            raise ValueError("Anthropic API key is required")

        self.client = Anthropic(api_key=options.api_key)
        self.min_confidence_threshold = options.min_confidence_threshold
        self.model_id = (
            options.model_id or ANTHROPIC_MODEL_ID_CLAUDE_3_5_SONNET
        )

        default_max_tokens = 1000
        self.inference_config = {
            "max_tokens": options.inference_config.get(
                "max_tokens", default_max_tokens
            ),
            "temperature": options.inference_config.get("temperature", 0.0),
            "top_p": options.inference_config.get("top_p", 0.9),
            "stop_sequences": options.inference_config.get(
                "stop_sequences", []
            ),
        }

        # Claude Tool Schema for intent classification and agent selection
        self.tools = [{
            "type": "function",
            "function": {
                "name": "analyzeIntent",
                "description": "Analyze user input and select the most appropriate agent based on intent",
                "parameters": {
                    "type": "object",
                    "required": ["intents", "primary_agent", "fallback_agents", "reasoning"],
                    "properties": {
                        "userinput": {
                            "type": "string",
                            "description": "The original user input"
                        },
                        "primary_agent": {
                            "type": "object",
                            "required": ["name", "confidence", "reasoning"],
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Name of the selected primary agent"
                                },
                                "confidence": {
                                    "type": "number",
                                    "description": "Confidence level between 0 and 1"
                                },
                                "reasoning": {
                                    "type": "string",
                                    "description": "Reasoning for selecting this agent"
                                }
                            }
                        },
                        "fallback_agents": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["name", "confidence", "reasoning"],
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "Name of the fallback agent"
                                    },
                                    "confidence": {
                                        "type": "number",
                                        "description": "Confidence level between 0 and 1"
                                    },
                                    "reasoning": {
                                        "type": "string",
                                        "description": "Reasoning for selecting this fallback agent"
                                    }
                                }
                            }
                        },
                        "detected_intents": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "List of detected intents in the user input"
                        }
                    }
                }
            }
        }]

        self.system_prompt = """You are an expert intent classifier for a multi-agent AI system. Your role is to:
1. Analyze user input to detect intents and required capabilities
2. Select the most appropriate primary agent based on their specialties and the detected intents
3. Identify suitable fallback agents in case the primary agent is unavailable
4. Provide clear reasoning for each agent selection

Consider the following when making selections:
- Match agent specialties to detected intents
- Consider agent capabilities against task requirements
- Factor in context from chat history when provided
- Use confidence scores to reflect certainty of match

Agents must be selected from the provided list of available agents."""

    async def process_request(
        self, input_text: str, chat_history: List[ConversationMessage]
    ) -> ClassifierResult:
        user_message = {"role": "user", "content": input_text}
        agent_context = "\n\nAvailable agents:\n" + "\n".join(
            [f"- {agent.name}: {agent.description}" for agent in self.agents.values()]
        )

        try:
            response = await self.client.messages.create(
                model=self.model_id,
                max_tokens=self.inference_config["max_tokens"],
                messages=[user_message],
                system=self.system_prompt + agent_context,
                temperature=self.inference_config["temperature"],
                top_p=self.inference_config["top_p"],
                tools=self.tools,
                tool_choice={"type": "function", "function": {"name": "analyzePrompt"}}  # Force tool usage
            )

            # Debug logging
            Logger.info(f"API Response: {response}")

            # Find tool call with our analyzePrompt function
            tool_calls_content = next(
                (c for c in response.content if is_tool_input(c)), None
            )
            
            if not tool_calls_content:
                Logger.error("No tool calls found in response")
                Logger.debug(f"Response content: {response.content}")
                raise ValueError("No tool calls found in response")
            
            analyze_tool = next(
                (t for t in tool_calls_content.tool_calls 
                 if t.function.name == "analyzePrompt"), None
            )
            
            if not analyze_tool:
                Logger.error("analyzePrompt tool not found in tool calls")
                raise ValueError("No analyzePrompt tool call found in response")

            tool_data = json.loads(analyze_tool.function.arguments)
            
            if not tool_data:
                raise ValueError("Tool response data is empty")

            # Extract and store detected intents
            self.detected_intents = tool_data["detected_intents"]
            
            # Extract required capabilities and adjust based on chat history context
            self.required_capabilities = self._extract_required_capabilities(
                self.detected_intents,
                chat_history
            )
            
            # Log extracted information for debugging
            Logger.info(f"Detected intents: {self.detected_intents}")
            Logger.info(f"Required capabilities: {self.required_capabilities}")

            # Get primary agent selection with capability score
            primary_agent_data = tool_data["primary_agent"]
            primary_result = self.get_agent_by_name(primary_agent_data["name"])
            
            if primary_result:
                primary_agent, capability_score = primary_result
                primary_confidence = float(primary_agent_data["confidence"]) * capability_score

                # Enhanced validation with reason tracking
                is_available, reason = await self.check_agent_availability(primary_agent)
                
                if is_available and primary_confidence >= self.min_confidence_threshold:
                    return ClassifierResult(
                        selected_agent=primary_agent,
                        confidence=primary_confidence,
                        intents=self.detected_intents,
                        capabilities_score=capability_score
                    )
                    
                # Try fallback agents with enhanced validation
                for fallback in tool_data["fallback_agents"]:
                    fallback_result = self.get_agent_by_name(fallback["name"])
                    if fallback_result:
                        fallback_agent, fallback_score = fallback_result
                        fallback_confidence = float(fallback["confidence"]) * fallback_score
                        
                        is_available, reason = await self.check_agent_availability(fallback_agent)
                        if is_available and fallback_confidence >= self.min_confidence_threshold:
                            return ClassifierResult(
                                selected_agent=fallback_agent,
                                confidence=fallback_confidence,
                                intents=self.detected_intents,
                                capabilities_score=fallback_score,
                                fallback_reason=f"Primary agent unavailable: {reason}"
                            )

            # No suitable agent found
            fallback_reason = ("Primary agent confidence or capability score too low" 
                             if primary_result else "No matching agent found")
            
            return ClassifierResult(
                selected_agent=None,
                confidence=0.0,
                intents=tool_data["detected_intents"],
                fallback_reason="No suitable agent available"
            )

        except Exception as error:
            Logger.error(f"Error in intent classification: {str(error)}")
            raise error
            
    async def check_agent_availability(self, agent: Agent) -> tuple[bool, Optional[str]]:
        """
        Enhanced check for agent availability and capabilities.
        Returns (is_available, reason) tuple.
        """
        try:
            # Basic interface check
            if not hasattr(agent, 'process_request') or not callable(getattr(agent, 'process_request')):
                return False, "Agent lacks required interface"
                
            # Check if agent is initialized
            if not hasattr(agent, 'is_initialized') or not agent.is_initialized:
                return False, "Agent not properly initialized"
                
            # Check agent state
            if hasattr(agent, 'get_state'):
                state = await agent.get_state()
                if state.get('status') == 'busy':
                    return False, "Agent is currently busy"
                if state.get('error_count', 0) > agent.MAX_CONSECUTIVE_ERRORS:
                    return False, "Agent exceeded error threshold"
                    
            # Check agent capabilities match required intents
            if hasattr(agent, 'capabilities'):
                if not any(cap in agent.capabilities for cap in self.required_capabilities):
                    return False, "Agent lacks required capabilities"
                    
            return True, None
            
        except Exception as e:
            Logger.error(f"Error checking agent availability: {str(e)}")
            return False, f"Availability check failed: {str(e)}"
            
    def get_agent_by_name(self, name: str) -> Optional[tuple[Agent, float]]:
        """
        Get agent by name with capability matching score.
        Returns tuple of (agent, matching_score) or None if not found.
        """
        matching_agents = []
        
        for agent in self.agents.values():
            if agent.name.lower() == name.lower():
                # Calculate capability match score
                score = 0.0
                if hasattr(agent, 'capabilities'):
                    matching_caps = set(agent.capabilities) & set(self.required_capabilities)
                    if self.required_capabilities:
                        score = len(matching_caps) / len(self.required_capabilities)
                    else:
                        score = 1.0 if matching_caps else 0.0
                        
                # Check specialization score
                if hasattr(agent, 'specializations'):
                    spec_match = any(spec in self.detected_intents for spec in agent.specializations)
                    score = score * 1.2 if spec_match else score * 0.8
                    
                matching_agents.append((agent, score))
                
        # Return agent with highest matching score
        return max(matching_agents, key=lambda x: x[1]) if matching_agents else None

    def _extract_required_capabilities(self, intents: List[str], chat_history: List[ConversationMessage]) -> List[str]:
        """
        Extract required capabilities based on workflow stages and cognitive functions.
        
        Args:
            intents: List of detected intents
            chat_history: Chat history for context analysis
            
        Returns:
            List of required capabilities aligned with Memory→Reasoning→Execution workflow
        """
        # Initialize capabilities set to handle duplicates
        capabilities = set()
        
        # Map intents to core cognitive functions
        for intent in intents:
            if intent in self.capability_map:
                capabilities.update(self.capability_map[intent])
                
        # Analyze chat history to determine workflow stage requirements
        if chat_history:
            recent_messages = chat_history[-5:]  # Look at last 5 messages for context
            for msg in recent_messages:
                content = msg.content[0]['text'].lower() if isinstance(msg.content, list) else str(msg.content).lower()
                
                # Memory stage indicators
                if any(term in content for term in ['remember', 'recall', 'previous', 'history', 'context']):
                    capabilities.update(self.workflow_stages['memory']['capabilities'])
                    
                # Reasoning stage indicators
                if any(term in content for term in ['think', 'analyze', 'consider', 'decide', 'evaluate']):
                    capabilities.update(self.workflow_stages['reasoning']['capabilities'])
                    
                # Execution stage indicators
                if any(term in content for term in ['do', 'execute', 'perform', 'implement', 'run']):
                    capabilities.update(self.workflow_stages['execution']['capabilities'])
                    
        # Log capability extraction results
        Logger.info(f"Extracted capabilities from intents: {capabilities}")
        return list(capabilities)
