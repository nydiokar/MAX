from typing import List, Dict, Optional, AsyncIterable, Any, Tuple
from MAX.agents import Agent, AgentOptions
from MAX.types import ConversationMessage, ParticipantRole
from MAX.utils import Logger
import ollama
from dataclasses import dataclass

@dataclass
class OllamaAgentOptions(AgentOptions):
    model_id: str = "llama3.1:8b-instruct-q8_0"
    streaming: bool = False
    temperature: float = 0.7
    top_p: float = 0.9

    # Adding classification-specific templates
    classification_prompt: str = """
    Classify the intent of the following input into one of these categories:
    - question (asking for information)
    - command (requesting an action)
    - statement (providing information)
    - clarification (asking for explanation)
    - task (requesting to perform a task)
    
    Provide your answer in the format:
    INTENT: <category>
    CONFIDENCE: <0.0-1.0>
    
    Input: {input_text}
    """

class OllamaAgent(Agent):
    def __init__(self, options: OllamaAgentOptions):
        super().__init__(options)
        self.model_id = options.model_id
        self.streaming = options.streaming
        self.temperature = options.temperature
        self.top_p = options.top_p
        self.classification_prompt = options.classification_prompt
        self.logger = Logger()

    async def classify_intent(self, input_text: str) -> Tuple[str, float]:
        """Classify the intent of the input text"""
        self.logger.info(f"Classifying intent for: {input_text}")
        
        try:
            # Format the classification prompt
            prompt = self.classification_prompt.format(input_text=input_text)
            
            # Use lower temperature for classification
            response = ollama.chat(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.1,  # Lower temperature for more consistent classification
                    "top_p": 0.9
                }
            )
            
            response_text = response['message']['content']
            self.logger.debug(f"Raw classification response: {response_text}")
            
            # Parse the response
            intent = "unknown"
            confidence = 0.0
            
            for line in response_text.split('\n'):
                if line.startswith('INTENT:'):
                    intent = line.replace('INTENT:', '').strip().lower()
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.replace('CONFIDENCE:', '').strip())
                    except ValueError:
                        confidence = 0.0
            
            self.logger.info(f"Classified intent: {intent} (confidence: {confidence})")
            return intent, confidence
            
        except Exception as e:
            self.logger.error(f"Error in intent classification: {str(e)}")
            raise

    async def process_request(self, input_text: str, user_id: str, session_id: str,
                            chat_history: List[ConversationMessage],
                            additional_params: Optional[Dict[str, str]] = None) -> ConversationMessage | AsyncIterable[Any]:
        # First classify the intent
        intent, confidence = await self.classify_intent(input_text)
        
        self.logger.info(f"Processing request with classified intent: {intent} (confidence: {confidence})")
        
        # Continue with regular processing...
        messages = [
            {"role": msg.role, "content": msg.content[0]['text']}
            for msg in chat_history
        ]
        messages.append({"role": ParticipantRole.USER.value, "content": input_text})

        try:
            if self.streaming:
                return await self.handle_streaming_response(messages)
            else:
                response = ollama.chat(
                    model=self.model_id,
                    messages=messages,
                    options={
                        "temperature": self.temperature,
                        "top_p": self.top_p
                    }
                )
                return ConversationMessage(
                    role=ParticipantRole.ASSISTANT.value,
                    content=[{
                        "text": response['message']['content'],
                        "metadata": {
                            "intent": intent,
                            "confidence": confidence
                        }
                    }]
                )
        except Exception as e:
            self.logger.error(f"Error in process_request: {str(e)}")
            raise
