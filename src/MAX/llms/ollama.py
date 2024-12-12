from typing import Dict, List, Optional, Union
from enum import Enum
import ollama
from MAX.llms.base import AsyncLLMBase
from MAX.config.llms.base import BaseLlmConfig

class OllamaModelType(Enum):
    GENERAL = "llama2"
    CODE = "codellama"
    FAST = "mistral"
    INSTRUCT = "llama2:13b-instruct"

class OllamaLLM(AsyncLLMBase):
    """Enhanced Ollama implementation with model specialization"""
    
    DEFAULT_MODELS = {
        "general": OllamaModelType.GENERAL.value,
        "code": OllamaModelType.CODE.value,
        "fast": OllamaModelType.FAST.value,
        "instruct": OllamaModelType.INSTRUCT.value
    }
    
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)
        self.client = ollama.Client()
        self._ensure_model_exists()

    def _ensure_model_exists(self):
        """Ensure the specified model exists locally"""
        if not self.config.model:
            self.config.model = self.DEFAULT_MODELS["general"]
            
        local_models = self.client.list()["models"]
        if not any(model.get("name") == self.config.model for model in local_models):
            self.client.pull(self.config.model)

    async def _generate(self, messages: List[Dict[str, str]]) -> str:
        """Implement Ollama-specific generation with enhanced options"""
        try:
            response = await ollama.chat(
                model=self.config.model,
                messages=messages,
                options={
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "num_predict": self.config.max_tokens,
                    "stop": messages[-1].get("stop", []),  # Optional stop sequences
                    "repeat_penalty": 1.1,  # Prevent repetitive outputs
                    "num_ctx": 4096,  # Context window size
                }
            )
            return response['message']['content']
            
        except Exception as e:
            # Log error and try fallback model if configured
            if hasattr(self.config, 'fallback_model'):
                return await self._generate_with_fallback(messages)
            raise e

    async def _generate_with_fallback(self, messages: List[Dict[str, str]]) -> str:
        """Fallback generation with a simpler model"""
        response = await ollama.chat(
            model=self.DEFAULT_MODELS["fast"],  # Use faster, smaller model as fallback
            messages=messages,
            options={
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            }
        )
        return response['message']['content']

    @classmethod
    def get_model_for_task(cls, task_type: str) -> str:
        """Get appropriate model for specific task types"""
        task_models = {
            "code": cls.DEFAULT_MODELS["code"],
            "fast": cls.DEFAULT_MODELS["fast"],
            "instruction": cls.DEFAULT_MODELS["instruct"],
            "general": cls.DEFAULT_MODELS["general"]
        }
        return task_models.get(task_type, cls.DEFAULT_MODELS["general"])
    