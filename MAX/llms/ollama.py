from typing import Dict, List
from enum import Enum

import ollama

from MAX.llms.base import (
    AsyncLLMBase,
)  # Adapt this import to your actual codebase
from MAX.config.llms.base import BaseLlmConfig
from MAX.utils import Logger


class OllamaModelType(Enum):
    """
    Enumerated default model variants for Ollama.
    Customize these as needed for your environment.
    """

    GENERAL = "llama3.1:8b-instruct-q8_0"
    CODE = "llama3.1:8b-instruct-q8_0"
    FAST = "llama3.1:8b-instruct-q8_0"
    INSTRUCT = "llama3.1:8b-instruct-q8_0"


class OllamaLLM(AsyncLLMBase):
    """
    An Ollama-based LLM implementation with specialized model selection,
    fallback handling, and advanced generation options.

    Usage:
    1. Create a BaseLlmConfig (or a subclass) specifying 'model', 'temperature', etc.
    2. Instantiate OllamaLLM with that config.
    3. Call `.generate()` with a list of messages (dicts) to get a response.
    """

    DEFAULT_MODELS = {
        "general": OllamaModelType.GENERAL.value,
        "code": OllamaModelType.CODE.value,
        "fast": OllamaModelType.FAST.value,
        "instruct": OllamaModelType.INSTRUCT.value,
    }

    def __init__(self, config: BaseLlmConfig):
        """
        Initialize the OllamaLLM with a given configuration.

        Args:
            config (BaseLlmConfig): Configuration object specifying model, temperature, etc.
        """
        super().__init__(config)
        self.client = self._create_client()
        self.logger = Logger

    def _create_client(self):
        """
        Create an Ollama client instance.
        This is separated out for easier mocking in tests.

        Returns:
            A new instance of `ollama.Client`.
        """
        from ollama import Client

        if self.config.api_base_url:
            return Client(host=self.config.api_base_url)
        # If no api_base_url is specified, it will use the default.
        return Client()

    async def _ensure_model_exists(self) -> None:
        """
        Check if the model is available locally; if not and auto_pull_models is True, pull it.
        Failures are logged but non-fatal, as the model might already exist locally.
        """
        try:
            if self.config.auto_pull_models and self.config.model:
                # As of Ollama, `pull(model_name)` downloads the model if absent
                await self.client.pull(self.config.model)
        except Exception as e:
            # Log the warning, do not raise
            self.logger.warn(
                f"Failed to pull model '{self.config.model}': {str(e)}"
            )

    async def _generate(self, messages: List[Dict[str, str]]) -> str:
        """
        Perform text generation using Ollama's chat API.
        Applies advanced inference options (temperature, top_p, etc.).

        Args:
            messages (List[Dict[str, str]]): Conversation messages,
                                             typically with keys: {'role': ..., 'content': ...}.

        Returns:
            str: The generated text from the model.

        Raises:
            Exception: If the generation fails and no fallback_model is set.
        """
        await self._ensure_model_exists()

        try:
            response = await ollama.chat(
                model=self.config.model,
                messages=messages,
                options={
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "num_predict": self.config.max_tokens,
                    # Optionally read "stop" sequences from the last message
                    "stop": messages[-1].get("stop", []) if messages else [],
                    "repeat_penalty": 1.1,  # Helps reduce repetitiveness
                    "num_ctx": 4096,  # Context window size
                },
            )
            # The Ollama library's response typically looks like: {"message": {"content": "..."}}
            return response["message"]["content"]

        except Exception as e:
            self.logger.error(f"Ollama generation error: {e}")
            # If a fallback model is specified, try that
            if self.config.fallback_model:
                return await self._generate_with_fallback(messages)
            # Otherwise raise the exception
            raise

    async def _generate_with_fallback(
        self, messages: List[Dict[str, str]]
    ) -> str:
        """
        If primary model fails and fallback_model is specified, generate with a simpler or smaller model.

        Args:
            messages (List[Dict[str, str]]): The conversation history.

        Returns:
            str: The generated text from the fallback model.
        """
        self.logger.info(
            f"Using fallback model '{self.config.fallback_model}' due to primary model error."
        )
        try:
            response = await ollama.chat(
                model=self.config.fallback_model,
                messages=messages,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            )
            return response["message"]["content"]
        except Exception as e:
            # If fallback also fails, re-raise
            self.logger.error(
                f"Fallback model '{self.config.fallback_model}' also failed: {e}"
            )
            raise

    @classmethod
    def get_model_for_task(cls, task_type: str) -> str:
        """
        Retrieve a default model name for a given type of task.

        Args:
            task_type (str): The category of the task, e.g. 'code', 'fast', 'instruction'.

        Returns:
            str: The corresponding model name, or the 'general' model if no match.
        """
        task_models = {
            "code": cls.DEFAULT_MODELS["code"],
            "fast": cls.DEFAULT_MODELS["fast"],
            "instruction": cls.DEFAULT_MODELS["instruct"],
            "general": cls.DEFAULT_MODELS["general"],
        }
        return task_models.get(
            task_type.lower(), cls.DEFAULT_MODELS["general"]
        )
