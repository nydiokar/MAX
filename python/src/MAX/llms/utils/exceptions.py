from typing import Optional


class LLMError(Exception):
    """Base exception for LLM-related errors"""

    def __init__(self, message: str, provider: Optional[str] = None):
        self.provider = provider
        super().__init__(f"[{provider}] {message}" if provider else message)


class LLMProviderError(LLMError):
    """Raised when the LLM provider encounters an error"""

    pass


class LLMConfigError(LLMError):
    """Raised when there's a configuration error"""

    pass


class LLMRateLimitError(LLMProviderError):
    """Raised when hitting rate limits"""

    pass


class LLMAuthenticationError(LLMProviderError):
    """Raised when authentication fails"""

    pass
