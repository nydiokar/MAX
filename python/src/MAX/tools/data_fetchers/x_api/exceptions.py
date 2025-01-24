from ..base.exceptions import FetcherError


class XAPIError(FetcherError):
    """Base exception for X API specific errors."""

    pass


class XAPIAuthError(XAPIError):
    """Raised when authentication fails."""

    pass


class XAPIResourceNotFound(XAPIError):
    """Raised when requested resource doesn't exist."""

    pass


class XAPIInvalidRequest(XAPIError):
    """Raised when request is malformed or invalid."""

    def __init__(self, message: str, errors: list = None):
        self.errors = errors or []
        super().__init__(message)
