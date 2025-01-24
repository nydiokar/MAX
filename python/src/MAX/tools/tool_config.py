from typing import Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field
from enum import Enum


class ToolPermission(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    FULL = "full"


class ToolAccessLevel(str, Enum):
    RESTRICTED = "restricted"
    STANDARD = "standard"
    ELEVATED = "elevated"
    ADMIN = "admin"


class ToolConfig(BaseModel):
    """Configuration for tools"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: Optional[str] = None
    enabled: bool = True
    required_permissions: List[ToolPermission] = [ToolPermission.READ]
    access_level: ToolAccessLevel = ToolAccessLevel.STANDARD
    rate_limit: Optional[int] = None  # Requests per minute
    timeout_seconds: float = 30.0
    max_retries: int = 3
    timeout: float = 30.0
    cache_enabled: bool = True
    # Change Dict[str, Any] to more specific types
    settings: Dict[str, Union[str, int, float, bool, dict, list]] = Field(
        default_factory=dict
    )
    parameters: Dict[str, Union[str, int, float, bool, dict, list]] = Field(
        default_factory=dict
    )


class ToolCategoryConfig(BaseModel):
    """Configuration for tool categories"""

    enabled: bool = True
    default_access_level: ToolAccessLevel = ToolAccessLevel.STANDARD
    default_permissions: List[ToolPermission] = [ToolPermission.READ]
    tools: Dict[str, ToolConfig] = Field(default_factory=dict)


class GlobalToolConfig(BaseModel):
    """Global tool configuration"""

    categories: Dict[str, ToolCategoryConfig] = Field(default_factory=dict)
    default_timeout: float = 30.0
    max_concurrent_tools: int = 10
    enable_tool_metrics: bool = True
    tool_execution_log_level: str = "INFO"

    def get_tool_config(
        self, category: str, tool_name: str
    ) -> Optional[ToolConfig]:
        """Get configuration for a specific tool"""
        category_config = self.categories.get(category)
        if not category_config:
            return None
        return category_config.tools.get(tool_name)


# Example usage:
default_config = GlobalToolConfig(
    categories={
        "file_operation": ToolCategoryConfig(
            tools={
                "file_reader": ToolConfig(
                    name="file_reader",  # <— Required field
                    required_permissions=[ToolPermission.READ],
                    access_level=ToolAccessLevel.STANDARD,
                    rate_limit=100,
                    # Replace custom_settings with settings (the field in ToolConfig)
                    settings={
                        "max_file_size_mb": 10,
                        "allowed_extensions": [".txt", ".log", ".csv"],
                    },
                )
            }
        )
    }
)
