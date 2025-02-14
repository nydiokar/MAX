from typing import Dict, Type, Optional, Any, List, Tuple
from enum import Enum
from pydantic import BaseModel, Field, ValidationError

from datetime import datetime
from typing import Callable
import logging
from MAX.tools.semantic_matcher import ToolSemanticMatcher
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    FILE_OPERATION = "file_operation"
    DATA_FETCHING = "data_fetching"
    ANALYSIS = "analysis"
    COMMUNICATION = "communication"
    SYSTEM = "system"


class ToolMetadata(BaseModel):
    """Metadata for tool tracking and management"""

    created_at: datetime
    last_used: Optional[datetime] = None
    usage_count: int = 0
    average_latency: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None


class BaseToolInput(BaseModel):
    """Base class for all tool inputs"""

    class Config:
        arbitrary_types_allowed = True


class BaseToolOutput(BaseModel):
    """Base class for all tool outputs"""

    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float


class BaseTool(BaseModel):
    """Base class for all tools"""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    category: ToolCategory
    input_schema: Type[BaseToolInput]
    output_schema: Type[BaseToolOutput] = BaseToolOutput
    requires_auth: bool = False
    rate_limited: bool = False
    version: str = "1.0.0"
    similes: List[str] = Field(
        default_factory=list, description="Alternative names for the tool"
    )

    _execute: Callable  # The actual tool implementation
    _metadata: ToolMetadata = ToolMetadata(created_at=datetime.now())

    def execute(self, **kwargs) -> BaseToolOutput:
        """Execute the tool with validation and error handling"""
        try:
            # Validate input
            input_data = self.input_schema(**kwargs)

            # Execute tool
            start_time = datetime.now()
            result = self._execute(**input_data.dict())
            execution_time = (datetime.now() - start_time).total_seconds()

            # Update metadata
            self._metadata.last_used = datetime.now()
            self._metadata.usage_count += 1
            self._metadata.average_latency = (
                self._metadata.average_latency
                * (self._metadata.usage_count - 1)
                + execution_time
            ) / self._metadata.usage_count

            # Validate and return output
            output = self.output_schema(
                success=True, result=result, execution_time=execution_time
            )
            return output

        except ValidationError as e:
            self._metadata.error_count += 1
            self._metadata.last_error = str(e)
            return self.output_schema(
                success=False,
                result=None,
                error=f"Validation error: {str(e)}",
                execution_time=0.0,
            )
        except Exception as e:
            self._metadata.error_count += 1
            self._metadata.last_error = str(e)
            return self.output_schema(
                success=False,
                result=None,
                error=f"Execution error: {str(e)}",
                execution_time=0.0,
            )


class ToolRegistry:
    """Central registry for all tools"""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[str, List[str]] = defaultdict(list)
        self._semantic_matcher = ToolSemanticMatcher()
        self._usage_stats: Dict[str, Dict[str, Any]] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool with metadata and examples"""
        self._tools[tool.name] = tool
        self._categories[tool.category].append(tool.name)
        self._usage_stats[tool.name] = {
            "calls": 0,
            "success_rate": 0.0,
            "avg_latency": 0.0,
            "last_used": None,
        }

    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute tool with tracking and error handling"""
        tool = self._tools.get(tool_name)
        if not tool:
            return {"success": False, "error": f"Tool {tool_name} not found"}

        start_time = time.time()
        try:
            result = await tool.execute(**kwargs)
            self._update_stats(tool_name, True, time.time() - start_time)
            return result
        except Exception as e:
            self._update_stats(tool_name, False, time.time() - start_time)
            return {"success": False, "error": str(e)}

    def get_tools_description(self, format_type: str = "natural") -> str:
        """Get tool descriptions in different formats"""
        if format_type == "natural":
            return self._get_natural_language_description()
        elif format_type == "structured":
            return self._get_structured_description()
        return self._get_basic_description()

    def find_similar_tools(
        self, query: str, threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """Find semantically similar tools"""
        return self._semantic_matcher.find_similar(
            query, self._tools.values(), threshold
        )

    def _update_stats(
        self, tool_name: str, success: bool, execution_time: float
    ) -> None:
        """Update tool usage statistics"""
        stats = self._usage_stats[tool_name]
        stats["calls"] += 1
        stats["success_rate"] = (
            stats["success_rate"] * (stats["calls"] - 1) + success
        ) / stats["calls"]
        stats["avg_latency"] = (
            stats["avg_latency"] * (stats["calls"] - 1) + execution_time
        ) / stats["calls"]
        stats["last_used"] = datetime.now()

    def _get_natural_language_description(self) -> str:
        """Generate natural language description of tools"""
        descriptions = []
        for tool in self._tools.values():
            description = f"{tool.name}: {tool.description}"
            if tool.similes:
                description += f" (also known as {', '.join(tool.similes)})"
            descriptions.append(description)
        return "\n".join(descriptions)

    def _get_structured_description(self) -> str:
        """Generate structured description of tools"""
        return "\n".join(
            [
                f"{tool.name}: {tool.description}"
                for tool in self._tools.values()
            ]
        )

    def _get_basic_description(self) -> str:
        """Generate basic description of tools"""
        return "\n".join(
            [
                f"{tool.name}: {tool.description}"
                for tool in self._tools.values()
            ]
        )

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name"""
        return self._tools.get(name)

    def list_tools(self, category: Optional[ToolCategory] = None) -> List[str]:
        """List all registered tools, optionally filtered by category"""
        if category:
            return self._categories[category]
        return list(self._tools.keys())

    def get_tool_info(self, name: str) -> Dict[str, Any]:
        """Get detailed information about a tool"""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool {name} not found")

        return {
            "name": tool.name,
            "description": tool.description,
            "category": tool.category.value,
            "version": tool.version,
            "requires_auth": tool.requires_auth,
            "rate_limited": tool.rate_limited,
            "metadata": tool._metadata.dict(),
            "input_schema": self._get_schema_info(tool.input_schema),
            "output_schema": self._get_schema_info(tool.output_schema),
        }

    def _get_schema_info(self, schema: Type[BaseModel]) -> Dict[str, Any]:
        """Extract schema information from a Pydantic model"""
        return {
            "fields": {
                name: str(field.type_)
                for name, field in schema.model_fields.items()
            },
            "required": schema.model_fields,
        }
