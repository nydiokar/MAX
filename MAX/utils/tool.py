from typing import Any, Optional, Callable, get_type_hints, Union, Dict, List
import inspect
from functools import wraps
import re
from dataclasses import dataclass
from datetime import datetime
from MAX.types import AgentProviderType, ConversationMessage, ParticipantRole

@dataclass
class PropertyDefinition:
    type: str
    description: str
    enum: Optional[list] = None

@dataclass
class AgentToolResult:
    tool_use_id: str
    content: Any

    def to_anthropic_format(self) -> dict:
        return {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": self.content
        }

    def to_bedrock_format(self) -> dict:
        return {
            "toolResult": {
                "toolUseId": self.tool_use_id,
                "content": [{"text": self.content}]
            }
        }

@dataclass
class AgentTool:
    """Tool definition for agents"""
    name: str
    description: str
    provider: AgentProviderType
    function: Callable
    parameters: Dict[str, Any]
    required: List[str]
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # Extract docstring if description not provided
        if self.description is None:
            docstring = inspect.getdoc(self.function)
            if docstring:
                # Get the first paragraph of the docstring (before any parameter descriptions)
                self.func_description = docstring.split('\n\n')[0].strip()
            else:
                self.func_description = f"Function to {self.name}"
        else:
            self.func_description = self.description

        if not self.function:
            raise ValueError("Function must be provided")

        # Extract properties from the function if not passed
        self.properties = self._extract_properties(self.function)
        self.required = self.required or list(self.properties.keys())
        self.func = self._wrap_function(self.function)

        # Add enum values to properties if they exist
        for prop_name, enum_vals in self.enum_values.items():
            if prop_name in self.properties:
                self.properties[prop_name]["enum"] = enum_vals

    def _extract_properties(self, func: Callable) -> dict[str, dict[str, Any]]:
        """Extract properties from the function's signature and type hints"""
        # Get function's type hints and signature
        type_hints = get_type_hints(func)
        sig = inspect.signature(func)

        # Parse docstring for parameter descriptions
        docstring = inspect.getdoc(func) or ""
        param_descriptions = {}

        # Extract parameter descriptions using regex
        param_matches = re.finditer(r':param\s+(\w+)\s*:\s*([^:\n]+)', docstring)
        for match in param_matches:
            param_name = match.group(1)
            description = match.group(2).strip()
            param_descriptions[param_name] = description

        properties = {}
        for param_name, param in sig.parameters.items():
            # Skip 'self' parameter for class methods
            if param_name == 'self':
                continue

            param_type = type_hints.get(param_name, Any)

            # Convert Python types to JSON schema types
            type_mapping = {
                int: "integer",
                float: "number",
                str: "string",
                bool: "boolean",
                list: "array",
                dict: "object"
            }

            json_type = type_mapping.get(param_type, "string")

            # Use docstring description if available, else create a default one
            description = param_descriptions.get(param_name, f"The {param_name} parameter")

            properties[param_name] = {
                "type": json_type,
                "description": description
            }

        return properties

    def _wrap_function(self, func: Callable) -> Callable:
        """Wrap the function to preserve its metadata and handle async/sync functions"""
        @wraps(func)
        async def wrapper(**kwargs):
            result = func(**kwargs)
            if inspect.iscoroutine(result):
                return await result
            return result
        return wrapper

    def to_claude_format(self) -> dict[str, Any]:
        """Convert generic tool definition to Claude format"""
        return {
            "name": self.name,
            "description": self.func_description,
            "input_schema": {
                "type": "object",
                "properties": self.properties,
                "required": self.required
            }
        }

    def to_openai_format(self) -> dict[str, Any]:
        """Convert generic tool definition to OpenAI format"""
        return {
            "type": "function",
            "function": {
                "name": self.name.lower().replace("_tool", ""),
                "description": self.func_description,
                "parameters": {
                    "type": "object",
                    "properties": self.properties,
                    "required": self.required,
                    "additionalProperties": False
                }
            }
        }

class AgentTools:
    """Collection of tools available to agents"""
    def __init__(self):
        self.tools: Dict[str, AgentTool] = {}

    def register(self, tool: AgentTool) -> None:
        """Register a new tool"""
        self.tools[tool.name] = tool

    def get(self, name: str) -> Optional[AgentTool]:
        """Get a tool by name"""
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        """List all available tool names"""
        return list(self.tools.keys())

    async def tool_handler(self, provider_type, response: Any, _conversation: list[dict[str, Any]]) -> Any:
        if not response.content:
            raise ValueError("No content blocks in response")

        tool_results = []
        content_blocks = response.content

        for block in content_blocks:
            # Determine if it's a tool use block based on platform
            tool_use_block = self._get_tool_use_block(provider_type, block)
            if not tool_use_block:
                continue

            tool_name = (
                tool_use_block.get("name")
                if  provider_type ==  AgentProviderType.BEDROCK.value
                else tool_use_block.name
            )

            tool_id = (
                tool_use_block.get("toolUseId")
                if  provider_type ==  AgentProviderType.BEDROCK.value
                else tool_use_block.id
            )

            # Get input based on platform
            input_data = (
                tool_use_block.get("input", {})
                if  provider_type ==  AgentProviderType.BEDROCK.value
                else tool_use_block.input
            )

            # Process the tool use
            result = await self._process_tool(tool_name, input_data)

            # Create tool result
            tool_result = AgentToolResult(tool_id, result)

            # Format according to platform
            formatted_result = (
                tool_result.to_bedrock_format()
                if  provider_type ==  AgentProviderType.BEDROCK.value
                else tool_result.to_anthropic_format()
            )

            tool_results.append(formatted_result)

        # Create and return appropriate message format
        if  provider_type ==  AgentProviderType.BEDROCK.value:
            return ConversationMessage(
                role=ParticipantRole.USER.value,
                content=tool_results
            )
        else:
            return {
                'role': ParticipantRole.USER.value,
                'content': tool_results
            }

    def _get_tool_use_block(self, provider_type:AgentProviderType, block: dict) -> Union[dict, None]:
        """Extract tool use block based on platform format."""
        if provider_type == AgentProviderType.BEDROCK.value and "toolUse" in block:
            return block["toolUse"]
        elif provider_type ==  AgentProviderType.ANTHROPIC.value and block.type == "tool_use":
            return block
        return None

    def _process_tool(self, tool_name, input_data):
        try:
            tool = next(tool for tool in self.tools.values() if tool.name == tool_name)
            return tool.func(**input_data)
        except StopIteration:
            return (f"Tool '{tool_name}' not found")

    def to_claude_format(self) -> list[dict[str, Any]]:
        """Convert all tools to Claude format"""
        return [tool.to_claude_format() for tool in self.tools.values()]

    def to_bedrock_format(self) -> list[dict[str, Any]]:
        """Convert all tools to Bedrock format"""
        return [tool.to_bedrock_format() for tool in self.tools.values()]


