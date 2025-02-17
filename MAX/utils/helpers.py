"""
Helpers method
"""

from typing import Any, List, Dict, Union
from MAX.types import ConversationMessage, TimestampedMessage


def is_tool_input(input_obj: Any) -> bool:
    """Check if the input object is a tool input from Anthropic API."""
    return (
        hasattr(input_obj, "type") 
        and input_obj.type == "tool_calls"
        and hasattr(input_obj, "tool_calls")
        and len(input_obj.tool_calls) > 0
    )


def conversation_to_dict(
    conversation: Union[
        ConversationMessage,
        TimestampedMessage,
        List[Union[ConversationMessage, TimestampedMessage]],
    ]
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Convert conversation to dictionary format."""
    if isinstance(conversation, list):
        return [message_to_dict(msg) for msg in conversation]
    return message_to_dict(conversation)


def message_to_dict(
    message: Union[ConversationMessage, TimestampedMessage]
) -> Dict[str, Any]:
    """Convert a single message to dictionary format."""
    result = {
        "role": (
            message.role.value
            if hasattr(message.role, "value")
            else str(message.role)
        ),
        "content": message.content,
    }
    if isinstance(message, TimestampedMessage):
        result["timestamp"] = message.timestamp
    return result
