"""Test helper utilities"""
from typing import Any, Dict, Union, List
from datetime import datetime
from MAX.types import (
    ConversationMessage,
    TimestampedMessage,
    ParticipantRole
)

def is_tool_input(data: Any) -> bool:
    """Test if the input matches tool input format"""
    if not isinstance(data, dict):
        return False
    return all(key in data for key in ['selected_agent', 'confidence'])

def conversation_to_dict(
    messages: Union[ConversationMessage, TimestampedMessage, List[Union[ConversationMessage, TimestampedMessage]]]
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Convert conversation messages to dictionary format"""
    if isinstance(messages, list):
        return [conversation_to_dict(msg) for msg in messages]

    result = {
        "role": messages.role,
        "content": messages.content
    }

    if isinstance(messages, TimestampedMessage):
        result["timestamp"] = messages.timestamp

    return result
