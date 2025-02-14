from typing import Any, Callable, Dict, List
from pydantic import BaseModel


class Tool(BaseModel):
    """Simplified tool structure for LLM interaction"""

    name: str
    description: str
    examples: List[str]  # Natural language examples
    function: Callable
    requires_auth: bool = False

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute tool with flexible arguments"""
        try:
            return {"success": True, "result": self.function(**kwargs)}
        except Exception as e:
            return {"success": False, "error": str(e)}
