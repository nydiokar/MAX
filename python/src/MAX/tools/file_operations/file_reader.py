from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, field_validator
import os
from ..base import BaseTool, BaseToolInput, BaseToolOutput, ToolCategory


class FileReaderInput(BaseToolInput):
    file_path: str
    encoding: str = "utf-8"
    max_size_mb: float = 10.0  # Maximum file size in MB

    @field_validator("file_path")
    def validate_path(cls, v):
        path = Path(v)
        if not path.exists():
            raise ValueError(f"File does not exist: {v}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {v}")
        return str(path.absolute())

    @field_validator("file_path")
    def validate_file_size(cls, v, values):
        max_size = (
            values.get("max_size_mb", 10.0) * 1024 * 1024
        )  # Convert to bytes
        if os.path.getsize(v) > max_size:
            raise ValueError(f"File exceeds maximum size of {max_size_mb}MB")
        return v


class FileReaderOutput(BaseToolOutput):
    content: Optional[str] = None
    size_bytes: int = 0
    encoding_used: str = "utf-8"
    line_count: Optional[int] = None


class FileReaderTool(BaseTool):
    name: str = "file_reader"
    description: str = (
        "Reads content from files with size and encoding validation"
    )
    category: ToolCategory = ToolCategory.FILE_OPERATION
    input_schema: Type[BaseToolInput] = FileReaderInput
    output_schema: Type[BaseToolOutput] = FileReaderOutput
    similes: List[str] = ["read_file", "load_file", "open_file"]
    requires_auth: bool = True  # File operations should require authorization

    def _execute(
        self,
        file_path: str,
        encoding: str = "utf-8",
        max_size_mb: float = 10.0,
    ) -> dict:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()

            size_bytes = os.path.getsize(file_path)
            line_count = len(content.splitlines())

            return {
                "content": content,
                "size_bytes": size_bytes,
                "encoding_used": encoding,
                "line_count": line_count,
            }
        except UnicodeDecodeError:
            # Try with a different encoding if UTF-8 fails
            try:
                with open(file_path, "r", encoding="latin-1") as f:
                    content = f.read()
                return {
                    "content": content,
                    "size_bytes": os.path.getsize(file_path),
                    "encoding_used": "latin-1",
                    "line_count": len(content.splitlines()),
                }
            except Exception as e:
                raise RuntimeError(
                    f"Failed to read file with alternative encoding: {str(e)}"
                )
        except Exception as e:
            raise RuntimeError(f"Failed to read file: {str(e)}")
