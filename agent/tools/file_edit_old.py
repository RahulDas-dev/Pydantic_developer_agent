import difflib
import logging
from pathlib import Path
from typing import Any

import aiofiles

from .base import (
    BaseTool,
    ToolExecutionContext,
    ToolExecutionResult,
    ToolParameter,
    ToolSchema,
)

logger = logging.getLogger("tools")


class EditTool(BaseTool):
    """Advanced file editing tool with context validation and diff generation."""

    def __init__(self, root_directory: str | None = None):
        super().__init__(
            name="edit_file",
            description="Replaces text within a file with advanced context validation. "
            "Requires providing significant context around the change to ensure precise targeting.",
        )
        self.root_directory = Path(root_directory) if root_directory else Path.cwd()
        self.root_directory = self.root_directory.resolve()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="file_path",
                    data_type="string",
                    description="The absolute path to the file to modify. Must be within the allowed directory.",
                    required=True,
                ),
                ToolParameter(
                    name="old_string",
                    data_type="string",
                    description="The exact literal text to replace. Include at least 3 lines of context "
                    "BEFORE and AFTER the target text, matching whitespace and indentation precisely. "
                    "This string must uniquely identify the location to be changed.",
                    required=True,
                ),
                ToolParameter(
                    name="new_string",
                    data_type="string",
                    description="The exact literal text to replace old_string with. "
                    "Ensure the resulting code is correct and idiomatic.",
                    required=True,
                ),
                ToolParameter(
                    name="expected_replacements",
                    data_type="integer",
                    description="Number of replacements expected. Defaults to 1. "
                    "Use when you want to replace multiple occurrences.",
                    default=1,
                    extra={
                        "minimum": 1,
                    },
                ),
            ],
            required=["file_path", "old_string", "new_string"],
        )

    def _is_within_root(self, file_path: Path) -> bool:
        """Check if file path is within the root directory."""
        try:
            file_path.resolve().relative_to(self.root_directory)
            return True
        except ValueError:
            return False

    def _validate_parameters(self, parameters: dict[str, Any]) -> str | None:
        """Validate tool parameters."""
        file_path_str = parameters.get("file_path", "")
        old_string = parameters.get("old_string", "")
        new_string = parameters.get("new_string")

        if not file_path_str:
            return "file_path parameter is required"

        try:
            file_path = Path(file_path_str)
            if not file_path.is_absolute():
                return f"File path must be absolute: {file_path_str}"

            if not self._is_within_root(file_path):
                return f"File path must be within root directory ({self.root_directory}): {file_path_str}"
        except Exception as e:
            return f"Invalid file path: {e}"

        if not old_string:
            return "old_string parameter is required and cannot be empty"

        if new_string is None:
            return "new_string parameter is required"

        return None

    async def _calculate_edit(
        self, file_path: Path, old_string: str, new_string: str
    ) -> dict[str, Any]:
        """Calculate the potential edit outcome."""
        try:
            # Read current file content
            current_content = None
            is_new_file = not file_path.exists()

            if not is_new_file:
                async with aiofiles.open(file_path, encoding="utf-8") as f:
                    current_content = await f.read()

            # Count occurrences
            if is_new_file:
                if old_string == "":
                    # Creating new file
                    new_content = new_string
                    occurrences = 1
                else:
                    return {
                        "current_content": None,
                        "new_content": "",
                        "occurrences": 0,
                        "error": {
                            "display": f"File does not exist and old_string is not empty: {file_path}",
                            "raw": "file_not_found_with_old_string",
                        },
                        "is_new_file": True,
                    }
            else:
                if old_string == "":
                    # Cannot replace empty string in existing file
                    return {
                        "current_content": current_content,
                        "new_content": current_content,
                        "occurrences": 0,
                        "error": {
                            "display": "Cannot replace empty string in existing file",
                            "raw": "empty_old_string",
                        },
                        "is_new_file": False,
                    }

                if current_content is None:
                    return {
                        "current_content": None,
                        "new_content": "",
                        "occurrences": 0,
                        "error": {
                            "display": f"Could not read file: {file_path}",
                            "raw": "file_read_error",
                        },
                        "is_new_file": False,
                    }

                occurrences = current_content.count(old_string)
                if occurrences == 0:
                    return {
                        "current_content": current_content,
                        "new_content": current_content,
                        "occurrences": 0,
                        "error": {
                            "display": f"String not found in file: {old_string[:100]}{'...' if len(old_string) > 100 else ''}",
                            "raw": "string_not_found",
                        },
                        "is_new_file": False,
                    }

                new_content = current_content.replace(old_string, new_string)

            return {
                "current_content": current_content,
                "new_content": new_content,
                "occurrences": occurrences,
                "error": None,
                "is_new_file": is_new_file,
            }

        except Exception as e:
            return {
                "current_content": None,
                "new_content": "",
                "occurrences": 0,
                "error": {"display": f"Error reading file: {e!s}", "raw": str(e)},
                "is_new_file": False,
            }

    def _generate_diff(
        self, old_content: str, new_content: str, file_path: Path
    ) -> str:
        """Generate a unified diff showing the changes."""
        old_lines = old_content.splitlines(keepends=True) if old_content else []
        new_lines = new_content.splitlines(keepends=True)

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{file_path.name}",
            tofile=f"b/{file_path.name}",
            lineterm="",
        )

        return "".join(diff)

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ToolExecutionContext,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolExecutionResult:
        """Execute the edit tool."""
        try:
            # Validate parameters
            validation_error = self._validate_parameters(parameters)
            if validation_error:
                return ToolExecutionResult(
                    success=False,
                    content=f"Error: {validation_error}",
                    metadata={"error": "validation_failed"},
                )

            file_path = Path(parameters["file_path"])
            old_string = parameters["old_string"]
            new_string = parameters["new_string"]
            expected_replacements = parameters.get("expected_replacements", 1)

            # Calculate edit
            edit_result = await self._calculate_edit(file_path, old_string, new_string)

            if edit_result["error"]:
                return ToolExecutionResult(
                    success=False,
                    content=f"Error: {edit_result['error']['display']}",
                    metadata={
                        "error": edit_result["error"]["raw"],
                        "file_path": str(file_path),
                    },
                )

            occurrences = edit_result["occurrences"]

            # Check expected replacements
            if occurrences != expected_replacements:
                return ToolExecutionResult(
                    success=False,
                    content=f"Error: Expected {expected_replacements} replacement(s), but found {occurrences} occurrence(s) of the string.",
                    metadata={
                        "error": "replacement_count_mismatch",
                        "expected": expected_replacements,
                        "found": occurrences,
                        "file_path": str(file_path),
                    },
                )

            # Generate diff for display
            current_content = edit_result["current_content"] or ""
            new_content = edit_result["new_content"]
            diff_content = self._generate_diff(current_content, new_content, file_path)

            # Write the file
            file_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                await f.write(new_content)

            # Create result
            operation = "Created" if edit_result["is_new_file"] else "Modified"
            relative_path = file_path.relative_to(self.root_directory)

            display_content = f"{operation} file: {relative_path}"
            if occurrences > 1:
                display_content += f" ({occurrences} replacements made)"

            return ToolExecutionResult(
                success=True,
                content=f"Successfully {operation.lower()} {relative_path}\n\n{diff_content}",
                display_content=display_content,
                metadata={
                    "file_path": str(file_path),
                    "relative_path": str(relative_path),
                    "operation": operation.lower(),
                    "replacements_made": occurrences,
                    "is_new_file": edit_result["is_new_file"],
                },
            )

        except Exception as e:
            logger.error(f"Edit tool execution failed: {e}")
            return ToolExecutionResult(
                success=False,
                content=f"Error: Edit operation failed: {e!s}",
                metadata={"error": "execution_failed", "exception": str(e)},
            )
