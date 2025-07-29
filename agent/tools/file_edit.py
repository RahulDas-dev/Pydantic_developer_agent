import difflib
import logging
from pathlib import Path

import aiofiles
from pydantic import BaseModel, Field
from pydantic_ai import RunContext, ToolReturn

from agent.context import AgentContext

logger = logging.getLogger(__name__)


class EditResult(BaseModel):
    """Result of a file edit operation."""

    file_path: str = Field(description="Path to the file that was edited")
    relative_path: str = Field(description="Path relative to workspace root")
    operation: str = Field(description="Operation performed (created or modified)")
    replacements_made: int = Field(description="Number of replacements made")
    is_new_file: bool = Field(description="Whether the file was newly created")


def _validate_path(file_path: str, workspace_path: str) -> tuple[bool, Path | None, str | None, list[str] | None]:
    """Validate file path and ensure it's within workspace."""
    try:
        path = Path(file_path)
        if not path.is_absolute():
            path = Path(workspace_path) / file_path

        path = path.resolve()
        root_directory = Path(workspace_path).resolve()

        # Check if file is within workspace
        try:
            path.relative_to(root_directory)
            return True, path, None, None
        except ValueError:
            error_message = f"File path must be within workspace directory ({root_directory}): {file_path}"
            error_content = [
                "## Error: Invalid File Path",
                f"The file `{file_path}` must be within the workspace directory.",
            ]
            return False, None, error_message, error_content
    except Exception as e:
        error_message = f"Invalid file path: {e}"
        error_content = ["## Error: Invalid File Path", f"The file path `{file_path}` is invalid: {e}"]
        return False, None, error_message, error_content


def _validate_string_params(old_string: str, new_string: str, path: Path) -> tuple[bool, str | None, list[str] | None]:
    """Validate old_string and new_string parameters."""
    if not old_string and path.exists():
        error_message = "Cannot replace empty string in existing file"
        error_content = ["## Error: Invalid Replacement", "Cannot replace an empty string in an existing file."]
        return False, error_message, error_content

    if new_string is None:
        error_message = "new_string parameter is required"
        error_content = ["## Error: Missing Parameter", "The new_string parameter is required for file editing."]
        return False, error_message, error_content

    return True, None, None


def _read_file_content(path: Path) -> tuple[bool, str | None, str | None, list[str] | None]:
    """Read file content and handle errors."""
    try:
        with open(path, encoding="utf-8") as f:
            return True, f.read(), None, None
    except Exception as e:
        error_message = f"Could not read file: {e!s}"
        error_content = ["## Error: File Read Failed", f"Could not read file `{path}`:", f"- Error: {e!s}"]
        return False, None, error_message, error_content


# Constants
MAX_PREVIEW_LENGTH = 100
MAX_CONTENT_PREVIEW = 1000


def _calculate_new_content(
    is_new_file: bool, current_content: str | None, old_string: str, new_string: str
) -> tuple[bool, str | None, int, str | None, list[str] | None]:
    """Calculate new content and count replacements."""
    # For new files
    if is_new_file:
        if old_string == "":
            # Creating new file
            return True, new_string, 1, None, None

        error_message = "File does not exist and old_string is not empty"
        error_content = [
            "## Error: File Not Found",
            "The file does not exist, but old_string is not empty.",
            "To create a new file, use an empty old_string.",
        ]
        return False, None, 0, error_message, error_content

    # For existing files
    if current_content is None:
        error_message = "Could not read file content"
        error_content = ["## Error: File Read Failed", "Could not read the file content."]
        return False, None, 0, error_message, error_content

    occurrences = current_content.count(old_string)
    if occurrences == 0:
        # String preview with truncation if needed
        preview = old_string[:MAX_PREVIEW_LENGTH]
        if len(old_string) > MAX_PREVIEW_LENGTH:
            preview += "..."

        error_message = f"String not found in file: {preview}"
        error_content = [
            "## Error: String Not Found",
            "The string to replace was not found in the file.",
            "Make sure you've included enough context (3+ lines before and after the target text),",
            "with exact whitespace and indentation.",
        ]
        return False, current_content, 0, error_message, error_content

    new_content = current_content.replace(old_string, new_string)
    return True, new_content, occurrences, None, None


def _check_replacement_count(occurrences: int, expected_replacements: int) -> tuple[bool, str | None, list[str] | None]:
    """Check if the number of replacements matches the expected count."""
    if occurrences != expected_replacements:
        error_message = f"Expected {expected_replacements} replacement(s), but found {occurrences} occurrence(s)"
        error_content = [
            "## Error: Unexpected Replacement Count",
            f"Expected {expected_replacements} replacement(s), but found {occurrences} occurrence(s) of the string.",
            "Use a more specific context to target the exact location to replace.",
        ]
        return False, error_message, error_content

    return True, None, None


def _generate_diff(old_content: str | None, new_content: str, file_path: Path) -> str:
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


def _process_edit_operation(
    file_path: str, workspace_path: str, old_string: str, new_string: str, expected_replacements: int
) -> tuple[bool, ToolReturn | None, dict | None]:
    """Process the edit operation and return success status and tool return."""
    # Validate file path
    valid, path, error_msg, error_content = _validate_path(file_path, workspace_path)
    if not valid:
        return (
            False,
            ToolReturn(
                return_value=error_msg, content=error_content, metadata={"success": False, "error": "invalid_path"}
            ),
            None,
        )

    # Validate string parameters
    valid, error_msg, error_content = _validate_string_params(old_string, new_string, path)
    if not valid:
        return (
            False,
            ToolReturn(
                return_value=error_msg,
                content=error_content,
                metadata={"success": False, "error": "invalid_string_params"},
            ),
            None,
        )

    # Determine if it's a new file
    is_new_file = not path.exists()
    current_content = None

    # Read existing file content if needed
    if not is_new_file:
        valid, content, error_msg, error_content = _read_file_content(path)
        if not valid:
            return (
                False,
                ToolReturn(
                    return_value=error_msg,
                    content=error_content,
                    metadata={"success": False, "error": "file_read_error"},
                ),
                None,
            )
        current_content = content

    # Calculate new content and count replacements
    valid, new_content, occurrences, error_msg, error_content = _calculate_new_content(
        is_new_file, current_content, old_string, new_string
    )
    if not valid:
        return (
            False,
            ToolReturn(
                return_value=error_msg,
                content=error_content,
                metadata={"success": False, "error": "content_calculation_error"},
            ),
            None,
        )

    # Check if replacement count matches expected
    valid, error_msg, error_content = _check_replacement_count(occurrences, expected_replacements)
    if not valid:
        return (
            False,
            ToolReturn(
                return_value=error_msg,
                content=error_content,
                metadata={
                    "success": False,
                    "error": "replacement_count_mismatch",
                    "expected": expected_replacements,
                    "found": occurrences,
                },
            ),
            None,
        )

    # Generate diff for display
    diff_content = _generate_diff(current_content, new_content, path)

    # Prepare result info
    operation = "created" if is_new_file else "modified"
    root_directory = Path(workspace_path).resolve()
    relative_path = path.relative_to(root_directory)

    edit_result = EditResult(
        file_path=str(path),
        relative_path=str(relative_path),
        operation=operation,
        replacements_made=occurrences,
        is_new_file=is_new_file,
    )

    result_info = {
        "path": path,
        "new_content": new_content,
        "diff_content": diff_content,
        "operation": operation,
        "relative_path": relative_path,
        "occurrences": occurrences,
        "edit_result": edit_result,
    }

    return True, None, result_info


async def edit_file(
    ctx: RunContext[AgentContext], file_path: str, old_string: str, new_string: str, expected_replacements: int = 1
) -> ToolReturn:
    """
    Replace text within a file with advanced context validation.

    Args:
        ctx: The run context containing workspace path
        file_path: The path to the file to modify
        old_string: The exact literal text to replace, including context
        new_string: The exact literal text to replace old_string with
        expected_replacements: Number of replacements expected (default: 1)

    Returns:
        Information about the edit operation and a diff showing the changes
    """
    workspace_path = ctx.deps.workspace_path
    logger.debug(f"Running edit_file with workspace_path: {workspace_path}")

    try:
        # Process the edit operation using our helper function
        success, tool_return, result_info = _process_edit_operation(
            file_path, workspace_path, old_string, new_string, expected_replacements
        )

        # If the processing failed, return the error
        if not success:
            return tool_return

        # Extract the results from processing
        path = result_info["path"]
        new_content = result_info["new_content"]
        diff_content = result_info["diff_content"]
        operation = result_info["operation"]
        relative_path = result_info["relative_path"]
        occurrences = result_info["occurrences"]
        edit_result = result_info["edit_result"]

        # Write the file
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(new_content)

        # Build content for display
        content_lines = [f"## File {operation.capitalize()}: `{relative_path}`"]
        if occurrences > 1:
            content_lines.append(f"{occurrences} replacements made")
        content_lines.append("")
        content_lines.append("### Changes")
        content_lines.append("```diff")
        content_lines.append(diff_content)
        content_lines.append("```")

        display_message = f"Successfully {operation} {relative_path}"
        if occurrences > 1:
            display_message += f" ({occurrences} replacements made)"

        logger.info(display_message)

        return ToolReturn(
            return_value=display_message,
            content=content_lines,
            metadata={"success": True, "edit_result": edit_result.model_dump()},
        )

    except Exception as e:
        logger.error(f"Edit operation failed: {type(e).__name__}: {e!s}")
        error_content = [
            "## Error: Edit Operation Failed",
            f"Failed to edit file `{file_path}`:",
            f"- Error Type: {type(e).__name__}",
            f"- Details: {e!s}",
        ]
        return ToolReturn(
            return_value=f"Error: Edit operation failed: {e!s}",
            content=error_content,
            metadata={"success": False, "error": "execution_failed", "exception": str(e)},
        )
