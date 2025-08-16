import difflib
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import aiofiles
from pydantic_ai import RunContext
from pydantic_ai.messages import ToolReturn

from lib.context import AgentContext

from .utils import read_file_content

logger = logging.getLogger(__name__)

MAX_PREVIEW_LENGTH = 100
MAX_CONTENT_PREVIEW = 1000


@dataclass(frozen=True, slots=True)
class EditResult:
    """Result of a file edit operation."""

    file_path: str
    relative_path: str
    operation: str
    replacements_made: int
    is_new_file: bool


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


async def edit_file(
    ctx: RunContext[AgentContext],
    file_path: str,
    old_string: str,
    new_string: str,
    expected_replacements: int = 1,
) -> ToolReturn:
    """
    Replace text within a file with advanced context validation.

    Args:
        file_path: Absolute path to file (must be inside workspace).
        old_string: The exact literal text to replace. Include at least 3 lines of context BEFORE and AFTER the target text, matching whitespace and indentation precisely. This string must uniquely identify the location to be changed.
        new_string: The exact literal text to replace old_string with Ensure the resulting code is correct and idiomatic
        expected_replacements: Number of replacements expected. Defaults to 1. Use when you want to replace multiple occurrences.
    """
    workspace_path = ctx.deps.workspace_path
    logger.debug(f"Running edit_file with workspace_path: {workspace_path}")

    if not Path(file_path).is_absolute():
        logger.debug(f"File path is not absolute, resolving relative to workspace: {file_path}")
        return ToolReturn(
            return_value=f"Error: file_path must be absolute: {file_path}",
            content=[
                "## Error: file_path is invalid",
                f"- Error: file_path must be absolute: {file_path}",
            ],
            metadata={"success": False, "error": "invalid_file_path"},
        )

    if not Path(file_path).is_relative_to(Path(workspace_path)):
        return ToolReturn(
            return_value=f"File path must be within workspace directory ({workspace_path}): {file_path}",
            content=[
                "## Error: Invalid File Path",
                f"- The file `{file_path}` must be within the workspace directory.",
            ],
            metadata={"success": False, "error": "base_path_outside_root"},
        )

    if Path(file_path).is_dir():
        return ToolReturn(
            return_value=f"Error: file_path must be a file, not a directory: {file_path}",
            content=[
                "## Error: file_path is invalid",
                f"- Error: file_path must be a file, not a directory: {file_path}",
            ],
            metadata={"success": False, "error": "invalid_file_path"},
        )

    current_content = None
    new_content = None
    is_new_file = False

    if Path(file_path).is_file():
        current_content, error = await read_file_content(Path(file_path))
        if error is not None:
            return ToolReturn(
                return_value=f"Could not read file: {error!s}",
                content=[
                    "## Error: File Read Failed",
                    f"- Could not read file '{file_path}':",
                    f"- Error: {error!s}",
                ],
                metadata={"success": False, "error": "file_read_failed"},
            )

    if current_content is None:
        return ToolReturn(
            return_value="No content found",
            content=[
                "## Error: File Read Failed",
                f"- Could not read file '{file_path}':",
            ],
            metadata={"success": False, "error": "no_content_found"},
        )

    if not old_string:
        return ToolReturn(
            return_value="Cannot replace empty string in existing file",
            content=[
                "## Error: Invalid Replacement",
                "- Cannot replace an empty string in an existing file.",
            ],
            metadata={"success": False, "error": "empty_string_replacement"},
        )

    if not new_string:
        return ToolReturn(
            return_value="new_string parameter is required",
            content=[
                "## Error: Missing Parameter",
                "- The new_string parameter is required for file editing.",
            ],
            metadata={"success": False, "error": "missing_new_string"},
        )

    no_of_occurrences = current_content.count(old_string)
    if no_of_occurrences == 0:
        preview = f"{old_string[:MAX_PREVIEW_LENGTH]}..." if len(old_string) > MAX_PREVIEW_LENGTH else old_string
        return ToolReturn(
            return_value=f"String not found in file: {preview!r}",
            content=[
                "## Error: String Not Found",
                "- The string to replace was not found in the file.",
                "- Ensure you've included enough context (3+ lines before/after target text).",
                "with exact White space and indentation.",
            ],
            metadata={"success": False, "error": "no_match_string_found"},
        )

    new_content = current_content.replace(old_string, new_string)
    if no_of_occurrences != expected_replacements:
        return ToolReturn(
            return_value=(
                f"Expected {expected_replacements} replacement(s), but found {no_of_occurrences} occurrence(s)."
            ),
            content=[
                "## Error: Unexpected Replacement Count",
                f"- Expected {expected_replacements} replacement(s), but found {no_of_occurrences} occurrence(s) of the string.",
                "- Use a more specific context to target the exact location to replace.",
            ],
            metadata={
                "success": False,
                "error": "replacement_count_mismatch",
                "expected": expected_replacements,
                "found": no_of_occurrences,
            },
        )

    if not Path(file_path).exists():
        if old_string:
            return ToolReturn(
                return_value="File does not exist and old_string is not empty",
                content=[
                    "## Error: File Not Found",
                    "- The file does not exist, but old_string is not empty.",
                    "- To create a new file, use an empty old_string.",
                ],
                metadata={"success": False, "error": "string_replacement_on_non_existing_file"},
            )
        is_new_file = True
        new_content = old_string

    try:
        diff_content = _generate_diff(current_content, new_content, Path(file_path))
        operation = "created" if is_new_file else "modified"
        relative_path = Path(file_path).relative_to(workspace_path)

        edit_result = EditResult(
            file_path=str(file_path),
            relative_path=str(relative_path),
            operation=operation,
            replacements_made=no_of_occurrences,
            is_new_file=is_new_file,
        )

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(new_content)

        content_lines = [f"## File {operation.capitalize()}: {relative_path}"]
        if no_of_occurrences > 1:
            content_lines.append(f"{no_of_occurrences} replacements made")
        content_lines.append("")
        content_lines.append("### Changes")
        content_lines.append("```diff")
        content_lines.append(diff_content)
        content_lines.append("```")

        display_message = f"Successfully {operation} {relative_path}"
        if no_of_occurrences > 1:
            display_message += f" ({no_of_occurrences} replacements made)"

        return ToolReturn(
            return_value=display_message,
            content=content_lines,
            metadata={"success": True, "edit_result": asdict(edit_result)},
        )

    except Exception as e:
        logger.error(f"Edit operation failed: {type(e).__name__}: {e!s}")
        return ToolReturn(
            return_value="Edit operation failed",
            content=[
                "## Error: Edit Operation Failed",
                f"- Failed to edit file '{file_path}':",
                f"- Error Type: {type(e).__name__}",
                f"- Details: {e!s}",
            ],
            metadata={"success": False, "error": type(e).__name__, "details": str(e)},
        )
