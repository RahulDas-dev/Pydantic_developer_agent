import logging
from datetime import datetime, timezone
from pathlib import Path

import aiofiles
from pydantic import BaseModel, Field
from pydantic_ai import RunContext, ToolReturn

from agent.context import AgentContext

logger = logging.getLogger(__name__)


class FileInfo(BaseModel):
    """Information about a read file."""

    path: str = Field(description="Absolute path to the file")
    size: int = Field(description="Size of the file in bytes")
    modified: str = Field(description="ISO formatted timestamp of last modification")
    lines_read: int = Field(description="Number of lines read from the file")


async def read_file(
    ctx: RunContext[AgentContext],
    path: str,
    encoding: str = "utf-8",
    start_line: int = 1,
    end_line: int | None = None,
) -> ToolReturn:
    """
    Read the contents of a file.

    Args:
        path: Path to the file to read
        encoding: File encoding (default: utf-8)
        start_line: Start reading from this line (1-based, default: 1)
        end_line: Stop reading at this line (1-based, default: None - read to end)

    Returns:
        The file contents with file metadata
    """
    logger.info(f"Reading file: {path} (encoding: {encoding}, lines: {start_line}-{end_line or 'end'})")

    # Access the workspace path from AgentContext if needed
    workspace_path = ctx.deps.workspace_path
    try:
        # Resolve path
        file_path = Path(path).resolve()
        logger.debug(f"Resolved path: {file_path}")

        # Security check - ensure path exists and is a file
        if not file_path.exists():
            logger.warning(f"File not found: {path}")
            return ToolReturn(return_value=f"File not found: {path}", metadata={"success": False})

        if not file_path.is_file():
            logger.warning(f"Path is not a file: {path}")
            return ToolReturn(return_value=f"Path is not a file: {path}", metadata={"success": False})

        # Read file
        async with aiofiles.open(file_path, encoding=encoding) as f:
            if start_line or end_line:
                # Read specific lines
                lines = await f.readlines()

                start_idx = (start_line - 1) if start_line else 0
                end_idx = end_line if end_line else len(lines)

                selected_lines = lines[start_idx:end_idx]
                content = "".join(selected_lines)
                logger.debug(f"Read lines {start_idx + 1} to {end_idx} from file")
            else:
                # Read entire file
                content = await f.read()
                logger.debug(f"Read entire file, size: {len(content)} bytes")

        # Get file info
        stat = file_path.stat()
        file_info = FileInfo(
            path=str(file_path),
            size=stat.st_size,
            modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            lines_read=len(content.splitlines()) if content else 0,
        )

        # Determine which lines were read
        lines_info = ""
        if start_line or end_line:
            actual_start = start_idx + 1  # Convert to 1-based for display
            actual_end = end_idx
            lines_info = f" (lines {actual_start}-{actual_end})"

        # Brief summary for the return_value
        summary = f"Successfully read file: {path}{lines_info}"

        # Create content for multimodal display
        file_content = [
            f"## File: {path}",
            f"- Size: {file_info.size} bytes",
            f"- Modified: {file_info.modified}",
            f"- Lines Read: {file_info.lines_read}",
        ]
        # Add line range info if applicable
        if start_line or end_line:
            file_content.append(f"- Line Range: {actual_start}-{actual_end}")
        file_content.extend(
            [
                "### Content:",
                content,
            ]
        )

        logger.info(f"Successfully read file: {path}, size: {file_info.size} bytes, lines: {file_info.lines_read}")

        # Return result to the model with structured content for multimodal display
        # and include metadata for the application
        return ToolReturn(
            return_value=summary, content=file_content, metadata={"success": True, "file_info": file_info.model_dump()}
        )

    except UnicodeDecodeError as e:
        # Include line range in error message if specified
        lines_info = ""
        if start_line or end_line:
            lines_info = f" (lines {start_line}-{end_line or 'end'})"
        error_message = f"Failed to decode file with encoding '{encoding}'{lines_info}"
        error_details = f"{e!s}"
        logger.error(f"UnicodeDecodeError: {error_message} - {error_details}")

        error_content = [
            f"## Error Reading File: {path}",
            "- Error Type: UnicodeDecodeError",
            f"- Encoding: {encoding}",
        ]
        # Add line range info if applicable
        if start_line or end_line:
            error_content.append(f"- Line Range: {start_line}-{end_line or 'end'}")
        error_content.append(f"- Details: {error_details}")

        return ToolReturn(
            return_value=error_message,
            content=error_content,
            metadata={"success": False, "error": "UnicodeDecodeError", "details": str(e)},
        )
    except Exception as e:
        # Include line range in error message if specified
        lines_info = ""
        if start_line or end_line:
            lines_info = f" (lines {start_line}-{end_line or 'end'})"
        error_message = f"Failed to read file: {path}{lines_info}"
        error_details = f"{e!s}"
        logger.error(f"Error reading file {path}: {type(e).__name__}: {error_details}")

        error_content = [
            f"## Error Reading File: {path}",
            f"- Error Type: {type(e).__name__}",
        ]
        # Add line range info if applicable
        if start_line or end_line:
            error_content.append(f"- Line Range: {start_line}-{end_line or 'end'}")
        error_content.append(f"- Details: {error_details}")

        return ToolReturn(
            return_value=error_message,
            content=error_content,
            metadata={"success": False, "error": type(e).__name__, "details": str(e)},
        )
