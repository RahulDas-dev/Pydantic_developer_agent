import logging
from datetime import datetime, timezone
from pathlib import Path

import aiofiles
from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from pydantic_ai.messages import ToolReturn

from coder_agent.context import AgentContext

logger = logging.getLogger(__name__)


class FileInfo(BaseModel):
    """Information about a file that was written."""

    path: str = Field(description="Absolute path to the file")
    size: int = Field(description="Size of the file in bytes")
    lines_written: int = Field(description="Number of lines written to the file")
    created: str = Field(description="ISO timestamp when the file was created/modified")


async def write_file(
    ctx: RunContext[AgentContext],
    path: str,
    content: str,
    encoding: str = "utf-8",
    create_dirs: bool = True,
) -> ToolReturn:
    """
    Write content to a file.

    Args:
        path: Path to the file to write
        content: Content to write to the file
        encoding: File encoding (default: utf-8)
        create_dirs: Create parent directories if they don't exist (default: True)

    Returns:
        Information about the file that was written
    """
    workspace_path = ctx.deps.workspace_path
    logger.debug(f"Running write_file workspace_path: {workspace_path}")

    # Log the operation (without the full content for privacy/size reasons)
    preview_length = 50  # Maximum characters to show in preview
    content_preview = content[:preview_length] + "..." if len(content) > preview_length else content
    logger.info(
        f"Writing to file: {path} (encoding: {encoding}, create_dirs: {create_dirs}) Content preview: {content_preview}"
    )

    try:
        # Resolve path relative to workspace root if it's not absolute
        if not Path(path).is_absolute():
            file_path = Path(workspace_path) / path
            logger.debug(f"Resolved relative path using workspace_root: {file_path}")
        else:
            file_path = Path(path)

        file_path = file_path.resolve()
        logger.debug(f"Resolved path: {file_path}")

        # Create parent directories if needed
        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created parent directories: {file_path.parent}")

        # Write file atomically
        temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        logger.debug(f"Using temporary file for atomic write: {temp_path}")

        async with aiofiles.open(temp_path, "w", encoding=encoding) as f:
            await f.write(content)

        # Atomic move
        temp_path.replace(file_path)
        logger.debug(f"Completed atomic move from {temp_path} to {file_path}")

        # Get file info
        try:
            stat = file_path.stat()
            file_info = FileInfo(
                path=str(file_path),
                size=stat.st_size,
                lines_written=len(content.splitlines()),
                created=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            )
            logger.debug(f"File info: {file_info.model_dump()}")
        except Exception as info_err:
            logger.warning(f"Error getting file stats: {info_err}")
            file_info = FileInfo(
                path=str(file_path),
                size=-1,  # Unknown size
                lines_written=len(content.splitlines()),
                created=datetime.now(timezone.utc).isoformat(),
            )

        # Build successful result
        chars_written = len(content)
        lines_written = len(content.splitlines())
        bytes_written = file_info.size

        summary = f"Successfully wrote {chars_written} characters ({lines_written} lines) to {path}"

        # Prepare detailed content with Markdown
        content_output = [
            "## File written successfully",
            f"- **Path**: `{file_path}`",
            f"- **Size**: {bytes_written} bytes",
            f"- **Lines**: {lines_written}",
            f"- **Characters**: {chars_written}",
            f"- **Encoding**: {encoding}",
            f"- **Timestamp**: {file_info.created}",
        ]

        logger.info(f"Successfully wrote {chars_written} characters to {path}")

        return ToolReturn(
            return_value=summary,
            content=content_output,
            metadata={"success": True, "file_info": file_info.model_dump()},
        )

    except PermissionError as e:
        logger.error(f"Permission error writing to file {path}: {e}")
        return ToolReturn(
            return_value=f"Permission denied writing to file: {path}",
            content=["## Error: Permission Denied", f"Cannot write to file `{path}`: Permission denied."],
            metadata={"success": False, "error": "PermissionError"},
        )

    except FileNotFoundError as e:
        logger.error(f"File not found error writing to {path}: {e}")
        return ToolReturn(
            return_value=f"Invalid file location: {path}",
            content=[
                "## Error: File Location Invalid",
                f"Cannot write to file `{path}`: Directory does not exist and",
                "create_dirs is disabled or location is invalid.",
            ],
            metadata={"success": False, "error": "FileNotFoundError"},
        )

    except Exception as e:
        logger.error(f"Error writing to file {path}: {type(e).__name__}: {e}")
        return ToolReturn(
            return_value=f"Failed to write file: {e}",
            content=[f"## Error Writing File: {path}", f"- Error Type: {type(e).__name__}", f"- Details: {e}"],
            metadata={"success": False, "error": type(e).__name__, "details": str(e)},
        )
