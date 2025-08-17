import logging
from datetime import datetime, timezone
from pathlib import Path

import aiofiles
from attr import dataclass
from pydantic_ai import RunContext
from pydantic_ai.messages import ToolReturn

from lib.agents.context import AgentContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FileInfo:
    """Information about a read file."""

    path: str
    size: int
    modified: str
    lines_read: int


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
    file_path = Path(path).resolve()

    if not file_path.exists():
        logger.warning(f"File not found: {path}")
        return ToolReturn(return_value=f"File not found: {path}", metadata={"success": False})

    if not file_path.is_file():
        logger.warning(f"Path is not a file: {path}")
        return ToolReturn(return_value=f"Path is not a file: {path}", metadata={"success": False})

    workspace_path = ctx.deps.workspace_path

    try:
        async with aiofiles.open(file_path, encoding=encoding) as f:
            lines = await f.readlines()
            start_idx = (start_line - 1) if start_line else 0
            end_idx = end_line if end_line else len(lines)
            selected_lines = lines[start_idx:end_idx]
            content = "".join(selected_lines)
            lines_info = f" (lines {start_idx + 1}-{end_idx})"
            logger.debug(f"Read lines {start_idx + 1} to {end_idx} from file")

        stat = file_path.stat()
        file_info = FileInfo(
            path=str(file_path),
            size=stat.st_size,
            modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            lines_read=len(content.splitlines()) if content else 0,
        )

        summary = f"Successfully read file: {path} {lines_info}"

        file_content = [
            f"## File: {path}",
            f"- Size: {file_info.size} bytes",
            f"- Modified: {file_info.modified}",
            f"- Lines Read: {file_info.lines_read}",
        ]

        if end_line is not None:
            file_content.append(f"-- Line Range: {start_idx + 1}-{end_idx}")

        file_content.extend(["### Content:", content])

        logger.info(summary)

        return ToolReturn(
            return_value=summary,
            content=file_content,
            metadata={"success": True, "file_info": file_info},
        )

    except UnicodeDecodeError as e:
        lines_info = ""
        if start_line or end_line:
            lines_info = f" (lines {start_line}-{end_line or 'end'})"

        error_message = f"Failed to decode file with encoding '{encoding}'{lines_info}"
        error_details = f"{e!s}"
        logger.error(f"UnicodeDecodeError: {error_message} -> {error_details}")

        error_content = [
            f"## Error Reading File: {path}",
            "- Error Type: UnicodeDecodeError",
            f"- Encoding: {encoding}",
        ]

        if end_line is not None:
            error_content.append(f"- Line Range: {start_line}-{end_line or 'end'}")

        error_content.append(f"- Details: {error_details}")

        return ToolReturn(
            return_value=error_message,
            content=error_content,
            metadata={"success": False, "error": "UnicodeDecodeError", "details": error_details},
        )

    except Exception as e:
        lines_info = ""
        if start_line or end_line:
            lines_info = f" (lines {start_line}-{end_line or 'end'})"

        error_message = f"Failed to read file{lines_info}: {type(e).__name__}"
        error_details = f"{e!s}"
        logger.error(f"Error reading file {path}: {error_details}")

        error_content = [
            f"## Error Reading File: {path}",
            f"- Error Type: {type(e).__name__}",
        ]

        if end_line is not None:
            error_content.append(f"- Line Range: {start_line}-{end_line or 'end'}")

        error_content.append(f"- Details: {error_details}")

        return ToolReturn(
            return_value=error_message,
            content=error_content,
            metadata={"success": False, "error": type(e).__name__, "details": error_details},
        )
