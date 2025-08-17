import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic_ai import RunContext
from pydantic_ai.messages import ToolReturn

from lib.agents.context import AgentContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DirectoryEntry:
    """Information about a directory entry."""

    name: str
    path: str
    is_dir: bool
    depth: int
    size: int


@dataclass(frozen=True, slots=True)
class DirectoryInfo:
    """Information about a directory listing."""

    path: str
    total_entries: int = 0
    directories: int = 0
    files: int = 0


async def list_directory(
    ctx: RunContext[AgentContext],
    path: str = ".",
    show_hidden: bool = False,
    recursive: bool = False,
    max_depth: int = 3,
) -> ToolReturn:
    """
    List the contents of a directory.
    Args:
        path: Path to the directory to list (default: ".")
        show_hidden: Include hidden files and directories (default: False)
        recursive: List directories recursively (default: False)
        max_depth: Maximum depth to list (default: 3, min: 1, max: 10)
    Returns:
        Formatted directory listing with information about files and directories
    """
    # Get the agent context which might be used for workspace paths
    workspace_path = str(ctx.deps.workspace_path)
    logger.debug(f"Running list_directory with workspace_path: {workspace_path}")
    logger.info(
        f"Listing directory: {path} (show_hidden: {show_hidden}, recursive: {recursive}, max_depth: {max_depth})"
    )
    # Validate max_depth
    min_depth_limit = 1
    max_depth_limit = 10
    max_depth = max(min_depth_limit, min(max_depth, max_depth_limit))
    dir_path = Path(path).resolve()
    if not dir_path.exists():
        logger.warning(f"Directory not found: {path}")
        return ToolReturn(
            return_value=f"Directory not found: {path}",
            content=["# Error: Directory Not Found", f"The directory `{path}` could not be found."],
            metadata={"success": False},
        )
    if not dir_path.is_dir():
        logger.warning(f"Path is not a directory: {path}")
        return ToolReturn(
            return_value=f"Path is not a directory: {path}",
            content=["# Error: Not a Directory", f"The path `{path}` exists but is not a directory."],
            metadata={"success": False},
        )

    try:
        # Resolve path relative to workspace root if it's not absolute
        entries = await _list_directory_recursive(dir_path, show_hidden, recursive, max_depth, 0)
        # Prepare info for metadata
        dir_info = DirectoryInfo(
            path=str(dir_path),
            total_entries=len(entries),
            directories=sum(1 for e in entries if e.is_dir),
            files=sum(1 for e in entries if not e.is_dir),
        )
        # Format the summary for return_value
        summary = {
            f"Directory: {path} - ",
            f"{dir_info.total_entries} items ",
            f"({dir_info.directories} directories, {dir_info.files} files)",
        }
        # Format detailed output for content
        output_lines = [f"# Directory listing for: `{path}`"]
        # Add stats line
        output_lines.append(
            f"- Total: {dir_info.total_entries} items ({dir_info.directories} directories, {dir_info.files} files)"
        )
        output_lines.append("")  # Empty line before listing
        # Add entries
        for entry in entries:
            indent = "  " * entry.depth
            type_indicator = "/" if entry.is_dir else ""
            size_info = f" ({entry.size} bytes)" if not entry.is_dir else ""
            output_lines.append(f"{indent}{entry.name}{type_indicator}{size_info}")
        logger.info(f"Successfully listed directory: {path} with {dir_info.total_entries} items")
        # Return the result with both summary and detailed content
        return ToolReturn(
            return_value=summary,
            content=output_lines,
            metadata={"success": True, "directory_info": asdict(dir_info)},
        )
    except PermissionError as e:
        logger.error(f"Permission error listing directory {path}: {e}")
        return ToolReturn(
            return_value=f"Permission denied: {path}",
            content=[
                "# Error: Permission Denied",
                f"Cannot access directory `{path}`:",
                f"- Error Type: {type(e).__name__}",
                f"- Details: {e}",
            ],
            metadata={"success": False, "error": "PermissionError"},
        )
    except Exception as e:
        logger.error(f"Error listing directory {path}: {type(e).__name__}: {e}")
        return ToolReturn(
            return_value=f"Failed to list directory: {e}",
            content=[
                f"# Error Listing Directory: {path}",
                f"- Error Type: {type(e).__name__}",
                f"- Details: {e}",
            ],
            metadata={"success": False, "error": type(e).__name__, "details": str(e)},
        )


async def _list_directory_recursive(
    path: Path, show_hidden: bool, recursive: bool, max_depth: int, current_depth: int
) -> list[DirectoryEntry]:
    """List directory contents recursively."""
    entries = []
    try:
        for item in path.iterdir():
            # Skip hidden files if not requested
            if not show_hidden and item.name.startswith("."):
                continue
            try:
                # Get file size if it's a file
                size = item.stat().st_size if item.is_file() else 0
            except OSError:
                size = 0
            entry = DirectoryEntry(name=item.name, path=str(item), is_dir=item.is_dir(), depth=current_depth, size=size)
            entries.append(entry)
            if recursive and item.is_dir() and current_depth < max_depth:
                try:
                    sub_entries = await _list_directory_recursive(
                        item, show_hidden, recursive, max_depth, current_depth + 1
                    )
                    entries.extend(sub_entries)
                except PermissionError:
                    # Skip directories we can't access
                    logger.debug(f"Permission denied for subdirectory: {item}")
    except PermissionError:
        # Handle permission denied for the main directory
        logger.debug(f"Permission denied listing directory: {path}")
    return entries
