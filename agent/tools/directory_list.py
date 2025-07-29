import logging
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field
from pydantic_ai import RunContext, Tool, ToolReturn

from agent.context import AgentContext

logger = logging.getLogger(__name__)


class DirectoryEntry(BaseModel):
    """Information about a directory entry."""

    name: str = Field(description="Name of the file or directory")
    path: str = Field(description="Full path to the file or directory")
    is_dir: bool = Field(description="Whether the entry is a directory")
    depth: int = Field(0, description="Nesting depth in directory structure")
    size: int = Field(0, description="Size in bytes (0 for directories)")


class DirectoryInfo(BaseModel):
    """Information about a directory listing."""

    path: str = Field(description="Absolute path to the directory")
    total_entries: int = Field(0, description="Total number of entries")
    directories: int = Field(0, description="Number of directories")
    files: int = Field(0, description="Number of files")


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
    workspace_path = ctx.deps.workspace_path
    logger.debug(f"Running list_directory with workspace_path: {workspace_path}")
    logger.info(
        f"Listing directory: {path} (show_hidden: {show_hidden}, recursive: {recursive}, max_depth: {max_depth})"
    )

    # Validate max_depth
    min_depth = 1
    max_depth_limit = 10
    if max_depth < min_depth:
        max_depth = min_depth
        logger.warning(f"max_depth was less than {min_depth}, setting to {min_depth}")
    elif max_depth > max_depth_limit:
        max_depth = max_depth_limit
        logger.warning(f"max_depth was greater than {max_depth_limit}, setting to {max_depth_limit}")

    try:
        # Resolve path relative to workspace root if it's not absolute
        if not Path(path).is_absolute():
            dir_path = Path(workspace_path) / path
            logger.debug(f"Resolved relative path using workspace_root: {dir_path}")
        else:
            dir_path = Path(path)

        dir_path = dir_path.resolve()
        logger.debug(f"Resolved path: {dir_path}")

        if not dir_path.exists():
            logger.warning(f"Directory not found: {path}")
            error_content = ["## Error: Directory Not Found", f"The directory `{path}` could not be found."]
            return ToolReturn(
                return_value=f"Directory not found: {path}", content=error_content, metadata={"success": False}
            )

        if not dir_path.is_dir():
            logger.warning(f"Path is not a directory: {path}")
            error_content = ["## Error: Not a Directory", f"The path `{path}` exists but is not a directory."]
            return ToolReturn(
                return_value=f"Path is not a directory: {path}", content=error_content, metadata={"success": False}
            )

        # List directory contents
        entries = await _list_directory_recursive(dir_path, show_hidden, recursive, max_depth, 0)

        # Prepare info for metadata
        dir_info = DirectoryInfo(
            path=str(dir_path),
            total_entries=len(entries),
            directories=sum(1 for e in entries if e.is_dir),
            files=sum(1 for e in entries if not e.is_dir),
        )

        # Format the summary for return_value
        summary = (
            f"Directory: {path} - "
            f"{dir_info.total_entries} items "
            f"({dir_info.directories} directories, {dir_info.files} files)"
        )

        # Format detailed output for content
        output_lines = [f"## Directory listing for: `{path}`"]

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

        logger.info(f"Successfully listed directory: {path} with {dir_info.total_entries} entries")

        # Return the result with both summary and detailed content
        return ToolReturn(
            return_value=summary,
            content=output_lines,
            metadata={"success": True, "directory_info": dir_info.model_dump()},
        )

    except PermissionError as e:
        logger.error(f"Permission error listing directory {path}: {e!s}")
        error_content = ["## Error: Permission Denied", f"Cannot access directory `{path}`: Permission denied."]
        return ToolReturn(
            return_value=f"Permission denied: {path}",
            content=error_content,
            metadata={"success": False, "error": "PermissionError"},
        )
    except Exception as e:
        logger.error(f"Error listing directory {path}: {type(e).__name__}: {e!s}")
        error_content = [
            f"## Error Listing Directory: {path}",
            f"- Error Type: {type(e).__name__}",
            f"- Details: {e!s}",
        ]
        return ToolReturn(
            return_value=f"Failed to list directory: {e!s}",
            content=error_content,
            metadata={"success": False, "error": type(e).__name__, "details": str(e)},
        )


async def _list_directory_recursive(
    path: Path, show_hidden: bool, recursive: bool, max_depth: int, current_depth: int
) -> List[DirectoryEntry]:
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

            # Create entry for this item
            entry = DirectoryEntry(name=item.name, path=str(item), is_dir=item.is_dir(), depth=current_depth, size=size)

            entries.append(entry)

            # Recurse into directories if requested and depth limit not reached
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


# Create a Tool instance that can be used with the Agent
list_directory_tool = Tool(list_directory, takes_ctx=True)


# Example of how to register this tool with an agent:
"""
from pydantic_ai import Agent
from agent.tools.list_directory import list_directory_tool

# Create an agent with the list_directory tool
agent = Agent(
    'your-model-name',
    tools=[list_directory_tool],
    system_prompt="You are an assistant that can browse the file system."
)
"""
