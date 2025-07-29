import fnmatch
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field
from pydantic_ai import RunContext, ToolReturn

from agent.context import AgentContext

logger = logging.getLogger(__name__)


class FileMatch(BaseModel):
    """Information about a file or directory matching a glob pattern."""

    path: str = Field(description="Relative path to the file or directory")
    absolute_path: str = Field(description="Absolute path to the file or directory")
    type: str = Field(description="Type of the match (file or directory)")
    size: Optional[int] = Field(None, description="Size of the file in bytes (None for directories)")
    modified: Optional[float] = Field(None, description="Modification timestamp (None if unavailable)")


class GlobResult(BaseModel):
    """Result of a glob search operation."""

    pattern: str = Field(description="The glob pattern that was searched for")
    base_path: str = Field(description="The base path where the search was performed")
    include_dirs: bool = Field(description="Whether directories were included in the results")
    matches_found: int = Field(0, description="Total number of matches found")
    files_count: int = Field(0, description="Number of files found")
    dirs_count: int = Field(0, description="Number of directories found")
    results_limited: bool = Field(False, description="Whether results were limited by max_results")
    matches: List[FileMatch] = Field(default_factory=list, description="List of file and directory matches")


def _validate_parameters(
    pattern: str, max_results: int
) -> Tuple[bool, Optional[str], Optional[str], Optional[List[str]]]:
    """Validate input parameters and return error information if invalid."""
    if not pattern or not pattern.strip():
        error_message = "Pattern parameter is required and cannot be empty"
        error_content = ["## Error: Missing Pattern", "The glob pattern is required and cannot be empty."]
        return False, error_message, "empty_pattern", error_content

    # Validate max_results
    min_results = 1
    max_results_limit = 10000
    if max_results < min_results:
        logger.warning(f"max_results was less than {min_results}, setting to {min_results}")
        max_results = min_results
    elif max_results > max_results_limit:
        logger.warning(f"max_results was greater than {max_results_limit}, setting to {max_results_limit}")
        max_results = max_results_limit

    return True, None, None, None


def _validate_path(
    base_path: Optional[str], root_directory: Path
) -> Tuple[bool, Optional[Path], Optional[str], Optional[str], Optional[List[str]]]:
    """Validate and resolve the base path, return error information if invalid."""
    # Determine base path for the search
    if base_path:
        base_path_obj = Path(base_path)
        if not base_path_obj.is_absolute():
            # If not absolute, resolve relative to workspace root
            base_path_obj = root_directory / base_path

        base_path_obj = base_path_obj.resolve()

        # Security check: Make sure base_path is within root directory
        try:
            base_path_obj.relative_to(root_directory)
        except ValueError:
            error_message = f"Base path must be within root directory ({root_directory}): {base_path}"
            error_content = [
                "## Error: Invalid Base Path",
                f"The base path `{base_path}` must be within the workspace root directory.",
            ]
            return False, None, error_message, "base_path_outside_root", error_content
    else:
        # Default to root directory
        base_path_obj = root_directory

    # Check if base path exists
    if not base_path_obj.exists():
        error_message = f"Base path does not exist: {base_path_obj}"
        error_content = ["## Error: Base Path Not Found", f"The specified base path `{base_path_obj}` does not exist."]
        return False, None, error_message, "base_path_not_found", error_content

    return True, base_path_obj, None, None, None


def _should_ignore_path(path: Path, name: str) -> bool:
    """Check if a path should be ignored based on common patterns."""
    ignore_patterns = [
        "__pycache__",
        ".git",
        ".svn",
        ".hg",
        "node_modules",
        ".DS_Store",
        ".env",
        "*.pyc",
        "*.pyo",
        "*.pyd",
    ]

    path_str = str(path)

    for pattern in ignore_patterns:
        if pattern.startswith("*"):
            if fnmatch.fnmatch(name, pattern):
                return True
        elif pattern in path_str:
            return True

    return False


def _collect_file_info(match_path: Path, root_directory: Path, include_dirs: bool) -> Optional[FileMatch]:
    """Collect information about a file or directory match."""
    # Check if it's a file or directory
    is_dir = match_path.is_dir()
    is_file = match_path.is_file()

    # Filter directories if not requested
    if is_dir and not include_dirs:
        return None

    # Skip broken symlinks, etc.
    if not is_file and not is_dir:
        return None

    # Calculate relative path
    try:
        relative_path = match_path.relative_to(root_directory)
    except ValueError:
        relative_path = match_path

    # Get file info
    try:
        stat_info = match_path.stat()
        size = stat_info.st_size if is_file else None
        modified = stat_info.st_mtime
    except Exception as e:
        logger.debug(f"Couldn't get stats for {match_path}: {e}")
        size = None
        modified = None

    # Create and return file match info
    return FileMatch(
        path=str(relative_path),
        absolute_path=str(match_path),
        type="directory" if is_dir else "file",
        size=size,
        modified=modified,
    )


def _format_results(results: List[FileMatch], pattern: str, max_results: int) -> Tuple[str, List[str]]:
    """Format the results for display."""
    files_count = sum(1 for r in results if r.type == "file")
    dirs_count = sum(1 for r in results if r.type == "directory")

    # Build summary
    summary = f"Found {len(results)} match(es) for pattern '{pattern}'"
    if len(results) >= max_results:
        summary += f" (limited to {max_results})"

    # Build detailed content with Markdown formatting
    content = [f"## Results for glob pattern: `{pattern}`"]

    # Add summary line
    result_info = f"Found {len(results)} match(es)"
    if len(results) >= max_results:
        result_info += f" (limited to {max_results})"
    content.append(result_info)
    content.append("")

    # Add files section if applicable
    if files_count > 0:
        content.append(f"### Files ({files_count}):")
        content.extend(
            [
                f"- `{result.path}`{' (' + str(result.size) + ' bytes)' if result.size is not None else ''}"
                for result in results
                if result.type == "file"
            ]
        )
        content.append("")

    # Add directories section if applicable
    if dirs_count > 0:
        content.append(f"### Directories ({dirs_count}):")
        content.extend([f"- `{result.path}/`" for result in results if result.type == "directory"])
        content.append("")

    return summary, content


def _perform_glob_search(base_path_obj: Path, pattern: str) -> list[Path]:
    """
    Perform the glob search using Path.glob.

    Args:
        base_path_obj: The base directory path to search from
        pattern: The glob pattern to search for

    Returns:
        A list of matching paths
    """
    search_path = base_path_obj
    logger.debug(f"Searching in path: {search_path} with pattern: {pattern}")

    # Handle recursive patterns
    if "**" in pattern:
        pattern_parts = pattern.split("**", 1)
        if pattern_parts[0]:
            search_path = search_path / pattern_parts[0].rstrip("/\\")
            pattern = "**" + pattern_parts[1]

    # Use Path.glob to perform the actual search
    matches = list(search_path.glob(pattern))
    logger.debug(f"Found {len(matches)} raw matches before filtering")
    return matches


def _process_matches(
    matches: list[Path], root_directory: Path, include_dirs: bool, max_results: int
) -> list[FileMatch]:
    """
    Process and filter glob search matches.

    Args:
        matches: List of path matches from glob search
        root_directory: Root directory for computing relative paths
        include_dirs: Whether to include directories in results
        max_results: Maximum number of results to return

    Returns:
        List of FileMatch objects
    """
    results = []
    for match_path in matches:
        # Skip if path should be ignored
        if _should_ignore_path(match_path, match_path.name):
            logger.debug(f"Ignoring path: {match_path}")
            continue

        file_match = _collect_file_info(match_path, root_directory, include_dirs)
        if file_match:
            results.append(file_match)

            # Check max results limit
            if len(results) >= max_results:
                logger.debug(f"Reached max_results limit of {max_results}")
                break

    # Sort results by path
    results.sort(key=lambda x: x.path)
    return results


def _create_glob_result(
    results: list[FileMatch], pattern: str, base_path_obj: Path, include_dirs: bool, max_results: int
) -> GlobResult:
    """
    Create a GlobResult object from the search results.

    Args:
        results: List of FileMatch objects
        pattern: The search pattern used
        base_path_obj: The base directory path searched
        include_dirs: Whether directories were included
        max_results: Maximum number of results allowed

    Returns:
        GlobResult object with metadata about the search
    """
    # Count files and directories
    files_count = sum(1 for r in results if r.type == "file")
    dirs_count = sum(1 for r in results if r.type == "directory")

    return GlobResult(
        pattern=pattern,
        base_path=str(base_path_obj),
        include_dirs=include_dirs,
        matches_found=len(results),
        files_count=files_count,
        dirs_count=dirs_count,
        results_limited=len(results) >= max_results,
        matches=results,
    )


def _handle_no_results(pattern: str, base_path_obj: Path, glob_result: GlobResult) -> ToolReturn:
    """
    Create a ToolReturn for when no results are found.

    Args:
        pattern: The search pattern used
        base_path_obj: The base directory path searched
        glob_result: The GlobResult object

    Returns:
        ToolReturn with no results message
    """
    logger.info(f"No files found matching pattern '{pattern}'")
    content = ["## No matches found", f"No files or directories match the pattern `{pattern}` in `{base_path_obj}`."]
    return ToolReturn(
        return_value=f"No files found matching pattern '{pattern}'",
        content=content,
        metadata={"success": True, "glob_result": glob_result.model_dump()},
    )


def _handle_error(pattern: str, error: Exception) -> ToolReturn:
    """
    Create a ToolReturn for when an error occurs.

    Args:
        pattern: The search pattern used
        error: The exception that occurred

    Returns:
        ToolReturn with error information
    """
    logger.error(f"Glob search failed: {type(error).__name__}: {error!s}")
    error_content = [
        "## Error: Glob Search Failed",
        f"Failed to search with pattern `{pattern}`",
        f"- Error Type: {type(error).__name__}",
        f"- Details: {error!s}",
    ]
    return ToolReturn(
        return_value=f"Error: Glob search failed: {error!s}",
        content=error_content,
        metadata={"success": False, "error": "glob_failed", "exception": str(error)},
    )


async def glob_search(
    ctx: RunContext[AgentContext],
    pattern: str,
    include_dirs: bool = False,
    max_results: int = 1000,
    base_path: Optional[str] = None,
) -> ToolReturn:
    """
    Find files and directories using glob patterns.

    Args:
        pattern: Glob pattern to match files and directories (e.g., '*.py', '**/*.js')
        include_dirs: Whether to include directories in results (default: False)
        max_results: Maximum number of results to return (default: 1000, min: 1, max: 10000)
        base_path: Base directory to search from (optional, defaults to workspace root)

    Returns:
        Matching files and directories information
    """
    workspace_path = ctx.deps.workspace_path

    logger.debug(f"Running glob search with workspace_path: {workspace_path}")
    logger.info(
        f"Searching with pattern: {pattern} (include_dirs: {include_dirs}, "
        f"max_results: {max_results}, base_path: {base_path})"
    )

    # Validate pattern and max_results
    valid, error_msg, error_code, error_content = _validate_parameters(pattern, max_results)
    if not valid:
        return ToolReturn(
            return_value=error_msg, content=error_content, metadata={"success": False, "error": error_code}
        )

    pattern = pattern.strip()

    # Determine root directory (for security checks)
    root_directory = Path(workspace_path).resolve()

    # Validate and resolve the base path
    valid, base_path_obj, error_msg, error_code, error_content = _validate_path(base_path, root_directory)
    if not valid:
        return ToolReturn(
            return_value=error_msg, content=error_content, metadata={"success": False, "error": error_code}
        )

    try:
        # Perform glob search and get matches
        matches = _perform_glob_search(base_path_obj, pattern)

        # Process matches into results
        results = _process_matches(matches, root_directory, include_dirs, max_results)

        # Create result metadata
        glob_result = _create_glob_result(results, pattern, base_path_obj, include_dirs, max_results)

        # Handle empty results case
        if not results:
            return _handle_no_results(pattern, base_path_obj, glob_result)

        # Format successful results
        summary, content = _format_results(results, pattern, max_results)
        logger.info(f"Successfully found {len(results)} matches for '{pattern}'")

        return ToolReturn(
            return_value=summary, content=content, metadata={"success": True, "glob_result": glob_result.model_dump()}
        )

    except Exception as e:
        return _handle_error(pattern, e)
