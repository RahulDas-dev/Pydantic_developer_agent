import logging
from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from pydantic_ai.messages import ToolReturn

from coder_agent.context import AgentContext

from .utils import is_within_root, should_ignore_path

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
                f"- `{result.path}` (" + str(result.size) + " bytes)"
                if result.size is not None
                else f"- `{result.path}`"
                for result in results
                if result.type == "file"
            ]
        )
        content.append("")

    # Add directories section if applicable
    if dirs_count > 0:
        content.append(f"### Directories ({dirs_count}):")
        content.extend([f"- `{result.path}/`" for result in results if result.type == "directory"])

    return summary, content


def _perform_glob_search(base_path: Path | str, pattern: str) -> list[Path]:
    """
    Perform the glob search using Path.glob.

    Args:
        base_path: The base directory path to search from
        pattern: The glob pattern to search for

    Returns:
        A list of matching paths
    """
    search_path = Path(base_path)
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


def _process_matches(matches: list[Path], root_directory: str, include_dirs: bool, max_results: int) -> list[FileMatch]:
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
        if should_ignore_path(match_path, match_path.name):
            logger.debug(f"Ignoring path: {match_path}")
            continue

        file_match = _collect_file_info(match_path, Path(root_directory), include_dirs)
        if file_match:
            results.append(file_match)

        # Check max results limit
        if len(results) >= max_results:
            logger.debug(f"Reached max_results limit of {max_results}")
            break

    # Sort results by path
    results.sort(key=lambda x: x.path)
    return results


async def glob_search(
    ctx: RunContext[AgentContext],
    pattern: str,
    include_dirs: bool = False,
    max_results: int = 1000,
    base_path: str | None = None,
) -> ToolReturn:
    """
    Find files and directories using glob patterns.

    Args:
        pattern: Glob pattern to match files and directories (e.g., '*.py', '**/*.js', 'src/**/test_*.py')
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

    if not pattern:
        return ToolReturn(
            return_value="Pattern parameter is required and cannot be empty",
            content=["# Error: Missing Pattern", "The glob pattern is required and cannot be empty."],
            metadata={"success": False, "error": "empty_pattern"},
        )

    if base_path and not Path(base_path).is_absolute():
        return ToolReturn(
            return_value=f"Error: base_path must be absolute: {base_path}",
            content=["# Error: base_path must be absolute", f"The base path must be absolute: {base_path}"],
            metadata={"success": False, "error": "invalid_base_path"},
        )

    if base_path and (not is_within_root(Path(base_path), Path(workspace_path))):
        return ToolReturn(
            return_value=f"Base path must be within root directory ({workspace_path}): {base_path}",
            content=[
                "# Error: Invalid Base Path",
                f"The base path `{base_path}` must be within the workspace root directory.",
            ],
            metadata={"success": False, "error": "base_path_outside_root"},
        )
    base_path = workspace_path

    if not Path(base_path).exists():
        return ToolReturn(
            return_value=f"Error: base_path does not exist: {base_path}",
            content=["# Error: Invalid Base Path", f"Error: The base path `{base_path}` does not exist."],
            metadata={"success": False, "error": "base_path_not_found"},
        )

    try:
        # Perform glob search and get matches
        matches = _perform_glob_search(base_path, pattern)

        # Process matches into results
        results = _process_matches(matches, workspace_path, include_dirs, max_results)

        files_count = sum(1 for r in results if r.type == "file")
        dirs_count = sum(1 for r in results if r.type == "directory")

        glob_result = GlobResult(
            pattern=pattern,
            base_path=base_path,
            include_dirs=include_dirs,
            matches_found=len(results),
            files_count=files_count,
            dirs_count=dirs_count,
            results_limited=len(results) >= max_results,
            matches=results,
        )

        # Handle empty results case
        if not results:
            logger.info(f"No files found matching pattern '{pattern}'")
            return ToolReturn(
                return_value=f"No files found matching pattern '{pattern}'",
                content=[
                    "## No matches found",
                    f"No files or directories match the pattern `{pattern}` in `{base_path}`.",
                ],
                metadata={"success": True, "glob_result": glob_result.model_dump()},
            )

        # Format successful results
        summary, content = _format_results(results, pattern, max_results)
        logger.info(f"Successfully found {len(results)} matches for '{pattern}'")

        return ToolReturn(
            return_value=summary, content=content, metadata={"success": True, "glob_result": glob_result.model_dump()}
        )

    except Exception as e:
        logger.error(f"Glob search failed: {type(e).__name__}: {e!s}")
        error_content = [
            "## Error: Glob Search Failed",
            f"Failed to search with pattern `{pattern}`",
            f"- Error Type: {type(e).__name__}",
            f"- Details: {e!s}",
        ]
        return ToolReturn(
            return_value=f"Error: Glob search failed: {e!s}",
            content=error_content,
            metadata={"success": False, "error": "glob_failed", "exception": str(e)},
        )
