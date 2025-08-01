"""
Grep tool for searching text patterns in files.
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiofiles
from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from pydantic_ai.messages import ToolReturn

from lib.context import AgentContext

from .utils import is_within_root, should_ignore_path

# Constants
MAX_RESULTS_LIMIT = 1000
DEFAULT_CONTEXT_LINES = 2
DEFAULT_MAX_RESULTS = 100
MAX_PREVIEW_LENGTH = 100

logger = logging.getLogger(__name__)


class GrepMatch(BaseModel):
    """A match found by grep."""

    file_path: str = Field(description="Path to the file where the match was found")
    relative_path: str = Field(description="Path relative to workspace root")
    line_number: int = Field(description="Line number of the match")
    line: str = Field(description="Matched line content")
    context_lines: list[str] = Field(description="Context lines surrounding the match")


class GrepResult(BaseModel):
    """Result of a grep search operation."""

    pattern: str = Field(description="Search pattern used")
    files_searched: int = Field(description="Number of files searched")
    files_with_matches: int = Field(description="Number of files with matches")
    matches_found: int = Field(description="Total number of matches found")
    results_limited: bool = Field(description="Whether results were limited")
    matches: list[GrepMatch] = Field(description="list of matches found")


@dataclass
class FormatData:
    """Helper class to pass formatting data."""

    pattern: str
    all_matches: list[dict[str, Any]]
    files_dict: dict[str, list[dict[str, Any]]]
    files_with_matches: int
    max_results: int
    workspace_path: str


@dataclass
class GrepParams:
    """Parameters for the grep function."""

    pattern: str
    path: str | None = None
    include: str | None = None
    max_results: int = DEFAULT_MAX_RESULTS
    context_lines: int = DEFAULT_CONTEXT_LINES


@dataclass
class GrepError:
    """Represents an error in the grep tool."""

    error_msg: str
    error_code: str
    exception: str | None = None

    def to_tool_return(self) -> ToolReturn:
        """Convert to a ToolReturn."""
        return ToolReturn(
            return_value=self.error_msg,
            content=f"## Error\n\n{self.error_msg}",
            metadata={
                "success": False,
                "error": self.error_code,
                **({"exception": self.exception} if self.exception else {}),
            },
        )


async def _search_file(file_path: Path, pattern: re.Pattern, context_lines: int) -> list[dict[str, Any]]:
    """Search for pattern in a single file."""
    try:
        async with aiofiles.open(file_path, encoding="utf-8", errors="ignore") as f:
            content = await f.read()

        lines = content.splitlines()
        matches = []

        for line_num, line in enumerate(lines, 1):
            if pattern.search(line):
                # Get context lines
                start_line = max(0, line_num - 1 - context_lines)
                end_line = min(len(lines), line_num + context_lines)

                context = []
                for i in range(start_line, end_line):
                    prefix = ">>> " if i == line_num - 1 else "    "
                    context.append(f"{prefix}{i + 1:4d}: {lines[i]}")

                matches.append(
                    {
                        "file_path": str(file_path),
                        "line_number": line_num,
                        "line": line,
                        "context": context,
                    }
                )

        return matches

    except Exception as e:
        logger.debug(f"Error searching file {file_path}: {e}")
        return []


def _get_files_by_pattern(search_path: Path, include_pattern: str) -> list[Path]:
    """Find files matching a glob pattern."""
    # If the include pattern is a full path, use it as is
    if Path(include_pattern).is_absolute():
        path_obj = Path(include_pattern)
        if (
            is_within_root(path_obj, search_path)
            and path_obj.is_file()
            and not should_ignore_path(path_obj, path_obj.name)
        ):
            return [path_obj]
        return []

    # Try direct glob
    pattern_matches = list(search_path.glob(include_pattern))
    result_files = [path for path in pattern_matches if path.is_file() and not should_ignore_path(path, path.name)]

    # Try recursive glob if no files found and not already recursive
    if not result_files and "**" not in include_pattern:
        recursive_matches = list(search_path.glob(f"**/{include_pattern}"))
        result_files = [
            path for path in recursive_matches if path.is_file() and not should_ignore_path(path, path.name)
        ]

    return result_files


def _get_all_files(search_path: Path) -> list[Path]:
    """Get all files recursively from a directory."""
    try:
        return [
            file_path
            for file_path in search_path.rglob("*")
            if file_path.is_file() and not should_ignore_path(file_path, file_path.name)
        ]
    except Exception as e:
        logger.warning(f"Error traversing directory {search_path}: {e}")
        return []


def _get_files_to_search(search_path: Path, include_pattern: str | None = None) -> list[Path]:
    """Get list of files to search based on include pattern."""
    if not include_pattern:
        # Search all files recursively
        return _get_all_files(search_path)

    # Use glob pattern to find matching files
    try:
        return _get_files_by_pattern(search_path, include_pattern)
    except Exception as e:
        logger.warning(f"Error with pattern '{include_pattern}': {e}")
        return []


async def _process_search_results(
    files_to_search: list[Path], regex: re.Pattern, context_lines: int, max_results: int
) -> dict[str, Any]:
    """Process search results in batches.

    Returns:
        dict with search results
    """
    all_matches = []
    files_with_matches = 0

    # Process files in batches to avoid overwhelming the system
    batch_size = 50
    for i in range(0, len(files_to_search), batch_size):
        batch = files_to_search[i : i + batch_size]
        search_tasks = [_search_file(file_path, regex, context_lines) for file_path in batch]
        batch_results = await asyncio.gather(*search_tasks)

        for matches in batch_results:
            if matches:
                all_matches.extend(matches)
                files_with_matches += 1

                # Check if we've hit the max results limit
                if len(all_matches) >= max_results:
                    all_matches = all_matches[:max_results]
                    break

        if len(all_matches) >= max_results:
            break

    return {"all_matches": all_matches, "files_with_matches": files_with_matches}


def _format_no_results(pattern: str, search_path: Path, include: str | None, files_searched: int) -> ToolReturn:
    """Format response when no results are found."""
    if files_searched == 0:
        message = "No files found matching the search criteria."
    else:
        message = f"No matches found for pattern '{pattern}' in {files_searched} files."

    return ToolReturn(
        return_value=message,
        content=f"## Search Results\n\n{message}",
        metadata={
            "success": True,
            "pattern": pattern,
            "search_path": str(search_path),
            "include_pattern": include,
            "files_searched": files_searched,
            "matches_found": 0,
        },
    )


def _prepare_structured_matches(
    all_matches: list[dict[str, Any]], workspace_path: str
) -> tuple[list[GrepMatch], dict[str, list[dict[str, Any]]]]:
    """Convert matches to structured format and group by file.

    Returns:
        tuple with (structured_matches, files_dict)
    """
    # Group matches by file
    files_dict = {}
    for match in all_matches:
        file_path = match["file_path"]
        if file_path not in files_dict:
            files_dict[file_path] = []
        files_dict[file_path].append(match)

    # Convert to structured matches with relative paths
    root_directory = Path(workspace_path).resolve()
    structured_matches = []

    for file_path, matches in files_dict.items():
        path_obj = Path(file_path)
        try:
            relative_path = path_obj.relative_to(root_directory)
        except ValueError:
            relative_path = path_obj

        structured_matches.extend(
            [
                GrepMatch(
                    file_path=file_path,
                    relative_path=str(relative_path),
                    line_number=match["line_number"],
                    line=match["line"],
                    context_lines=match["context"],
                )
                for match in matches
            ]
        )

    return structured_matches, files_dict


def _format_content_lines(data: FormatData) -> list[str]:
    """Format content lines for the output.

    Returns:
        list of content lines
    """
    content_lines = [
        f"## Search Results: `{data.pattern}`",
        "",
        f"Found {len(data.all_matches)} match(es) in {data.files_with_matches} file(s)",
    ]

    if len(data.all_matches) >= data.max_results:
        content_lines.append(f"_Results limited to {data.max_results} matches_")

    content_lines.append("")

    # Group results by file for readability
    root_directory = Path(data.workspace_path).resolve()
    for file_path, matches in data.files_dict.items():
        path_obj = Path(file_path)
        try:
            relative_path = path_obj.relative_to(root_directory)
        except ValueError:
            relative_path = path_obj

        content_lines.append(f"### `{relative_path}`")
        content_lines.append("")

        for match in matches:
            content_lines.append(f"**Line {match['line_number']}**:")
            content_lines.append("```")
            content_lines.extend(match["context"])
            content_lines.append("```")
            content_lines.append("")

    return content_lines


async def grep(
    ctx: RunContext[AgentContext],
    params: GrepParams,
) -> ToolReturn:
    """
    Searches for a regex pattern in files within the specified directory.

    Args:
        ctx: The run context
        pattern: The regular expression pattern to search for
        path: Directory to search within, relative to workspace root or absolute
        include: Glob pattern to filter which files are searched (e.g. '*.py')
        max_results: Maximum number of matches to return (default: 100)
        context_lines: Number of context lines to show around each match (default: 2)

    Returns:
        A ToolReturn containing search results and metadata
    """
    logger.info(f"Running grep with pattern '{params.pattern}'")
    workspace_path = ctx.deps.workspace_path

    try:
        # Validate parameters
        if not params.pattern.strip():
            return GrepError(error_msg="Search pattern cannot be empty", error_code="empty_pattern").to_tool_return()

        # Limit max results to a reasonable value
        max_results = min(max(1, params.max_results), MAX_RESULTS_LIMIT)

        # Resolve search path
        search_path = Path(workspace_path)
        if params.path:
            custom_path = Path(params.path)
            if not custom_path.is_absolute():
                custom_path = search_path / custom_path

            # Security check: ensure path is within workspace
            try:
                custom_path.resolve().relative_to(search_path.resolve())
                search_path = custom_path
            except ValueError:
                return GrepError(
                    error_msg=f"Search path must be within workspace: {params.path}",
                    error_code="path_outside_workspace",
                ).to_tool_return()

        # Verify path exists and is a directory
        if not search_path.exists():
            error_msg = f"Search path does not exist: {search_path}"
            return ToolReturn(
                return_value=error_msg,
                content=f"## Error\n\n{error_msg}",
                metadata={"success": False, "error": "path_not_found"},
            )

        if not search_path.is_dir():
            error_msg = f"Search path is not a directory: {search_path}"
            return ToolReturn(
                return_value=error_msg,
                content=f"## Error\n\n{error_msg}",
                metadata={"success": False, "error": "not_directory"},
            )

        # Compile regex pattern
        try:
            regex = re.compile(params.pattern, re.MULTILINE)
        except re.error as e:
            return GrepError(error_msg=f"Invalid regular expression: {e}", error_code="invalid_regex").to_tool_return()

        # Get files to search
        files_to_search = _get_files_to_search(search_path, params.include)

        if not files_to_search:
            return _format_no_results(params.pattern, search_path, params.include, 0)

        # Search files
        results = await _process_search_results(files_to_search, regex, params.context_lines, max_results)
        all_matches = results["all_matches"]
        files_with_matches = results["files_with_matches"]

        # Format results
        if not all_matches:
            return _format_no_results(params.pattern, search_path, params.include, len(files_to_search))

        # Convert to structured matches and group by file
        structured_matches, files_dict = _prepare_structured_matches(all_matches, workspace_path)

        # Generate readable output
        format_data = FormatData(
            pattern=params.pattern,
            all_matches=all_matches,
            files_dict=files_dict,
            files_with_matches=files_with_matches,
            max_results=max_results,
            workspace_path=workspace_path,
        )
        content_lines = _format_content_lines(format_data)

        # Create result model
        result = GrepResult(
            pattern=params.pattern,
            files_searched=len(files_to_search),
            files_with_matches=files_with_matches,
            matches_found=len(all_matches),
            results_limited=len(all_matches) >= max_results,
            matches=structured_matches,
        )

        return ToolReturn(
            return_value=result.model_dump(),
            content="\n".join(content_lines),
            metadata={
                "success": True,
                "pattern": params.pattern,
                "search_path": str(search_path),
                "include_pattern": params.include,
                "files_searched": len(files_to_search),
                "files_with_matches": files_with_matches,
                "matches_found": len(all_matches),
                "results_limited": len(all_matches) >= max_results,
            },
        )

    except Exception as e:
        logger.error(f"Grep tool execution failed: {e}")
        return GrepError(
            error_msg=f"Search operation failed: {e}", error_code="execution_failed", exception=str(e)
        ).to_tool_return()
