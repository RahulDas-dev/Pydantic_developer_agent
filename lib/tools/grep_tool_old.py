import asyncio
import fnmatch
import glob as glob_module
import logging
import re
from pathlib import Path
from typing import Any

import aiofiles

from lib.base_types import CancellationToken

from .base import BaseTool, ToolExecutionContext, ToolExecutionResult, ToolParameter, ToolSchema

logger = logging.getLogger("tools")


class GrepTool(BaseTool):
    """Advanced text search tool with regex support and file filtering."""

    def __init__(self, root_directory: str | None = None):
        super().__init__(
            name="grep",
            description="Searches for a regular expression pattern within the content of files "
            "in a specified directory. Can filter files by glob patterns.",
        )
        self.root_directory = Path(root_directory) if root_directory else Path.cwd()
        self.root_directory = self.root_directory.resolve()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="pattern",
                    data_type="string",
                    description="The regular expression pattern to search for within file contents ",
                    required=True,
                ),
                ToolParameter(
                    name="path",
                    data_type="string",
                    description="Optional: The absolute path to the directory to search within. "
                    "If omitted, searches the current working directory.",
                ),
                ToolParameter(
                    name="include",
                    data_type="string",
                    description="Optional: A glob pattern to filter which files are searched "
                    "(e.g., '*.js', '*.{py,pyx}', 'src/**'). If omitted, searches all files.",
                ),
                ToolParameter(
                    name="max_results",
                    data_type="integer",
                    description="Maximum number of matches to return (default: 100)",
                    default=100,
                    extra={
                        "minimum": 1,
                        "maximum": 1000,
                    },
                ),
                ToolParameter(
                    name="context_lines",
                    data_type="integer",
                    description="Number of context lines to show before and after each match (default: 2)",
                    default=2,
                    extra={
                        "minimum": 0,
                        "maximum": 10,
                    },
                ),
            ],
            required=["pattern"],
        )

    def _is_within_root(self, path: Path) -> bool:
        """Check if path is within the root directory."""
        try:
            path.resolve().relative_to(self.root_directory)
            return True
        except ValueError:
            return False

    def _should_ignore_file(self, file_path: Path) -> bool:
        """Check if file should be ignored based on common patterns."""
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
            "*.so",
            "*.dll",
            "*.exe",
            "*.bin",
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.gif",
            "*.pdf",
            "*.zip",
            "*.tar",
            "*.gz",
        ]

        file_str = str(file_path)
        name = file_path.name

        for pattern in ignore_patterns:
            if pattern.startswith("*"):
                if fnmatch.fnmatch(name, pattern):
                    return True
            elif pattern in file_str:
                return True

        return False

    async def _search_file(self, file_path: Path, pattern: re.Pattern, context_lines: int) -> list[dict[str, Any]]:
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

    def _get_files_to_search(self, search_path: Path, include_pattern: str | None) -> list[Path]:
        """Get list of files to search."""
        files = []

        if include_pattern:
            # Use glob to find matching files
            try:
                pattern = str(search_path / include_pattern)
                matching_files = glob_module.glob(pattern, recursive=True)
                files.extend(
                    [
                        Path(file_str)
                        for file_str in matching_files
                        if Path(file_str).is_file() and not self._should_ignore_file(Path(file_str))
                    ]
                )
            except Exception as e:
                logger.warning(f"Error with glob pattern '{include_pattern}': {e}")
        else:
            # Search all files recursively
            try:
                files.extend(
                    [
                        file_path
                        for file_path in search_path.rglob("*")
                        if file_path.is_file() and not self._should_ignore_file(file_path)
                    ]
                )
            except Exception as e:
                logger.warning(f"Error traversing directory {search_path}: {e}")

        return files

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ToolExecutionContext,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolExecutionResult:
        """Execute the grep tool."""
        try:
            pattern_str = parameters.get("pattern", "").strip()
            search_path_str = parameters.get("path")
            include_pattern = parameters.get("include")
            max_results = parameters.get("max_results", 100)
            context_lines = parameters.get("context_lines", 2)

            if not pattern_str:
                return ToolExecutionResult(
                    success=False,
                    content="Error: pattern parameter is required and cannot be empty.",
                    metadata={"error": "empty_pattern"},
                )

            # Validate and resolve search path
            if search_path_str:
                search_path = Path(search_path_str)
                if not search_path.is_absolute():
                    return ToolExecutionResult(
                        success=False,
                        content=f"Error: search path must be absolute: {search_path_str}",
                        metadata={"error": "invalid_path"},
                    )

                if not self._is_within_root(search_path):
                    return ToolExecutionResult(
                        success=False,
                        content=f"Error: search path must be within root directory {self.root_directory}: {search_path_str}",
                        metadata={"error": "path_outside_root"},
                    )
            else:
                search_path = self.root_directory

            if not search_path.exists():
                return ToolExecutionResult(
                    success=False,
                    content=f"Error: search path does not exist: {search_path}",
                    metadata={"error": "path_not_found"},
                )

            if not search_path.is_dir():
                return ToolExecutionResult(
                    success=False,
                    content=f"Error: search path is not a directory: {search_path}",
                    metadata={"error": "not_directory"},
                )

            # Compile regex pattern
            try:
                pattern = re.compile(pattern_str, re.MULTILINE)
            except re.error as e:
                return ToolExecutionResult(
                    success=False,
                    content=f"Error: Invalid regular expression: {e}",
                    metadata={"error": "invalid_regex"},
                )

            # Get files to search
            files_to_search = self._get_files_to_search(search_path, include_pattern)

            if not files_to_search:
                return ToolExecutionResult(
                    success=True,
                    content="No files found matching the search criteria.",
                    display_content="No files found to search",
                    metadata={
                        "pattern": pattern_str,
                        "search_path": str(search_path),
                        "include_pattern": include_pattern,
                        "files_searched": 0,
                        "matches_found": 0,
                    },
                )

            # Search files
            all_matches = []
            files_with_matches = 0

            search_tasks = [self._search_file(file_path, pattern, context_lines) for file_path in files_to_search]

            # Process results in batches to avoid overwhelming the system
            batch_size = 50
            for i in range(0, len(search_tasks), batch_size):
                batch = search_tasks[i : i + batch_size]
                batch_results = await asyncio.gather(*batch)

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

            # Format results
            if not all_matches:
                return ToolExecutionResult(
                    success=True,
                    content=f"No matches found for pattern '{pattern_str}' in {len(files_to_search)} files.",
                    display_content="No matches found",
                    metadata={
                        "pattern": pattern_str,
                        "search_path": str(search_path),
                        "include_pattern": include_pattern,
                        "files_searched": len(files_to_search),
                        "matches_found": 0,
                    },
                )

            # Group matches by file
            files_dict = {}
            for match in all_matches:
                file_path = match["file_path"]
                if file_path not in files_dict:
                    files_dict[file_path] = []
                files_dict[file_path].append(match)

            # Generate output
            result_lines = []
            result_lines.append(
                f"Found {len(all_matches)} match(es) for pattern '{pattern_str}' in {files_with_matches} file(s):"
            )
            result_lines.append("")

            for file_path, matches in files_dict.items():
                try:
                    relative_path = Path(file_path).relative_to(self.root_directory)
                except ValueError:
                    relative_path = Path(file_path)

                result_lines.append(f"=== {relative_path} ===")

                for match in matches:
                    result_lines.append(f"Match at line {match['line_number']}:")
                    result_lines.extend(match["context"])
                    result_lines.append("")

            result_content = "\n".join(result_lines)

            display_content = f"Found {len(all_matches)} matches in {files_with_matches} files"
            if len(all_matches) >= max_results:
                display_content += f" (limited to {max_results} results)"

            return ToolExecutionResult(
                success=True,
                content=result_content,
                display_content=display_content,
                metadata={
                    "pattern": pattern_str,
                    "search_path": str(search_path),
                    "include_pattern": include_pattern,
                    "files_searched": len(files_to_search),
                    "files_with_matches": files_with_matches,
                    "matches_found": len(all_matches),
                    "results_limited": len(all_matches) >= max_results,
                },
            )

        except Exception as e:
            logger.error(f"Grep tool execution failed: {e}")
            return ToolExecutionResult(
                success=False,
                content=f"Error: Search operation failed: {e!s}",
                metadata={"error": "execution_failed", "exception": str(e)},
            )
