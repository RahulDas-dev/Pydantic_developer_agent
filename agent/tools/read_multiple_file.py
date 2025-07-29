import asyncio
import logging
from pathlib import Path
from typing import Any

import aiofiles

from lib.base_types import CancellationToken

from .base import BaseTool, ToolExecutionContext, ToolExecutionResult, ToolParameter, ToolSchema

logger = logging.getLogger("tools")


class ReadManyFilesTool(BaseTool):
    """Tool for reading multiple files at once."""

    def __init__(self, root_directory: str | None = None):
        super().__init__(
            name="read_many_files",
            description="Reads the contents of multiple files efficiently. "
            "Useful for examining several related files at once.",
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
                    name="file_paths",
                    data_type="array",
                    description="List of file paths to read",
                    required=True,
                    extra={
                        "items": {
                            "type": "string",
                            "description": "Path to a file to read",
                        },
                        "minItems": 1,
                        "maxItems": 50,
                    },
                ),
                ToolParameter(
                    name="max_file_size",
                    data_type="integer",
                    description="Maximum file size to read in bytes (default: 100KB)",
                    default=102400,
                    extra={
                        "minimum": 1024,
                        "maximum": 10485760,
                    },
                ),
                ToolParameter(
                    name="encoding",
                    data_type="string",
                    description="File encoding (default: utf-8)",
                    default="utf-8",
                ),
            ],
            required=["file_paths"],
        )

    def _is_within_root(self, path: Path) -> bool:
        """Check if path is within the root directory."""
        try:
            path.resolve().relative_to(self.root_directory)
            return True
        except ValueError:
            return False

    async def _read_single_file(self, file_path: Path, max_size: int, encoding: str) -> dict[str, Any]:
        """Read a single file and return result info."""
        try:
            if not file_path.exists():
                return {
                    "path": str(file_path),
                    "success": False,
                    "error": "File not found",
                    "content": None,
                    "size": None,
                }

            if not file_path.is_file():
                return {
                    "path": str(file_path),
                    "success": False,
                    "error": "Path is not a file",
                    "content": None,
                    "size": None,
                }

            # Check file size
            file_size = file_path.stat().st_size
            if file_size > max_size:
                return {
                    "path": str(file_path),
                    "success": False,
                    "error": f"File too large ({file_size} bytes, max {max_size})",
                    "content": None,
                    "size": file_size,
                }

            # Read file content
            async with aiofiles.open(file_path, encoding=encoding, errors="ignore") as f:
                content = await f.read()

            return {
                "path": str(file_path),
                "success": True,
                "error": None,
                "content": content,
                "size": file_size,
            }

        except Exception as e:
            return {
                "path": str(file_path),
                "success": False,
                "error": str(e),
                "content": None,
                "size": None,
            }

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ToolExecutionContext,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolExecutionResult:
        """Execute the read many files tool."""
        try:
            file_paths_list = parameters.get("file_paths", [])
            max_file_size = parameters.get("max_file_size", 102400)
            encoding = parameters.get("encoding", "utf-8")

            if not file_paths_list:
                return ToolExecutionResult(
                    success=False,
                    content="Error: file_paths parameter is required and cannot be empty.",
                    metadata={"error": "empty_file_paths"},
                )

            # Validate and resolve file paths
            resolved_paths = []
            for file_path_str in file_paths_list:
                try:
                    file_path = Path(file_path_str)

                    # Make absolute if relative
                    if not file_path.is_absolute():
                        file_path = self.root_directory / file_path

                    file_path = file_path.resolve()

                    # Check if within root directory
                    if not self._is_within_root(file_path):
                        return ToolExecutionResult(
                            success=False,
                            content=f"Error: File path outside root directory: {file_path_str}",
                            metadata={"error": "path_outside_root"},
                        )

                    resolved_paths.append(file_path)

                except Exception as e:
                    return ToolExecutionResult(
                        success=False,
                        content=f"Error: Invalid file path '{file_path_str}': {e!s}",
                        metadata={"error": "invalid_path"},
                    )

            # Read all files concurrently
            read_tasks = [self._read_single_file(path, max_file_size, encoding) for path in resolved_paths]

            results = await asyncio.gather(*read_tasks)

            # Process results
            successful_reads = []
            failed_reads = []

            for result in results:
                if result["success"]:
                    successful_reads.append(result)
                else:
                    failed_reads.append(result)

            if not successful_reads:
                error_summary = "; ".join([f"{r['path']}: {r['error']}" for r in failed_reads])
                return ToolExecutionResult(
                    success=False,
                    content=f"Error: Failed to read any files. Errors: {error_summary}",
                    metadata={
                        "error": "all_reads_failed",
                        "failed_files": failed_reads,
                    },
                )

            # Format output
            result_lines = []
            result_lines.append(f"Successfully read {len(successful_reads)}/{len(resolved_paths)} files:")
            result_lines.append("")

            for result in successful_reads:
                try:
                    relative_path = Path(result["path"]).relative_to(self.root_directory)
                except ValueError:
                    relative_path = Path(result["path"])

                result_lines.append(f"=== {relative_path} ===")
                result_lines.append(f"Size: {result['size']} bytes")
                result_lines.append("")
                result_lines.append(result["content"])
                result_lines.append("")

            if failed_reads:
                result_lines.append("=== Failed to read ===")
                result_lines.extend([f"{result['path']}: {result['error']}" for result in failed_reads])
                result_lines.append("")

            final_content = "\n".join(result_lines)

            display_content = f"Read {len(successful_reads)}/{len(resolved_paths)} files"
            if failed_reads:
                display_content += f" ({len(failed_reads)} failed)"

            return ToolExecutionResult(
                success=True,
                content=final_content,
                display_content=display_content,
                metadata={
                    "files_requested": len(resolved_paths),
                    "files_read": len(successful_reads),
                    "files_failed": len(failed_reads),
                    "successful_reads": successful_reads,
                    "failed_reads": failed_reads,
                    "total_bytes": sum(r["size"] for r in successful_reads),
                },
            )

        except Exception as e:
            logger.error(f"Read many files tool execution failed: {e}")
            return ToolExecutionResult(
                success=False,
                content=f"Error: Read operation failed: {e!s}",
                metadata={"error": "execution_failed", "exception": str(e)},
            )
