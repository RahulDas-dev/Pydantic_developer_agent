# ruff:  noqa: PLW0603
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles

from lib.base_types import CancellationToken

from .base import BaseTool, ToolExecutionContext, ToolExecutionResult, ToolParameter, ToolSchema

logger = logging.getLogger("tools")


# Memory configuration (matches TypeScript implementation)
CONFIG_DIR = ".config"
DEFAULT_CONTEXT_FILENAME = "Memory.md"
MEMORY_SECTION_HEADER = "## Added Memories"

# Global variable to track current memory filename (matches TypeScript)
_current_memory_md_filename = DEFAULT_CONTEXT_FILENAME


def set_memory_md_filename(new_filename: str) -> None:
    """Set the filename for memory storage (matches TypeScript setMemoryMdFilename)."""
    global _current_memory_md_filename
    if new_filename and new_filename.strip():
        _current_memory_md_filename = new_filename.strip()


def get_current_memory_md_filename() -> str:
    """Get current memory filename (matches TypeScript getCurrentMemoryMdFilename)."""
    return _current_memory_md_filename


def get_global_memory_file_path() -> Path:
    """Get global memory file path (matches TypeScript getGlobalMemoryFilePath)."""
    return Path.home() / CONFIG_DIR / get_current_memory_md_filename()


def ensure_newline_separation(current_content: str) -> str:
    """Ensure proper newline separation (matches TypeScript ensureNewlineSeparation)."""
    if len(current_content) == 0:
        return ""
    if current_content.endswith("\n\n"):
        return ""
    if current_content.endswith("\n"):
        return "\n"
    return "\n\n"


class MemoryTool(BaseTool):
    """Tool for saving information to long-term memory (matches TypeScript MemoryTool)."""

    def __init__(self, memory_dir: str | None = None):
        super().__init__(
            name="save_memory",
            description="Saves a specific piece of information or fact to your long-term memory. "
            "Use this when the user explicitly asks you to remember something, or when "
            "they state a clear, concise fact that seems important to retain for future interactions.",
        )

        if memory_dir:
            self.memory_dir = Path(memory_dir)
            self.memory_file = self.memory_dir / get_current_memory_md_filename()
        else:
            self.memory_file = get_global_memory_file_path()
            self.memory_dir = self.memory_file.parent

    def _get_global_memory_file_path(self) -> Path:
        """Get path to global memory file in user's home directory."""
        return Path.home() / CONFIG_DIR / DEFAULT_CONTEXT_FILENAME

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="fact",
                    data_type="string",
                    description="The specific fact or piece of information to remember. "
                    "Should be a clear, self-contained statement.",
                    required=True,
                ),
            ],
            required=["fact"],
        )

    async def _perform_add_memory_entry(self, text: str, memory_file_path: Path) -> None:
        """Add memory entry (matches TypeScript performAddMemoryEntry)."""
        processed_text = text.strip()
        # Remove leading hyphens and spaces that might be misinterpreted as markdown list items
        processed_text = processed_text.lstrip("- ").strip()
        new_memory_item = f"- {processed_text}"

        try:
            # Ensure directory exists
            memory_file_path.parent.mkdir(parents=True, exist_ok=True)

            content = ""
            try:
                async with aiofiles.open(memory_file_path, encoding="utf-8") as f:
                    content = await f.read()
            except FileNotFoundError:
                # File doesn't exist, will be created with header and item
                pass

            header_index = content.find(MEMORY_SECTION_HEADER)

            if header_index == -1:
                # Header not found, append header and then the entry
                separator = ensure_newline_separation(content)
                content += f"{separator}{MEMORY_SECTION_HEADER}\n{new_memory_item}\n"
            else:
                # Header found, find where to insert the new memory entry
                start_of_section_content = header_index + len(MEMORY_SECTION_HEADER)
                end_of_section_index = content.find("\n## ", start_of_section_content)
                if end_of_section_index == -1:
                    end_of_section_index = len(content)  # End of file

                before_section_marker = content[:start_of_section_content].rstrip()
                section_content = content[start_of_section_content:end_of_section_index].rstrip()
                after_section_marker = content[end_of_section_index:]

                section_content += f"\n{new_memory_item}"
                content = f"{before_section_marker}\n{section_content.lstrip()}\n{after_section_marker}".rstrip() + "\n"

            async with aiofiles.open(memory_file_path, "w", encoding="utf-8") as f:
                await f.write(content)

        except Exception as error:
            logger.error(f"Error adding memory entry to {memory_file_path}: {error}")
            raise Exception(f"Failed to add memory entry: {error!s}")

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ToolExecutionContext,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolExecutionResult:
        """Execute the memory tool (matches TypeScript execute method)."""
        try:
            fact = parameters.get("fact", "").strip()

            if not fact:
                error_message = 'Parameter "fact" must be a non-empty string.'
                return ToolExecutionResult(
                    content=f"Error: {error_message}",
                    success=False,
                    metadata={"error": "empty_fact"},
                )

            try:
                await self._perform_add_memory_entry(fact, self.memory_file)
                success_message = f'Okay, I\'ve remembered that: "{fact}"'
                return ToolExecutionResult(
                    content=success_message,
                    success=True,
                    metadata={
                        "fact": fact,
                        "memory_file": str(self.memory_file),
                        "timestamp": datetime.now().isoformat(),
                    },
                )

            except Exception as error:
                error_message = str(error)
                logger.error(f"Error executing save_memory for fact '{fact}': {error_message}")
                return ToolExecutionResult(
                    content=f"Error saving memory: {error_message}",
                    success=False,
                    metadata={
                        "error": "execution_failed",
                        "exception": error_message,
                        "fact": fact,
                    },
                )

        except Exception as e:
            logger.error(f"Memory tool execution failed: {e}")
            return ToolExecutionResult(
                content=f"Error: Memory operation failed: {e!s}",
                success=False,
                metadata={"error": "tool_execution_failed", "exception": str(e)},
            )
