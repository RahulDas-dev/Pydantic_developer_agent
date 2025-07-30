import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import aiofiles
from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from pydantic_ai.messages import ToolReturn

from .classifier import classify_memory
from .summarizer import summarize_memory

logger = logging.getLogger(__name__)

# Memory configuration
CONFIG_DIR = ".memories"
DEFAULT_MEMORY_FILENAME = "memories.md"
MEMORY_SECTION_HEADER = "## Added Memories"


class MemoryOutput(BaseModel):
    """Output model for memory operations."""

    success: bool = Field(description="Whether the operation was successful")
    message: str = Field(description="Success or error message")
    memories: Optional[List[str]] = Field(None, description="List of retrieved memories")


class Memory(BaseModel):
    """Model for storing memory information."""

    fact: str = Field(description="The fact to remember")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(), description="When the memory was created"
    )
    source: str = Field(default="user", description="Source of the memory")


async def _get_memory_file_path() -> Path:
    """Get the path to the memory file."""
    home_dir = Path.home()
    memory_dir = home_dir / CONFIG_DIR
    memory_file = memory_dir / DEFAULT_MEMORY_FILENAME

    # Ensure directory exists
    memory_dir.mkdir(parents=True, exist_ok=True)

    # Create file if it doesn't exist
    if not memory_file.exists():
        async with aiofiles.open(memory_file, "w", encoding="utf-8") as f:
            await f.write(f"{MEMORY_SECTION_HEADER}\n\n")

    return memory_file


async def _read_memories() -> List[str]:
    """Read all stored memories."""
    memory_file = await _get_memory_file_path()

    try:
        async with aiofiles.open(memory_file, "r", encoding="utf-8") as f:
            content = await f.read()

        # Find the memories section
        header_index = content.find(MEMORY_SECTION_HEADER)
        if header_index == -1:
            return []

        # Extract memories
        section_start = header_index + len(MEMORY_SECTION_HEADER)
        section_end = content.find("\n## ", section_start)
        if section_end == -1:
            section_end = len(content)

        memory_section = content[section_start:section_end].strip()

        # Parse memory items
        memories = []
        for line_text in memory_section.split("\n"):
            cleaned_line = line_text.strip()
            if cleaned_line.startswith("- "):
                memories.append(cleaned_line[2:])

        return memories
    except Exception as e:
        logger.error(f"Error reading memories: {e}")
        return []


async def _store_memory(memory: Memory) -> bool:
    """Store a memory in the memory file."""
    memory_file = await _get_memory_file_path()

    try:
        # Read existing content
        async with aiofiles.open(memory_file, "r", encoding="utf-8") as f:
            content = await f.read()

        # Find or create the memories section
        header_index = content.find(MEMORY_SECTION_HEADER)
        if header_index == -1:
            content += f"\n{MEMORY_SECTION_HEADER}\n"
            header_index = content.find(MEMORY_SECTION_HEADER)

        # Format memory entry
        memory_entry = f"- {memory.fact}"

        # Insert at appropriate position
        section_start = header_index + len(MEMORY_SECTION_HEADER)
        section_end = content.find("\n## ", section_start)
        if section_end == -1:
            section_end = len(content)

        before_section = content[:section_start]
        section_content = content[section_start:section_end].strip()
        after_section = content[section_end:]

        # Add memory to section content
        section_content = f"{section_content}\n{memory_entry}" if section_content else memory_entry

        # Rebuild content
        new_content = f"{before_section}\n{section_content}\n{after_section}"

        # Write back
        async with aiofiles.open(memory_file, "w", encoding="utf-8") as f:
            await f.write(new_content)

        return True
    except Exception as e:
        logger.error(f"Error storing memory: {e}")
        return False


async def save_memory(
    # ctx: RunContext[AgentContext],
    fact: str,
) -> ToolReturn:
    """
    Save a fact to the agent's memory.

    Args:
        fact: The specific fact or piece of information to remember.
              Should be a clear, self-contained statement.

    Returns:
        ToolReturn: Success or failure information
    """
    try:
        # Read existing memories
        existing_memories = await _read_memories()

        # Use classifier to determine if we should remember this fact
        classifier_result = await classify_memory(existing_memories, fact)

        # If the classifier says we shouldn't remember, return early
        if not classifier_result.should_remember:
            message = f"I've decided not to remember this because: {classifier_result.reason}"
            output = MemoryOutput(success=True, message=message, memories=existing_memories)
            return ToolReturn(return_value=message, content=json.dumps(output.model_dump()))

        # Use summarizer to process the memory
        summarizer_result = await summarize_memory(existing_memories, fact)

        # Store the processed memory
        memory = Memory(fact=summarizer_result.summary, timestamp=datetime.now().isoformat(), source="user")

        success = await _store_memory(memory)

        if success:
            message = f"I've remembered: {memory.fact}"
            output = MemoryOutput(
                success=True,
                message=message,
                memories=await _read_memories(),  # Get updated list
            )
            return ToolReturn(return_value=message, content=json.dumps(output.model_dump()))

        message = "I couldn't save that to my memory due to a technical issue."
        output = MemoryOutput(success=False, message=message, memories=existing_memories)
        return ToolReturn(return_value=message, content=json.dumps(output.model_dump()))

    except Exception as e:
        logger.error(f"Error in save_memory: {e}")
        error_msg = f"Error saving memory: {e!s}"
        output = MemoryOutput(success=False, message=error_msg, memories=[])
        return ToolReturn(return_value=error_msg, content=json.dumps(output.model_dump()))


async def retrieve_memories(
    # ctx: RunContext[AgentContext],
    query: Optional[str] = None,
) -> ToolReturn:
    """
    Retrieve memories matching the query.

    Args:
        query: Optional search query to filter memories

    Returns:
        ToolReturn: List of matching memories
    """
    try:
        memories = await _read_memories()

        # If a query is provided, we could use an embedding search here
        # For now, we'll just return all memories

        output = MemoryOutput(success=True, message="Successfully retrieved memories", memories=memories)

        return ToolReturn(return_value="Successfully retrieved memories", content=json.dumps(output.model_dump()))

    except Exception as e:
        logger.error(f"Error in retrieve_memories: {e}")
        output = MemoryOutput(success=False, message=f"Error retrieving memories: {e!s}", memories=[])
        return ToolReturn(return_value=f"Error retrieving memories: {e!s}", content=json.dumps(output.model_dump()))
