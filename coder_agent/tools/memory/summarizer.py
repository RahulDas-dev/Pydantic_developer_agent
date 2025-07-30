import logging
from typing import List

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from coder_agent.factories import llm_factory

logger = logging.getLogger(__name__)


class MemorySummarizerContext(BaseModel):
    """Context for the memory summarizer agent."""

    existing_memories: List[str] = Field(default_factory=list, description="List of existing memories")
    user_input: str = Field("", description="User input to process")


class MemorySummary(BaseModel):
    """Model for summarized memory."""

    fact: str = Field(description="The original fact")
    summary: str = Field(description="Summarized version of the fact")
    key_points: List[str] = Field(description="Key points extracted from the fact")
    importance: int = Field(description="Importance score from 1-10")


# Create summarizer agent to process and store memories in a structured way
memory_summarizer_agent = Agent[MemorySummarizerContext, MemorySummary](
    name="Memory Summarizer",
    model=llm_factory("ollama", "qwen3:8b"),  # Can be configured based on your needs
    deps_type=MemorySummarizerContext,
    output_type=MemorySummary,
)


@memory_summarizer_agent.system_prompt
async def get_summarizer_system_prompt(ctx: RunContext[MemorySummarizerContext]) -> str:
    return """You are a memory summarizer for an AI assistant. Your job is to process user statements into concise, memorable facts.

For the given input:
1. Extract the core fact in third-person format (e.g., "The user likes chocolate")
2. Create a summary that captures the essential information
3. Identify key points that would be useful in future interactions
4. Assess the importance on a scale of 1-10

User input: {user_input}

Existing memories:
""" + "\n".join([f"- {memory}" for memory in ctx.deps.existing_memories])


async def summarize_memory(memories: List[str], user_input: str) -> MemorySummary:
    """
    Summarize a piece of user input into a structured memory.

    Args:
        memories: List of existing memories
        user_input: The user input to summarize

    Returns:
        MemorySummary: Structured summary of the input
    """
    try:
        memory_context = MemorySummarizerContext(existing_memories=memories, user_input=user_input)

        summarizer_result = await memory_summarizer_agent.run(
            "",  # Empty user prompt since we're passing context
            deps=memory_context,
        )

        return summarizer_result.output
    except Exception as e:
        logger.error(f"Error summarizing memory: {e}")
        # Create a basic summary in case of error
        return MemorySummary(fact=user_input, summary=user_input, key_points=["Error processing input"], importance=1)
