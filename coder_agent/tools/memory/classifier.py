import logging
from typing import List

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from coder_agent.factories import llm_factory

logger = logging.getLogger(__name__)


class MemoryClassifierContext(BaseModel):
    """Context for the memory classifier agent."""

    existing_memories: List[str] = Field(default_factory=list, description="List of existing memories")
    user_input: str = Field("", description="User input to process")


class MemoryClassifier(BaseModel):
    """Model for classifying memory importance."""

    should_remember: bool = Field(description="Whether this fact should be remembered")
    reason: str = Field(description="Reason for the decision")


# Create classifier agent to determine if a fact should be remembered
memory_classifier_agent = Agent[MemoryClassifierContext, MemoryClassifier](
    name="Memory Classifier",
    model=llm_factory("ollama", "qwen3:8b"),  # Can be configured based on your needs
    deps_type=MemoryClassifierContext,
    output_type=MemoryClassifier,
)


@memory_classifier_agent.system_prompt
async def get_classifier_system_prompt(ctx: RunContext[MemoryClassifierContext]) -> str:
    return """You are a memory classifier for an AI assistant. Your job is to determine if a user statement should be remembered as a long-term memory.

Only remember statements that:
1. Contain personal information about the user (preferences, facts, etc.)
2. Would be useful in future conversations
3. Are concise and specific

Don't remember:
1. Generic conversational exchanges 
2. Technical discussions not related to personal information
3. Temporary context only relevant to the current conversation
4. Vague or ambiguous statements

User input: {user_input}

Existing memories:
""" + "\n".join([f"- {memory}" for memory in ctx.deps.existing_memories])


async def classify_memory(memories: List[str], user_input: str) -> MemoryClassifier:
    """
    Classify whether a piece of user input should be remembered.

    Args:
        memories: List of existing memories
        user_input: The user input to classify

    Returns:
        MemoryClassifier: Classification result with decision and reason
    """
    try:
        memory_context = MemoryClassifierContext(existing_memories=memories, user_input=user_input)

        classifier_result = await memory_classifier_agent.run(
            "",  # Empty user prompt since we're passing context
            deps=memory_context,
        )

        return classifier_result.output
    except Exception as e:
        logger.error(f"Error classifying memory: {e}")
        # Default to not remembering in case of error
        return MemoryClassifier(should_remember=False, reason=f"Error in classification: {e!s}")
