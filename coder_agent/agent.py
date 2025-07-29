import logging
from typing import Union

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import Tool

from coder_agent.tools import edit_file, glob_search, list_directory, read_file, write_file

from .context import AgentContext
from .factories import llm_factory
from .prompts import CORE_SYSTEM_MESSAGE, FINAL_MESSAGE, INTERACTION_EXAMPLES

logger = logging.getLogger(__name__)


class Failure(BaseModel):
    """Represents structure for action failure"""

    reason: str = Field(description="The Reason why the task failed")


coder = Agent[AgentContext, str | Failure](
    name="Coder Agent",
    # description="An agent that can perform various Software Engineering tasks.",
    model=llm_factory("azure"),
    tools=[
        Tool(read_file, takes_ctx=True),
        Tool(list_directory, takes_ctx=True),
        Tool(write_file, takes_ctx=True),
        Tool(glob_search, takes_ctx=True),
        Tool(edit_file, takes_ctx=True),
    ],
    deps_type=AgentContext,
    output_type=Union[str, Failure],  # type: ignore  # noqa: PGH003
    retries=3,
)


@coder.system_prompt
async def get_system_prompt(ctx: RunContext[AgentContext]) -> str:
    _system_message = ctx.deps.retrieve_system_message()
    if _system_message:
        return _system_message
    _context_informations = []
    _sandbox_context = ctx.deps.retrieve_sandbox_context()
    if _sandbox_context:
        _context_informations.append(_sandbox_context)
    _git_context = ctx.deps.retrieve_git_context()
    if _git_context:
        _context_informations.append(_git_context)
    _has_python_context = ctx.deps.retrieve_python_context()
    if _has_python_context:
        _context_informations.append(_has_python_context)

    if _context_informations:
        return (
            CORE_SYSTEM_MESSAGE
            + "\n\n"
            + "\n".join(_context_informations)
            + "\n\n"
            + INTERACTION_EXAMPLES
            + "\n\n"
            + FINAL_MESSAGE
        )
    return CORE_SYSTEM_MESSAGE + "\n\n" + INTERACTION_EXAMPLES + "\n\n" + FINAL_MESSAGE
