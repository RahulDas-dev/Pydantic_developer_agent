import logging
from typing import Union

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, Tool

from lib.configs import config
from lib.tools import edit_file, glob_search, list_directory, read_file, write_file

from .context import AgentContext
from .factories import llm_factory

logger = logging.getLogger(__name__)


class Failure(BaseModel):
    """Represents structure for action failure"""

    reason: str = Field(description="The Reason why the task failed")


agent = Agent[AgentContext, str | Failure](
    name="Coder Agent",
    # description="An agent that can perform various Software Engineering tasks.",
    model=llm_factory(config.PROVIDER_NAME, config.MODEL_NAME),
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


@agent.system_prompt
async def get_system_prompt(ctx: RunContext[AgentContext]) -> str:
    return ctx.deps.get_system_prompt()
