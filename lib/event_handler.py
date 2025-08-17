from collections.abc import AsyncIterable

from pydantic_ai import RunContext
from pydantic_ai.messages import (
    AgentStreamEvent,
    HandleResponseEvent,
)

from lib.agents.context import HasEventBus

# TAgentDeps = TypeVar("TAgentDeps", bound=AgentDeps)


async def handel_streaming_events(
    ctx: RunContext[HasEventBus], event_stream: AsyncIterable[AgentStreamEvent | HandleResponseEvent]
) -> None:
    async for event in event_stream:
        ctx.deps.event_bus.emit(event)
