from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from pydantic_ai.messages import (
    AgentStreamEvent,
    BuiltinToolCallEvent,
    BuiltinToolResultEvent,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    HandleResponseEvent,
    PartDeltaEvent,
    PartStartEvent,
)

if TYPE_CHECKING:
    from datetime import datetime

EventType = Literal[
    # AgentStreamEvent - PartStartEvent
    "part_start | text",
    "part_start | tool-call",
    "part_start | builtin-tool-call",
    "part_start | builtin-tool-return",
    "part_start | thinking",
    # AgentStreamEvent - PartDeltaEvent
    "part_delta | text",
    "part_delta | thinking",
    "part_delta | tool_call",
    # AgentStreamEvent - FinalResultEvent
    "final_result | ",
    # HandleResponseEvent - FunctionToolCallEvent
    "function_tool_call | tool-call",
    # HandleResponseEvent - FunctionToolResultEvent
    "function_tool_result | tool-return",
    "function_tool_result | retry-prompt",
    # HandleResponseEvent - BuiltinToolCallEvent
    "builtin_tool_call | builtin-tool-call",
    # HandleResponseEvent - BuiltinToolResultEvent
    "builtin_tool_result | builtin-tool-return",
    # Edge case
    "Unknown",
    "input | text",
    "input | command",
    "input | exit",
]


@dataclass
class StreamOutEvent:
    session_id: str
    data: AgentStreamEvent | HandleResponseEvent

    @property
    def event_type(self) -> EventType:
        type2_ = "Unknown"
        if isinstance(self.data, (PartStartEvent, FunctionToolCallEvent, BuiltinToolCallEvent)):
            type2_ = self.data.part.part_kind
        elif isinstance(self.data, PartDeltaEvent):  # Changed to elif
            type2_ = self.data.delta.part_delta_kind
        elif isinstance(self.data, FinalResultEvent):  # Changed to elif
            type2_ = ""
        elif isinstance(self.data, (FunctionToolResultEvent, BuiltinToolResultEvent)):  # Changed to elif
            type2_ = self.data.result.part_kind

        if type2_ == "Unknown":
            return "Unknown"

        return cast("EventType", f"{self.data.event_kind} | {type2_}")

    @property
    def timestamp(self) -> datetime | None:
        # For tool result events
        if isinstance(self.data, (FunctionToolResultEvent, BuiltinToolResultEvent)) and hasattr(
            self.data.result, "timestamp"
        ):
            return self.data.result.timestamp

        if hasattr(self.data, "timestamp"):
            return self.data.timestamp

        return None


@dataclass
class UserInputEvent:
    session_id: str
    data: str
    timestamp: datetime

    @property
    def event_type(self) -> EventType:
        if self.data.startswith("/"):
            return "input | command"
        return "input | text"


# Type aliases
AsyncEventHandler = Callable[[StreamOutEvent | UserInputEvent], Awaitable[None]]
SyncEventHandler = Callable[[StreamOutEvent | UserInputEvent], None]
EventHandler = AsyncEventHandler | SyncEventHandler
