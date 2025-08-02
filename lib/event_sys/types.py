from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from pydantic_ai.messages import (
    AgentStreamEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
)


class EventType(str, Enum):
    STREAM_START = "stream_start"
    STREAM_CHUNK = "stream_chunk"
    STREAM_ERROR = "stream_error"
    STREAM_COMPLETE = "stream_complete"
    AGENT_THINKING = "agent_thinking"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    TOOL_CALL_ERROR = "tool_call_error"
    USER_INPUT = "user_input"


@dataclass
class StreamEvent:
    type: EventType
    session_id: str
    timestamp: datetime
    data: (
        AgentStreamEvent
        | PartDeltaEvent
        | PartStartEvent
        | TextPart
        | TextPartDelta
        | ThinkingPart
        | ThinkingPartDelta
        | ToolCallPart
        | ToolCallPartDelta
        | ToolReturnPart
    )
