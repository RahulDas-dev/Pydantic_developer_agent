# ruff: noqa: PLW0603
import logging

from .async_bus import EventBus, EventSubscription
from .types import EventHandler, EventType, StreamOutEvent, UserInputEvent

__all__ = (
    "EventBus",
    "EventHandler",
    "EventSubscription",
    "EventType",
    "StreamOutEvent",
    "UserInputEvent",
    "get_event_bus",
    "reset_event_bus",
    "reset_event_bus",
)

logger = logging.getLogger("event_sys")

# Global instance with proper typing
_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance (singleton pattern)"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
        logger.debug("Created new ModernEventBus instance")
    return _event_bus


async def reset_event_bus() -> None:
    """Reset the global event bus (useful for testing)"""
    global _event_bus
    if _event_bus is not None:
        await _event_bus.clear_all()
    _event_bus = None
    logger.debug("Reset global ModernEventBus instance")


# Example usage:
"""
# Basic usage
bus = get_event_bus()

# Subscribe with automatic cleanup
with bus.subscribe("part_start | text", my_handler) as subscription:
    # Handler is active
    pass
# Handler automatically cleaned up

# Session-specific handlers
session_sub = bus.subscribe_session("session123", "part_start | text", my_session_handler)

# Emit events
await bus.emit(StreamEvent(session_id="session123", data=some_event))

# Cleanup
await bus.cleanup_session("session123")
"""
