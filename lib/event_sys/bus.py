# events.py - Event definitions
import logging
from collections.abc import Callable

from pyee.asyncio import AsyncIOEventEmitter

from .types import EventType, StreamEvent

logger = logging.getLogger("event_sys")


class EventBus:
    def __init__(self):
        self.emitter = AsyncIOEventEmitter()
        self._active_sessions: dict[str, list[Callable]] = {}

    def on(self, event_type: EventType, handler: Callable) -> None:
        """Subscribe to events"""
        self.emitter.on(event_type.value, handler)
        logger.debug(f"Subscribed handler to {event_type.value}")

    def once(self, event_type: EventType, handler: Callable) -> None:
        """Subscribe to event once"""
        self.emitter.once(event_type.value, handler)

    def off(self, event_type: EventType, handler: Callable) -> None:
        """Unsubscribe from event"""
        self.emitter.remove_listener(event_type.value, handler)

    def emit(self, event: StreamEvent) -> None:
        """Emit event to all subscribers"""
        try:
            self.emitter.emit(event.type.value, event)
            logger.debug(f"Emitted {event.type.value} for session {event.session_id}")
        except Exception as e:
            logger.error(f"Error emitting event {event.type.value}: {e}")

    def session_on(self, session_id: str, event_type: EventType, handler: Callable) -> None:
        """Subscribe to events for specific session"""

        async def session_handler(event: StreamEvent) -> None:
            if event.session_id == session_id:
                await handler(event)

        self.emitter.on(event_type.value, session_handler)

        # Track for cleanup
        if session_id not in self._active_sessions:
            self._active_sessions[session_id] = []

        self._active_sessions[session_id].append(session_handler)

    def cleanup_session(self, session_id: str) -> None:
        """Clean up session-specific handlers"""
        if session_id in self._active_sessions:
            for handler in self._active_sessions[session_id]:
                logger.debug(f"Removing handler for session {session_id}")
                pass
            del self._active_sessions[session_id]
            logger.debug(f"Cleaned up session {session_id}")


event_bus = None


def get_event_bus() -> EventBus:
    global event_bus
    if event_bus is None:
        event_bus = EventBus()
    return event_bus
