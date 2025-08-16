# ruff: noqa: PLW0603

"""modern_events.py - Modern, fully-typed event system"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any
from weakref import WeakSet

from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from .types import EventHandler, EventType, StreamOutEvent, UserInputEvent

logger = logging.getLogger("event_sys")


class EventSubscription:
    """Represents a subscription that can be cancelled"""

    def __init__(self, event_bus: EventBus, event_type: EventType, handler: EventHandler):
        self.event_bus = event_bus
        self.event_type = event_type
        self.handler = handler
        self._cancelled = False

    def cancel(self) -> None:
        """Cancel this subscription"""
        if not self._cancelled:
            self.event_bus._remove_handler(self.event_type, self.handler)  # noqa: SLF001
            self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cancel()


class SessionEventSubscription(EventSubscription):
    """Session-specific subscription with automatic cleanup"""

    def __init__(self, event_bus: EventBus, session_id: str, event_type: EventType, handler: EventHandler):
        self.session_id = session_id

        # Create session-filtered handler
        async def session_handler(event: StreamOutEvent | UserInputEvent) -> None:
            if event.session_id == session_id:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Error in session handler for {session_id}: {e}")

        super().__init__(event_bus, event_type, session_handler)
        self.original_handler = handler


class EventBus:
    """Modern, fully-typed event bus with excellent asyncio support"""

    def __init__(self) -> None:
        self._handlers: dict[EventType, WeakSet[EventHandler]] = defaultdict(WeakSet)
        self._session_subscriptions: dict[str, list[SessionEventSubscription]] = defaultdict(list)
        self._once_handlers: dict[EventType, WeakSet[EventHandler]] = defaultdict(WeakSet)
        self._lock = asyncio.Lock()

    def subscribe(self, event_type: EventType, handler: EventHandler) -> EventSubscription:
        """Subscribe to events with a cancellable subscription"""
        self._handlers[event_type].add(handler)
        logger.debug(f"Subscribed handler to {event_type}")
        return EventSubscription(self, event_type, handler)

    def subscribe_once(self, event_type: EventType, handler: EventHandler) -> EventSubscription:
        """Subscribe to event once"""
        self._once_handlers[event_type].add(handler)
        logger.debug(f"Subscribed one-time handler to {event_type}")
        return EventSubscription(self, event_type, handler)

    def subscribe_session(
        self, session_id: str, event_type: EventType, handler: EventHandler
    ) -> SessionEventSubscription:
        """Subscribe to events for a specific session"""
        subscription = SessionEventSubscription(self, session_id, event_type, handler)
        self._handlers[event_type].add(subscription.handler)
        self._session_subscriptions[session_id].append(subscription)
        logger.debug(f"Subscribed session handler for {session_id} to {event_type}")
        return subscription

    async def emit(self, event: StreamOutEvent | UserInputEvent) -> None:
        """Emit event to all subscribers"""
        event_type = event.event_type

        try:
            # Get all handlers (regular + once)
            regular_handlers = list(self._handlers[event_type])
            once_handlers = list(self._once_handlers[event_type])

            # Clear once handlers immediately
            if once_handlers:
                self._once_handlers[event_type].clear()

            all_handlers = regular_handlers + once_handlers

            if not all_handlers:
                return

            # Execute all handlers concurrently
            tasks = []
            for handler in all_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        tasks.append(handler(event))
                    else:
                        # Run sync handler in thread pool to avoid blocking
                        tasks.append(asyncio.get_event_loop().run_in_executor(None, handler, event))
                except Exception as e:  # noqa: PERF203
                    logger.error(f"Error preparing handler for {event_type}: {e}")

            if tasks:
                # Wait for all handlers to complete, but don't fail if one handler fails
                results = await asyncio.gather(*tasks, return_exceptions=True)

            logger.debug(f"Emitted {event_type} to {len(all_handlers)} handlers for session {event.session_id}")

        except Exception as e:
            logger.error(f"Error emitting event {event_type}: {e}")

    def _remove_handler(self, event_type: EventType, handler: EventHandler) -> None:
        """Internal method to remove a handler"""
        self._handlers[event_type].discard(handler)
        self._once_handlers[event_type].discard(handler)
        logger.debug(f"Removed handler from {event_type}")

    async def cleanup_session(self, session_id: str) -> None:
        """Clean up all handlers for a session"""
        if session_id not in self._session_subscriptions:
            return

        async with self._lock:
            subscriptions = self._session_subscriptions[session_id]

            for subscription in subscriptions:
                subscription.cancel()

            del self._session_subscriptions[session_id]
            logger.debug(f"Cleaned up {len(subscriptions)} handlers for session {session_id}")

    def get_session_handler_count(self, session_id: str) -> int:
        """Get number of active handlers for a session"""
        return len([sub for sub in self._session_subscriptions[session_id] if not sub.is_cancelled])

    def get_active_sessions(self) -> list[str]:
        """Get list of sessions with active handlers"""
        return [
            session_id
            for session_id, subs in self._session_subscriptions.items()
            if any(not sub.is_cancelled for sub in subs)
        ]

    def get_handler_count(self, event_type: EventType) -> int:
        """Get total number of handlers for an event type"""
        return len(self._handlers[event_type]) + len(self._once_handlers[event_type])

    async def clear_all(self) -> None:
        """Clear all handlers and sessions"""
        async with self._lock:
            self._handlers.clear()
            self._once_handlers.clear()

            # Cancel all session subscriptions
            for subscriptions in self._session_subscriptions.values():
                for sub in subscriptions:
                    sub.cancel()

            self._session_subscriptions.clear()
            logger.debug("Cleared all handlers and sessions")

    @asynccontextmanager
    async def temporary_subscription(
        self, event_type: EventType, handler: EventHandler
    ) -> AsyncIterator[EventSubscription]:
        """Context manager for temporary subscriptions"""
        subscription = self.subscribe(event_type, handler)
        try:
            yield subscription
        finally:
            subscription.cancel()


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
