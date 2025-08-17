"""async_bus.py - Modern, fully-typed event system with backpressure handling"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from contextlib import asynccontextmanager, suppress
from enum import Enum
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from .types import EventHandler, EventType, StreamOutEvent, UserInputEvent

logger = logging.getLogger("event_sys")


class BackpressureStrategy(Enum):
    """Strategy for handling backpressure when queues are full"""

    DROP_OLDEST = "drop_oldest"  # Remove oldest events
    DROP_NEWEST = "drop_newest"  # Drop incoming events
    BLOCK = "block"  # Block until space available


class EventSubscription:
    """Represents a subscription that can be cancelled"""

    def __init__(
        self,
        event_bus: EventBus,
        event_type: EventType,
        handler: EventHandler,
        queue_size: int = 1000,
        backpressure_strategy: BackpressureStrategy = BackpressureStrategy.DROP_OLDEST,
    ):
        self.event_bus = event_bus
        self.event_type = event_type
        self.handler = handler
        self._cancelled = False

        # Backpressure handling
        self.queue_size = queue_size
        self.backpressure_strategy = backpressure_strategy
        self.event_queue: asyncio.Queue[StreamOutEvent | UserInputEvent] = asyncio.Queue(maxsize=queue_size)
        self._processing_task: asyncio.Task | None = None
        self._dropped_events = 0
        self._processed_events = 0
        self._sample_counter = 0
        self._sample_rate = 10  # Keep every 10th event when sampling

    async def enqueue_event(self, event: StreamOutEvent | UserInputEvent) -> bool:
        """
        Add event to queue with backpressure handling.
        Returns True if event was queued, False if dropped.
        """
        if self._cancelled:
            return False

        try:
            if self.backpressure_strategy == BackpressureStrategy.BLOCK:
                # Block until space is available
                await self.event_queue.put(event)
                return True

            if self.backpressure_strategy == BackpressureStrategy.DROP_NEWEST:
                # Drop the new event if queue is full
                self.event_queue.put_nowait(event)
                return True

            if self.backpressure_strategy == BackpressureStrategy.DROP_OLDEST:
                # Drop old events to make room for new ones
                if self.event_queue.full():
                    try:
                        self.event_queue.get_nowait()  # Remove oldest
                        self._dropped_events += 1
                    except asyncio.QueueEmpty:
                        pass
                self.event_queue.put_nowait(event)
                return True
        except asyncio.QueueFull:
            self._dropped_events += 1
            return False

        return True

    async def start_processing(self) -> None:
        """Start processing events from the queue"""
        if self._processing_task is None:
            self._processing_task = asyncio.create_task(self._process_events())

    async def stop_processing(self) -> None:
        """Stop processing events and cleanup"""
        if self._processing_task:
            self._processing_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._processing_task
            self._processing_task = None

    async def _process_events(self) -> None:
        """Process events from the queue"""
        try:
            while not self._cancelled:
                try:
                    # Get event with timeout to check cancellation
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=0.1)

                    # Process the event
                    try:
                        if asyncio.iscoroutinefunction(self.handler):
                            await self.handler(event)
                        else:
                            # Run sync handler in thread pool to avoid blocking
                            await asyncio.get_event_loop().run_in_executor(None, self.handler, event)

                        self._processed_events += 1
                        self.event_queue.task_done()

                    except Exception as e:
                        logger.error(f"Error in event handler: {e}")
                        self.event_queue.task_done()

                except asyncio.TimeoutError:
                    continue

        except asyncio.CancelledError:
            logger.debug(f"Event processing cancelled for {self.event_type}")

    @property
    def stats(self) -> dict[str, Any]:
        """Get subscription statistics"""
        return {
            "processed_events": self._processed_events,
            "dropped_events": self._dropped_events,
            "queue_size": self.event_queue.qsize(),
            "queue_max_size": self.queue_size,
            "backpressure_strategy": self.backpressure_strategy.value,
            "is_cancelled": self._cancelled,
        }

    def cancel(self) -> None:
        """Cancel this subscription"""
        if not self._cancelled:
            self._cancelled = True
            # Start cleanup task with proper reference handling
            if self._processing_task:
                cleanup_task = asyncio.create_task(self.stop_processing())
                # Store reference temporarily to prevent GC, it will be cleaned up when done
                cleanup_task.add_done_callback(lambda t: None)

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
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
    """Modern, fully-typed event bus with excellent asyncio support and backpressure handling"""

    def __init__(
        self,
        default_queue_size: int = 1000,
        default_backpressure_strategy: BackpressureStrategy = BackpressureStrategy.DROP_OLDEST,
    ) -> None:
        self._session_subscriptions: dict[str, list[SessionEventSubscription]] = defaultdict(list)
        self._lock = asyncio.Lock()

        # Subscriptions organized by event type
        self._subscriptions: dict[EventType, list[EventSubscription]] = defaultdict(list)

        # Track background tasks to prevent garbage collection
        self._background_tasks: set[asyncio.Task] = set()

        # Default settings
        self.default_queue_size = default_queue_size
        self.default_backpressure_strategy = default_backpressure_strategy

    def _create_background_task(self, coro) -> asyncio.Task:
        """Create a background task and track it to prevent garbage collection"""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)

        # Remove task from set when it's done to prevent memory leaks
        def remove_task(task_ref: asyncio.Task) -> None:
            self._background_tasks.discard(task_ref)

        task.add_done_callback(remove_task)
        return task

    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
    ) -> EventSubscription:
        """Subscribe to events with backpressure handling"""

        # Always use backpressure subscription with default settings
        subscription = EventSubscription(
            self,
            event_type,
            handler,
            self.default_queue_size,
            self.default_backpressure_strategy,
        )

        self._subscriptions[event_type].append(subscription)

        # Start processing task
        self._create_background_task(subscription.start_processing())

        logger.debug(f"Subscribed handler to {event_type} with backpressure")
        return subscription

    def subscribe_once(self, event_type: EventType, handler: EventHandler) -> EventSubscription:
        """Subscribe to event once"""

        # Create a wrapper handler that cancels itself after first execution
        async def once_wrapper(event: StreamOutEvent | UserInputEvent) -> None:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            finally:
                # Cancel this subscription after handling the event
                subscription.cancel()

        # Create backpressure subscription with wrapper
        subscription = EventSubscription(
            self,
            event_type,
            once_wrapper,
            self.default_queue_size,
            self.default_backpressure_strategy,
        )

        self._subscriptions[event_type].append(subscription)

        # Start processing task
        self._create_background_task(subscription.start_processing())

        logger.debug(f"Subscribed one-time handler to {event_type}")
        return subscription

    def subscribe_session(
        self, session_id: str, event_type: EventType, handler: EventHandler
    ) -> SessionEventSubscription:
        """Subscribe to events for a specific session"""
        subscription = SessionEventSubscription(self, session_id, event_type, handler)

        # Add to subscriptions by event type
        self._subscriptions[event_type].append(subscription)

        # Add to session tracking
        self._session_subscriptions[session_id].append(subscription)

        # Start processing task
        self._create_background_task(subscription.start_processing())

        logger.debug(f"Subscribed session handler for {session_id} to {event_type}")
        return subscription

    async def emit(self, event: StreamOutEvent | UserInputEvent) -> None:
        """
        Emit event to all subscribers.
        Returns statistics about event delivery.
        """
        event_type = event.event_type

        try:
            for subscription in self._subscriptions[event_type]:
                if subscription.is_cancelled:
                    continue
                success = await subscription.enqueue_event(event)
                logger.debug(f"Enqueued event {event_type} for subscription {subscription.event_type}: {success}")
        except Exception as e:
            logger.error(f"Error emitting event {event_type}: {e}")

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
        return len([sub for sub in self._subscriptions[event_type] if not sub.is_cancelled])

    def get_backpressure_stats(self) -> dict[str, Any]:
        """Get statistics for all subscriptions"""
        stats = {
            "total_subscriptions": 0,
            "active_subscriptions": 0,
            "total_processed": 0,
            "total_dropped": 0,
            "subscriptions": [],
        }

        for event_type, subscriptions in self._subscriptions.items():
            for subscription in subscriptions:
                sub_stats = subscription.stats
                stats["subscriptions"].append({"event_type": str(event_type), **sub_stats})
                stats["total_subscriptions"] += 1

                if not subscription.is_cancelled:
                    stats["active_subscriptions"] += 1
                    stats["total_processed"] += sub_stats["processed_events"]
                    stats["total_dropped"] += sub_stats["dropped_events"]

        return stats

    async def cleanup_cancelled_subscriptions(self) -> int:
        """Remove cancelled subscriptions and return count removed"""
        async with self._lock:
            removed_count = 0

            for event_type, subscriptions in self._subscriptions.items():
                # Create a new list without cancelled subscriptions
                active_subscriptions = []
                for subscription in subscriptions:
                    if subscription.is_cancelled:
                        await subscription.stop_processing()
                        removed_count += 1
                    else:
                        active_subscriptions.append(subscription)

                # Update the list for this event type
                self._subscriptions[event_type] = active_subscriptions

            logger.debug(f"Cleaned up {removed_count} cancelled subscriptions")
            return removed_count

    async def set_global_backpressure_strategy(self, strategy: BackpressureStrategy) -> None:
        """Change backpressure strategy for all active subscriptions"""
        async with self._lock:
            self.default_backpressure_strategy = strategy

            for subscriptions in self._subscriptions.values():
                for subscription in subscriptions:
                    if not subscription.is_cancelled:
                        subscription.backpressure_strategy = strategy

            logger.info(f"Updated global backpressure strategy to {strategy.value}")

    async def clear_all(self) -> None:
        """Clear all handlers and sessions"""
        async with self._lock:
            # Cancel all subscriptions
            for subscriptions in self._subscriptions.values():
                for subscription in subscriptions:
                    await subscription.stop_processing()
            self._subscriptions.clear()

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
